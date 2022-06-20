import argparse
from ast import arg
from operator import mod
import os
from pickletools import optimize
from pyexpat import model
from random import shuffle
from statistics import mode
import string
from tabnanny import check 
import time
from charset_normalizer import from_path
from cv2 import mean
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from zmq import device

from segmentation.DDRNet_23_slim import get_seg_model
from thirdParty import CrossEntropy
from thirdParty import FullModel #calculate loss between output and label of two outputs
from thirdParty import get_confusion_matrix
from thirdParty import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description="train segmentation network")
    parser.add_argument('--seed', type=int, default=304) #固定随机种子，每次重新训练的随机初始权重相同
    parser.add_argument('--local_rank', type=int, default=-1) #分布式训练
    parser.add_argument('--resume', type=bool, default=False) #断点训练
    parser.add_argument('--weights-path', type=string, default='./weights/ImageNet.pth')
    parser.add_argument('--outdir', string, './outdir')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #固定随机初始权重
    if args.seed > 0:
        import random
        print('seeding with: ', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    #data
    crop_size = [1024, 1024]
    train_dataset = eval('datasets.'+'cityscapes')(
                        root='data',
                        list_path='list/cityscapes/train.lst',
                        num_samples=None,
                        num_classes=19,
                        multi_scale=True,
                        flip=True,
                        ignore_label=255,
                        base_size=2048,
                        crop_size=crop_size,
                        downsample_rate=1,
                        scale_factor=16)
    test_size = [1024, 1024]
    test_dataset = eval('datasets.'+'cityscapes')(
                        root='data',
                        list_path='list/cityscapes/val.lst',
                        num_samples=None,
                        num_classes=19,
                        multi_scale=False,
                        flip=False,
                        ignore_label=255,
                        base_size=2048,
                        crop_size=test_size,
                        downsample_rate=1)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last = True)

    gpus = []
    distributed = args.local_rank >= 0

    #build modle
    model = get_seg_model(pretrained=False)
    model = model.to(device=device)
    if distributed:
        batch_size = 8*len(gpus)
    else:
        batch_size = 8
    
    #loss
    criterion = CrossEntropy(255, train_dataset.class_weights)
    fullModel = FullModel(model, criterion)

    #optim
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, nesterov=False)

    #check point
    best_mIoU = 0
    last_epoch = 0
    if args.resume:
        check_point = torch.load(os.path.join(args.outdir, 'check_point.pth.tar'), map_location=device)
        best_mIoU = check_point['best_mIoU']
        last_epoch = check_point['epoch']
        dct = check_point['state_dict']
        model.load_state_dict({k.replace('model.', ''):v for k,v in check_point['state_dict'].items() if k.startswith('model.')})
        optimizer.load_state_dict(check_point['optimizer'])
    
    end_epoch = 10
    start  = time.time()
    for epoch in range(last_epoch, end_epoch):
        #train
        fullModel.train()
        for step, batch in enumerate(train_loader, 0):
            images, labels, _, _ = batch
            images = images.cuda()
            labels = labels.long().cuda()

            losses, _,  acc = fullModel(images, labels)
            loss = losses.mean()
            acc = acc.mean()
            fullModel.zero_grad()
            loss.backward()
            optimizer.step()
        
        #eval
        if epoch%10 == 0:
            fullModel.eval()
            ave_loss = AverageMeter()
            nums = 2
            confusion_matrix = np.zeros((19, 19, nums))
            with torch.no_grad():
                for idx, batch in enumerate(test_loader):
                    image, label, _, _ = batch
                    size = label.size()
                    image = image.cuda()
                    label = label.long().cuda()
                    losses_eval, pred, _ = fullModel(image, label)
                    for id, xImage in enumerate(pred):
                        xImage = F.interpolate(input=xImage, size=size[-2:], mode='bilinear', align_corners=True)
                        confusion_matrix[..., id] += get_confusion_matrix(
                            label,
                            xImage,
                            size,
                            19,
                            255
                        )
                    loss_evl = losses_eval.mean()
                    ave_loss.update(loss_evl)
                
                for i in range(nums):
                    pos = confusion_matrix[..., i].sum(1)
                    res = confusion_matrix[..., i].sum(0)
                    tp = np.diag(confusion_matrix[..., i])#true positive预测和label符合
                    IoU_array = (tp/ np.maximum(1.0, pos+res-tp))
                    mean_IoU = IoU_array.mean()
        if args.local_rank <= 0:
            torch.save({
                'epoch': epoch+1,
                'best_mIoU': best_mIoU,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.outdir, 'check_point.pth.tar'))
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model.state_dict(), os.path.join(args.outdir, 'best_path'))
            print('Loss: {:.3f}, mean_IoU: {:4.4f}, best_IoU: {4.4f}'.format(ave_loss.average(), mean_IoU, best_mIoU))

    if args.local_rank <= 0:
        torch.save(model.state_dict(), os.path.join(args.outdir, 'final_state.pth'))

if __name__ == '__main__':
    main()