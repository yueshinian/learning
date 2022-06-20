
#!/usr/bin/env python3
# encoding: utf-8import os
import argparse
import torch
import time

from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fast_scnn',
                    help='model name (default: fast_scnn)')
parser.add_argument('--dataset', type=str, default='citys',
                    help='dataset name (default: citys)')
parser.add_argument('--weights-folder', default='./weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str,
                    default='./datasets/citys/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./test_result', type=str,
                    help='path to save the predict result')

parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=False)

args = parser.parse_args()

'''
torch.save(model.state_dict(), "./save/model.pth") #保存模型
model = Model()
model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset])))
'''


def demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # image transform
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),#尺寸hw
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])#将图像转换为tensor向量，同时归一化
    image = Image.open(args.input_pic).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)#[C H W]  [N C H W] 
    model = get_fast_scnn(args.dataset, pretrained=True, root=args.weights_folder, map_cpu=args.cpu).to(device) 
    #新建模型model = FastSCNN(datasets[dataset].NUM_CLASS, **kwargs) 
    # #model.load_state_dict(torch.load(os.path.join(root, 'fast_scnn_%s.pth' % acronyms[dataset])))
    print('Finished loading model!')
    start = time.time()
    model.eval() #关闭dropout和BN
    with torch.no_grad():#不求导
        outputs = model(image)
    pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()#0维最大值
    mask = get_color_pallete(pred, args.dataset)
    outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '.png'
    mask.save(os.path.join(args.outdir, outname))
    end = time.time()
    print("Time used is: ", end - start)

if __name__ == '__main__':
    demo()
