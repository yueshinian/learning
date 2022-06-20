# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Ishrat Badami (badami.ishrat@gmail.com)
# ------------------------------------------------------------------------------

import json
import os
import time

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from data_utils import VideoDataset, detect_table, image_tensor_to_numpy, label_tensor_to_numpy
from visualize import create_video_from_detection, create_video_from_images, draw_table, downsize_img
from model import DualResNet_hookmotion
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def prepare_model(path_to_model):
    """
    Read trained model and set it for evaluation
    :param path_to_model: (str) file path for saved model
    :return: model for inference
    """
    cfg = config
    cfg.defrost()
    cfg.MODEL.PRETRAINED = path_to_model

    model = DualResNet_hookmotion(cfg, pretrained=True)
    model = model.to(device=device)

    model.eval()
    model.half() # for faster inference
    return model


def load_data(path, scaling):
    video_dataset = VideoDataset(path, scaling)
    data_loader = DataLoader(video_dataset)
    return data_loader, video_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Infer table in the video')
    parser.add_argument('--result_dir', default="./results",  type=str, help='Path to results directory')
    parser.add_argument('--model_path', default="./pretrained_models/final_state.pth",  type=str, help='Path to model')
    parser.add_argument('--video_file', default="./dataset/videos/LuckyLadies.mp4", type=str, help='path to video')
    parser.add_argument('--visualize', default=False, type=bool, help='If true, shows the output during inference')
    parser.add_argument('--scaling', default=2, type=int, help='scaling down factor for input video frames')

    args = parser.parse_args()

    visualize = args.visualize
    result_dir = args.result_dir
    video_path = args.video_file
    output_path = os.path.join(result_dir, os.path.basename(video_path)[:-4])
    scale_factor = args.scaling

    model_path = args.model_path
    detection_file_path = os.path.join(output_path, 'detections.json')

    os.makedirs(output_path, exist_ok=True)

    if visualize:
        output_image_dir = os.path.join(output_path, 'images')
        os.makedirs(output_image_dir, exist_ok=True)

    net = prepare_model(model_path)
    dataloader, dataset = load_data(video_path, scale_factor)

    frames = 0
    time_total = 0
    detections = {'scale': scale_factor}
    with torch.no_grad():
        desc = 'inferring video stream'
        for i in tqdm(dataloader, total=(len(dataloader)), desc=desc):
            frames += 1

            start = time.time()  # start time calculation from model inference
            y = net(i['image'])

            mask = label_tensor_to_numpy(y[0])
            polygon = detect_table(mask)

            end = time.time()  # end time calculation after table segment detection
            time_total += end-start
            if len(polygon):
                detections[frames] = polygon.tolist()

                if visualize:
                    img = image_tensor_to_numpy(i['original'])
                    img = draw_table(img, polygon)
                    mask = draw_table(mask, polygon)
                    detection_frame = cv2.hconcat([img, mask])
                    detection_frame_resized = downsize_img(detection_frame)
                    file_path = os.path.join(output_image_dir, 'frame_' + "%04d" % frames + '.png')
                    cv2.imwrite(file_path, detection_frame)
                    cv2.imshow("prediction", detection_frame_resized)
                    cv2.waitKey(1)

    with open(detection_file_path, 'w') as file:
        json.dump(detections, file, indent=4)

    if visualize:
        output_video_path = os.path.join(output_path, os.path.basename(video_path))
        create_video_from_images(output_image_dir, output_video_path)

    end = time.time()
    print("total time taken in seconds = ", time_total)
    print("fps of HookMotion video = {} fps".format(dataset.fps))
    print("fps for inference = ", frames / time_total)

    if visualize:
        cv2.destroyAllWindows()
