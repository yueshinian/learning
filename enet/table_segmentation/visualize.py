# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Ishrat Badami (badami.ishrat@gmail.com)
# ------------------------------------------------------------------------------

import json
import os

import numpy as np
from cv2 import cv2


def draw_table(img, points):
    """
    Draw detection contour on the image as shown in the assignment example
    :param img: Nd-array of row x column x channels
    :param points: Nd-array of contour coordinates (2D)
    :return: image with contour drawn
    """
    if len(points) > 0:
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img, [points], -1, (0, 255, 255), 3)
        for pt in points:
            img = cv2.circle(img, (pt[0][0], pt[0][1]), 3, (0, 0, 255), -1)
    return img


def downsize_img(img, scale_factor=2, adjust_factor=8):
    """
    Resize image given the scale factor
    :param img: input image
    :param scale_factor: (int) the factor to down size the image
    :param adjust_factor: New dimensions must be divisible by this factor.
    :return: down sized image
    """
    im_height = int(img.shape[0] / scale_factor / adjust_factor) * adjust_factor
    im_width = int(img.shape[1] / scale_factor / adjust_factor) * adjust_factor
    dim = (im_width, im_height)
    im_down = cv2.resize(img, dim).astype(np.uint8)
    return im_down


def scale_back_detections(detection_dir, save_detections=False):
    detection_file = os.path.join(detection_dir, 'detections.json')
    with open(detection_file, 'r') as file:
        detections = json.load(file)
    scale = detections['scale']
    if detections['scale'] > 1:
        # run for loop
        for k in detections.keys():
            if k != 'scale':
                detections[k] = (np.array(detections[k])*scale).tolist()
        # make scale factor = 1
        if save_detections:
            detections['scale'] = 1
            # dump detection at the same path
            with open(detection_file, 'w') as file:
                json.dump(detections, file, indent=4, sort_keys=True)
    return detections


def create_video_from_images(img_dir, video_out_path=''):
    command = 'ffmpeg -r 60 -f image2 -i ' + os.path.join(img_dir, 'frame_%04d.png') \
              + ' -c:v libx264 -pix_fmt yuv420p -vsync 0 ' + video_out_path
    os.system(command)


def create_video_from_detection(detections_dir, path_to_video, original_scale=False):
    print('Please have patience, this can take a while....')
    detections_file = os.path.join(detections_dir, 'detections.json')
    with open(detections_file, 'r') as file:
        detections = json.load(file)

    video_output_path = os.path.join(detections_dir, os.path.basename(path_to_video)[:-4] + '_detections.mp4')
    tmp_dir = os.path.join(detections_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    cap = cv2.VideoCapture(path_to_video)
    scale_factor = detections['scale']

    frame_no = 0
    while 1:
        ret, frame = cap.read()
        if ret:
            frame_no += 1
            points = np.array(detections[str(frame_no)])
            if original_scale:
                points = points*scale_factor
            else:
                frame = downsize_img(frame, scale_factor)
            img = draw_table(frame, points)

            file_path = os.path.join(tmp_dir, 'frame_' + "%04d" % frame_no + '.png')
            cv2.imwrite(file_path, img)
        else:
            break

    # create_video from images
    create_video_from_images(tmp_dir, video_output_path)
    # delete temporary image folder
    os.system('rm -r ' + tmp_dir)

    print('Generated video is saved at ', video_output_path)


if __name__ == "__main__":
    root_dir = './results'
    video_path = './dataset/videos/Baccarat.mp4'
    video_name = os.path.basename(video_path)[:-4]
    detections_dir_path = os.path.join(root_dir, video_name)

    create_video_from_detection(detections_dir_path, video_path, original_scale=False)
