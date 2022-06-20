import glob
import json
import os.path

import numpy as np
from cv2 import cv2


def extract_contour(path_to_images):

    for image_path in glob.glob(os.path.join(path_to_images, '*.png')):

        image = cv2.imread(image_path)
        ret, thresh = cv2.threshold(image[:, :, 2], 251, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        vertices = []
        for contour in contours:
            vertex = np.mean(contour, axis=0).astype(int)
            vertices.append(vertex)
        table_contour = np.array(vertices)

        table_contour = cv2.convexHull(table_contour)
        image_mask = create_segmentation_mask(image, table_contour)
        visualize(image, image_mask, table_contour)

        # save masks
        cv2.imwrite(image_path[:-4] + '_mask.png', image_mask)


def create_segmentation_mask(img, points):
    image_mask = np.zeros([img.shape[0], img.shape[1]], dtype=np.uint8)
    cv2.fillPoly(image_mask, [points], [255])
    return image_mask


def read_annotations(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return np.array(data['shapes'][0]['points'], dtype=np.int32)


def process_data(path_to_images):
    for image_path in glob.glob(os.path.join(path_to_images, '*.png')):
        image = cv2.imread(image_path)
        json_path = image_path[:-3]+'json'
        points = read_annotations(json_path)
        points = points[:, np.newaxis, :]
        image_mask = create_segmentation_mask(image, points)
        visualize(image, image_mask, points)

        # save masks
        mask_path = image_path.replace('image', 'gt')[:-4]+ '_mask.png'
        cv2.imwrite(mask_path, image_mask)


def visualize(image, mask, points):
    cv2.drawContours(image, [points], -1, (0, 255, 0), 3)

    cv2.imshow("image", image)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)


if __name__ == "__main__":
    path = "./dataset/examples"
    extract_contour(path)
    # path = "./dataset/images"
    # process_data(path)
