# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Ishrat Badami (badami.ishrat@gmail.com)
# ------------------------------------------------------------------------------

import csv
import time

import numpy as np
import torch
from cv2 import cv2
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

MEAN = [0.485, 0.456, 0.406]  # cityscape pretrained model mean
STD = [0.229, 0.224, 0.225]  # cityscape pretrained model std


class VideoDataset(Dataset):
    """
    This class captures the video and generates a dataset object ready to use with pytorch model.
    """
    def __init__(self, path_to_video, scale_factor=1):
        """
        Args:
            path_to_video (string): file path to video or video streaming device id.
            scale_factor (int): Factor with which to scale down the image
            ### Note: the device id is not tested.
        """
        #  the model in use downscales the input by 8, hence the input requires to be multiple of 8
        self.adjust_factor = 8
        self.video_path = path_to_video
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.scale_factor = scale_factor
        # instead of passing the video_path pass the streaming device id/camera id
        self.cap = cv2.VideoCapture(self.video_path)

        # Check if camera opened successfully
        if not self.cap.isOpened():
            print("Error opening video stream or file")
        self.fps = np.ceil(self.cap.get(cv2.CAP_PROP_FPS)).astype(int)
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.mean = MEAN  # cityscape pretrained model mean
        self.std = STD  # cityscape pretrained model std

        # update height and width
        self.width = int(frame_width / self.scale_factor / self.adjust_factor) * self.adjust_factor
        self.height = int(frame_height / self.scale_factor / self.adjust_factor) * self.adjust_factor

    def __len__(self):
        # for some reason the number of frames per second is wrongly
        # calculated by opencv and hence the empty image is
        # returned in __getitem__ function when ret is False
        self.len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self.len

    def __getitem__(self, idx):
        ret, frame = self.cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (self.width, self.height))
            sample = self.input_transform(frame_resized)
            original = frame_resized
        else:
            sample = self.input_transform(np.zeros((self.height, self.width, 3), dtype=np.uint8))
            original = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return {'image': sample, 'original': original}

    def input_transform(self, frame):
        """
        Transform video frame to input tensor ready to be used on the gpu
        :param frame: numpy image array
        :return: half-precision pytorch tensor on gpu (if available)
        """
        tensor = transforms.ToTensor()(frame).to(
            device=self.device)  # 1 convert to float tensor, change to C x W x H and scale to  [0,1]
        tensor = tensor[[2, 1, 0], :]  # 2 convert BGR to RGB
        tensor = transforms.Normalize(self.mean, self.std)(tensor)  # 3 normalise image to given mean and variance
        return tensor.half()


class SegmentationDataset(Dataset):
    """Table top segmentation dataset for training."""

    def __init__(self, dataset_list_file, transforms_op=None):
        """
        Args:
            dataset_list_file (string): file path containing list of training image and ground truth labels.
            transforms_op: composition of image transformation
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mean = MEAN
        self.std = STD

        self.dataset_list_file = dataset_list_file
        self.sample_list = self.read_list()
        self.transforms = transforms_op

    def __len__(self):
        return len(self.sample_list)

    def read_list(self):
        """
        Read image training and label list
        :return:  list of paths of image and labels
        """
        with open(self.dataset_list_file, 'r') as f:
            reader = csv.reader(f)
            file_list = list(reader)

        files = []
        for item in file_list:
            image_path, label_path = item
            sample = {
                'image_path': image_path,
                'label_path': label_path,
            }
            files.append(sample)
        return files

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_path = self.sample_list[idx]
        img_name, label_name = sample_path['image_path'], sample_path['label_path']
        img = self.input_transform(cv2.imread(img_name))
        label = self.label_transform(cv2.imread(label_name, cv2.IMREAD_GRAYSCALE))
        sample = {'image': img, 'label': label}
        if self.transforms:
            sample = self.transforms(sample)

        return sample

    def input_transform(self, img):
        """
        Transform training image to tensor, ready to be used on the gpu
        :param img: numpy image array
        :return: pytorch tensor on gpu(if available)
        """
        tensor = transforms.ToTensor()(img).to(
            device=self.device)  # 1 convert to float tensor, change to C x W x H and
        # scale to  [0,1]
        tensor = tensor[[2, 1, 0], :]  # 2 convert BGR to RGB
        tensor = transforms.Normalize(self.mean, self.std)(tensor)  # 3 normalise image to given mean and variance
        return tensor

    def label_transform(self, label):
        """
        Transform label image array to tensor, ready to be used on the gpu
        :param label: numpy image array
        :return: pytorch tensor on gpu(if available)
        """
        tensor = transforms.ToTensor()(label).to(device=self.device)
        return tensor


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']

        h, w = img.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[:, top: top + new_h, left: left + new_w]
        label = label[:, top: top + new_h, left: left + new_w]

        return {'image': img, 'label': label}


def image_tensor_to_numpy(tensor):
    """
    Convert tensor back to numpy image array
    :param tensor: pytorch tensor on gpu
    :return: numpy image array
    """
    tensor = tensor.squeeze(0)
    img = tensor.detach().cpu().numpy().astype(np.uint8)
    return img


def label_tensor_to_numpy(tensor):
    """
    Convert prediction tensor to numpy image array
    :param tensor: pytorch tensor on gpu
    :return: numpy image array
    """
    tensor = torch.sigmoid(tensor) * 255
    # convert to numpy
    tensor = tensor.squeeze(0)
    img = tensor.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    return img


def detect_table(img):
    """

    :param img:
    :return:
    """
    # non-maxima suppression
    ret, thresh = cv2.threshold(img, 250, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = sorted(contours, key=cv2.contourArea)[-1:]
        table_contour = cv2.convexHull(largest_contour[0])
        return table_contour
    else:
        return []


if __name__ == "__main__":
    # debug the classes and it's methods
    training_data = './dataset/train.lst'
    video_path = 'dataset/videos/Baccarat.mp4'
    # ToTensor augmentation must be called in the end.
    training_dataset = SegmentationDataset(dataset_list_file=training_data,
                                           transforms_op=transforms.Compose([
                                               RandomCrop(512),
                                           ]))

    video_dataset = VideoDataset(path_to_video=video_path)
    # comment the appropriate dataloader to test the class of your choice
    # dataloader = DataLoader(training_dataset)
    dataloader = DataLoader(video_dataset)
    start = time.time()
    frames = 0
    for i, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        frames += 1
        mask = label_tensor_to_numpy(d['label'])
        image = image_tensor_to_numpy(d['image'])
        image = image[:, :, [2, 1, 0]]
        cv2.imshow("segmentation_mask", mask)
        cv2.imshow("image", image)
        cv2.waitKey(1)
    end = time.time()
    print("total_time_taken = ", (end - start))
    print("fps of HookMotion video = {} fps".format(video_dataset.fps))
    print("fps = ", frames / (end - start))
    cv2.destroyAllWindows()
