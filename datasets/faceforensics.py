"""Dataset utils for NN."""
import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile

import numpy as np
import ffmpeg
import skvideo.io
import pandas as pd
from skvideo.io import ffprobe
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class FaceForensicsDataset(Dataset):
    """UCF101 dataset for recognition. The class index start from 0.

    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
        test_sample_num： number of clips sampled from a video. 1 for clip accuracy.
    """

    def __init__(self, root_dir, clip_len, train=True, transforms_=None, test_sample_num=10, stride=2):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.train = train
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        self.data = []
        self.stride = stride
        self.class2idx = {'original_sequences': 0, 'Deepfakes': 1, 'Face2Face': 2, 'FaceSwap': 3, 'NeuralTextures': 4, 'DeepFakeDetection': 5}
        self.class_count = [0] * 6
        self.fake_count = 0

        for base, subdirs, files in os.walk(self.root_dir):
            # if 'DeepFakeDetection' in base:
            #     continue
            if len(files) < self.stride * self.clip_len:
                continue
            data = {}
            video = []
            files.sort()
            for i, f in enumerate(files):
                if f.endswith('.jpg'):
                    data_dict = {}
                    data_dict['frame'] = os.path.join(base, f)
                    data_dict['index'] = i
                    video.append(data_dict)
            data['video'] = video
            data['class'] = self.class2idx[base.split('/')[-2]]
            self.class_count[data['class']] += 1
            data['label'] = 0 if 'manipulated' in base else 1
            if data['label'] == 0:
                self.fake_count += 1
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def get_fake_count(self):
        return self.fake_count

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        video = self.data[idx]['video']
        label = self.data[idx]['label']
        sub_class = self.data[idx]['class']
        length = len(video)

        clip_start = random.randint(0, length - (self.clip_len * self.stride))
        clip = video[clip_start: clip_start + (self.clip_len * self.stride): self.stride]

        if self.transforms_:
            trans_clip = []
            # fix seed, apply the sample `random transformation` for all frames in the clip
            seed = random.random()
            for frame in clip:
                random.seed(seed)
                frame = Image.open(frame['frame'])
                frame = self.transforms_(frame)  # tensor [C x H x W]
                trans_clip.append(frame)
            # (T x C X H x W) to (C X T x H x W)
            clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
        else:
            clip = torch.tensor(clip)

        return clip, torch.tensor(int(label)), torch.tensor(sub_class)
