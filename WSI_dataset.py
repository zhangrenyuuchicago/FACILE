import torch
import torch.utils.data
import numpy as np
import csv
import random
import collections
from PIL import Image
from PIL import ImageFile
import glob
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(0)

class WSI_dataset(torch.utils.data.Dataset):
    def __init__(self, tile_dir, label_file, transform, instance_num=5):
        self.instance_num = instance_num
        self.tile_dir = tile_dir
        self.slide2label = {}
        self.slide_id_lt = []
        self.slide2tile_path = {}

        with open(label_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)
            for line in reader:
                slide_id, label = line[0], int(line[1])
                self.slide2label[slide_id] = label
                self.slide_id_lt.append(slide_id)

        print(f'slide num in label file: {len(self.slide2label)}')

        for path in glob.glob(tile_dir + '/*/*/*.jpg'):
            basename = os.path.basename(path)
            array = basename.split('_')
            slide_id = array[0]
            if slide_id in self.slide2label:
                if slide_id in self.slide2tile_path:
                    self.slide2tile_path[slide_id].append(path)
                else:
                    self.slide2tile_path[slide_id] = [path]
        print(f'slide id num in dict: {len(self.slide2tile_path)}')
        self.transform = transform

    def __len__(self):
        return len(self.slide_id_lt)

    def size(self):
        return len(self.slide_id_lt)

    def get_weight(self):
        labels = []
        for slide_id in self.slide_id_lt:
            labels.append(self.slide2label[slide_id])
        count = collections.Counter(labels)
        num_class = len(count)
        weight_lt = []
        for i in range(num_class):
            weight = 1./count[i]
            weight_lt.append(weight)
        weight_lt = np.array(weight_lt)
        sample_weight_lt = []
        for slide_id in self.slide_id_lt:
            label = self.slide2label[slide_id]
            weight = weight_lt[label]
            sample_weight_lt.append(weight)

        return sample_weight_lt

    def __getitem__(self, idx):
        slide_id = self.slide_id_lt[idx]
        label = self.slide2label[slide_id]
        path_lt = self.slide2tile_path[slide_id]
        sub_path_lt = random.choices(path_lt, k=self.instance_num)
        feat_lt = []
        for path in sub_path_lt:
            image = Image.open(path)
            feat = self.transform(image)
            feat_lt.append(feat)
        set_feat0 = torch.stack(feat_lt)
        feat_lt = []
        for path in sub_path_lt:
            image = Image.open(path)
            feat = self.transform(image)
            feat_lt.append(feat)
        set_feat1 = torch.stack(feat_lt)

        return (set_feat0, set_feat1), label
