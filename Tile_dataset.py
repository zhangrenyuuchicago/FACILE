import collections
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import torch

np.random.seed(0)
torch.manual_seed(0)

class Tile_dataset(Dataset):
    def __init__(self, data_folder, label_file, transform=None, two_view=False):
        self.data_folder = data_folder
        self.label_file = label_file
        self.transform = transform
        self.two_view = two_view 
        self.img_paths = []
        self.img_labels = []

        if isinstance(label_file, str):
            fin = open(self.label_file, 'r') 
            all_lines = fin.readlines()
            fin.close()
            line = all_lines[0]

            if len(line.strip().split(' ')) == 2:
                print(f'label file has labels')
                self.with_label = True
            else:
                print(f'label file has no labels')
                self.with_label = False
            
            if self.with_label:
                for line in all_lines:
                    array = line.strip().split(' ')
                    if array[1] == 'label':
                        continue
                    sub_path, label = array[0], int(array[1])  
                    self.img_paths.append(os.path.join(data_folder, sub_path))
                    self.img_labels.append(label)
            else:
                for line in all_lines:
                    line = line.strip()
                    if line == 'path':
                        continue
                    sub_path = line
                    self.img_paths.append(os.path.join(data_folder, sub_path))
            print(f'tile number: {len(self.img_paths)}')
        elif isinstance(label_file, list):
            fin = open(self.label_file[0], 'r') 
            all_lines = fin.readlines()
            fin.close()
            line = all_lines[0]
            if len(line.strip().split(' ')) == 2:
                print(f'label file has labels')
                self.with_label = True
            else:
                print(f'label file has no labels')
                self.with_label = False
            
            for path in self.label_file:
                fin = open(path, 'r') 
                all_lines = fin.readlines()
                if self.with_label:
                    for line in all_lines:
                        array = line.strip().split(' ')
                        if array[1] == 'label':
                            continue
                        sub_path, label = array[0], int(array[1])  
                        self.img_paths.append(os.path.join(data_folder, sub_path))
                        self.img_labels.append(label)
                else:
                    for line in all_lines:
                        line = line.strip()
                        if line == 'path':
                            continue
                        sub_path = line
                        self.img_paths.append(os.path.join(data_folder, sub_path))
                fin.close()
            print(f'tile number: {len(self.img_paths)}')

    def __len__(self):
        return len(self.img_paths)
    
    def get_weight(self):
        if self.with_label:
            count = collections.Counter(self.img_labels)
            num_class = len(count)
            weight_lt = []
            for i in range(num_class):
                weight = 1./count[i]
                weight_lt.append(weight)
            weight_lt = np.array(weight_lt)
            sample_weight_lt = []
            for label in self.img_labels:
                weight = weight_lt[label]
                sample_weight_lt.append(weight)
            return sample_weight_lt
        else:
            return [1.0 for i in range(len(self.img_paths))]

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        if self.two_view == False:
            feat = self.transform(image)
            if self.with_label:
                label = self.img_labels[idx]
                return feat, label
            else:
                return feat
        else:
            feat0 = self.transform(image)
            feat1 = self.transform(image)
            if self.with_label:
                label = self.img_labels[idx]
                return (feat0, feat1), label
            else:
                return (feat0, feat1)
 


