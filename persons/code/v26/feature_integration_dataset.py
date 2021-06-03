from PIL import Image
import pickle
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from types import SimpleNamespace
import numpy as np
import random # torch random transforms uses random
from .funcs import AutoSimpleNamespace

class FeatureIntegrationDatasetBase(data.Dataset):
    def __init__(self, root, nchars,ncolors, flag_type, nexamples = None, split = False,hsv = False,mean_image = None,existence_all = False):
        self.root = root
        self.nchars = nchars
        self.ncolors = ncolors
        self.flag_type = flag_type
        self.split = split
        self.splitsize=1000
        self.hsv = hsv
        self.mean_image = mean_image
        self.existence_all = existence_all
        if nexamples is None:
            # just in order to count the number of examples

            # all files recursively
            filenames=[os.path.join(dp, f) for dp, dn, fn in os.walk(root) for f in fn]

            images = [f for f in filenames if f.endswith('_img.jpg')]
            self.nexamples = len(images)
        else:
            self.nexamples = nexamples

    def get_root_by_index(self, index):
        if self.split:
            root= os.path.join(self.root,'%d' % (index//self.splitsize))
        else:
            root= self.root
        return root

    def get_raw_sample(self, index):
        root = self.get_root_by_index(index)
        data_fname=os.path.join(root,'%d_raw.pkl' % index)
        with open(data_fname, "rb") as new_data_file:
            raw_sample = pickle.load(new_data_file)
        return raw_sample

    def __len__(self):
        return self.nexamples

class FeatureIntegrationDataset(FeatureIntegrationDatasetBase):
    def __getitem__(self, index):
        root = self.get_root_by_index(index)
        fname=os.path.join(root,'%d_img.jpg' % index)
        if self.hsv:
            img = Image.open(fname).convert('HSV')
        else:
            img = Image.open(fname).convert('RGB')
        img = transforms.ToTensor()(img)
        img = 255*img
        if self.mean_image is not None:
            img -= self.mean_image
            img=img.float()
        sample = self.get_raw_sample(index)
        flag = sample.flag
        label_flag = sample.label_flag
        label_all = sample.label_all.astype(np.int)
        label_char = sample.label_char.astype(np.int)
        id = sample.id
        label_existence = np.any(sample.label_all,axis=1).astype(np.float)
        # from IPython.core.debugger import Pdb; ipdb = Pdb(); ipdb.set_trace()
        if self.existence_all:
            # also use color existence
            label_existence_colors = np.any(sample.label_all,axis=0).astype(np.float)
            label_existence = np.concatenate((label_existence,label_existence_colors),axis=0)
        label_all,label_char,label_flag,id,label_existence = map(
                torch.tensor, (label_all,label_char,label_flag,id,label_existence))
        if self.flag_type==1:
            char, color = flag
            char_ohe = torch.nn.functional.one_hot(torch.tensor(char), self.nchars)
            color_ohe = torch.nn.functional.one_hot(torch.tensor(color), self.ncolors)
            flag = torch.cat((char_ohe,color_ohe),dim=0)
        else:
            char = flag
            char_ohe = torch.nn.functional.one_hot(torch.tensor(char), self.nchars)
            flag = char_ohe
        flag = flag.float()
        label_task = label_flag
        label_task = label_task.view((-1))
        return img,label_all,label_char,label_flag,label_task, id, flag, label_existence


def inputs_to_struct_basic(inputs):
    image,label_all,label_char,label_flag,label_task, id, flag, label_existence = inputs
    sample = AutoSimpleNamespace(locals(), image,label_all,label_char,label_flag,label_task, id, flag, label_existence).tons()
    return sample


class FeatureIntegrationDatasetLabelAll(FeatureIntegrationDataset):

    def __getitem__(self, index):
        img,label_all,label_char,label_flag,label_task, id, flag, label_existence = super().__getitem__(index)

        label_task = label_all
        label_task = label_task.view((-1))
        return img,label_all,label_char,label_flag,label_task, id, flag, label_existence

