from PIL import Image
import pickle
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from types import SimpleNamespace
import numpy as np
import random # torch random transforms uses random

class EMNISTAdjDatasetBase(data.Dataset):
    def __init__(self, root, nclasses_existence,ndirections, nexamples = None, split = False,mean_image = None):
        self.root = root
        self.nclasses_existence = nclasses_existence
        self.ndirections = ndirections
        self.mean_image = mean_image
        self.split = split
        self.splitsize=1000
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

class EMNISTAdjDataset(EMNISTAdjDatasetBase):
    def __getitem__(self, index):
        root = self.get_root_by_index(index)
        fname=os.path.join(root,'%d_img.jpg' % index)
        img = Image.open(fname).convert('RGB')
        fname=os.path.join(root,'%d_seg.jpg' % index)
        seg = Image.open(fname).convert('RGB')
        img,seg = map(
                transforms.ToTensor(), (img,seg))
        img,seg = 255*img,255*seg
        if self.mean_image is not None:
            img -= self.mean_image
            img=img.float()
        sample = self.get_raw_sample(index)
        label_existence = sample.label_existence
        label_all = sample.label_ordered
        flag = sample.object_based.flag
        label_task = sample.object_based.label_task
        id = sample.id
        label_existence,label_all,label_task,id = map(
                torch.tensor, (label_existence,label_all,label_task,id))
        label_existence=label_existence.float()
        adj_type , char = flag
        adj_type_ohe = torch.nn.functional.one_hot(torch.tensor(adj_type), self.ndirections)
        char_ohe = torch.nn.functional.one_hot(torch.tensor(char), self.nclasses_existence)
        flag = torch.cat((adj_type_ohe,char_ohe),dim=0)
        flag = flag.float()
        label_task = label_task.view((-1))
        return img,seg,label_existence,label_all,label_task,id, flag

class EMNISTAdjDatasetNew(EMNISTAdjDatasetBase):
    def __getitem__(self, index):
        root = self.get_root_by_index(index)
        fname=os.path.join(root,'%d_img.jpg' % index)
        img = Image.open(fname).convert('RGB')
        fname=os.path.join(root,'%d_seg.jpg' % index)
        seg = Image.open(fname).convert('RGB')
        img,seg = map(
                transforms.ToTensor(), (img,seg))
        img,seg = 255*img,255*seg
        if self.mean_image is not None:
            img -= self.mean_image
            img=img.float()
        sample = self.get_raw_sample(index)
        label_existence = sample.label_existence
        label_all = sample.label_ordered
        flag = sample.flag
        label_task = sample.label_task
        id = sample.id
        label_existence,label_all,label_task,id = map(
                torch.tensor, (label_existence,label_all,label_task,id))
        label_existence=label_existence.float()
        adj_type , char = flag
        adj_type_ohe = torch.nn.functional.one_hot(torch.tensor(adj_type), self.ndirections)
        char_ohe = torch.nn.functional.one_hot(torch.tensor(char), self.nclasses_existence)
        flag = torch.cat((adj_type_ohe,char_ohe),dim=0)
        flag = flag.float()
        label_task = label_task.view((-1))
        return img,seg,label_existence,label_all,label_task,id, flag


def inputs_to_struct_basic(inputs):
    img,seg,label_existence,label_all,label_task,id, flag = inputs
    sample = SimpleNamespace()
    sample.image = img
    sample.seg = seg
    sample.label_occurence = label_existence
    sample.label_existence = label_existence
    sample.label_all = label_all
    sample.label_task = label_task
    sample.id = id
    sample.flag = flag
    return sample


def calc_label_adj_all(label_all,not_available_class,adj_type):
    obj_per_row = 6
    nclasses_existence = 47
    edge_class = nclasses_existence
    label_adj_all = not_available_class*np.ones(nclasses_existence,dtype=np.int64)
    for r,row in enumerate(label_all):
        for c,char in enumerate(row):
            if adj_type==0:
                # right
                if c==(obj_per_row-1):
                    res = edge_class
                else:
                    res = label_all[r,c+1]
            else:
                # left
                if c==0:
                    res = edge_class
                else:
                    res = label_all[r,c-1]
                    
            label_adj_all[char] = res    
    return label_adj_all


class EMNISTAdjDatasetLabelAdjAll(EMNISTAdjDataset):
    
    def __getitem__(self, index):
        img,seg,label_existence,label_all,label_task,id, flag = super().__getitem__(index)

        not_available_class=48
        # right-of of all characters
        label_adj_all = calc_label_adj_all(label_all.numpy(),not_available_class,adj_type = 0)
        label_adj_all = torch.tensor(label_adj_all)
        loss_weight = label_adj_all!=not_available_class
        label_task = label_adj_all
        loss_weight = loss_weight.float()
        return img,seg,label_existence,label_all,label_task,id, flag,loss_weight
        
def inputs_to_struct_label_adj_all(inputs):
    *basic_inputs,loss_weight = inputs
    sample = inputs_to_struct_basic(basic_inputs)
    sample.loss_weight = loss_weight
    return sample
    
