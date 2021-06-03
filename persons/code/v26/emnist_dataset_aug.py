from PIL import Image
import pickle
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from types import SimpleNamespace
import numpy as np
import random # torch random transforms uses random
from emnist_dataset import EMNISTAdjDatasetBase

class EMNISTAdjDatasetAug(EMNISTAdjDatasetBase):
    def __init__(self, root, nclasses_existence, ndirections, nlimit_aug_examples=-1, aug = False, nexamples = None, split = False):
        super(EMNISTAdjDatasetAug, self).__init__(root, nclasses_existence,ndirections,nexamples, split)

        aug_degrees = 2
        aug_translate = (0.1, 0.05)
        aug_color_range = 0.2
        self.aug = aug
        self.nlimit_aug_examples = nlimit_aug_examples
        if self.aug:
            if self.nlimit_aug_examples>-1:
                self.cur_aug_choice_id = np.zeros(self.nexamples,dtype=int)

            self.aug_transforms = transforms.Compose([
                # if needed identical for segmentation, subclass and use get_params and perform on image, seg pair
                transforms.RandomAffine(aug_degrees, aug_translate, scale=None, shear=None, resample=False, fillcolor=0),
                transforms.ColorJitter(brightness=aug_color_range, contrast=0, saturation=0, hue=0)
                ])

    def __getitem__(self, index):
        root = self.get_root_by_index(index)
        fname=os.path.join(root,'%d_img.jpg' % index)
        img = Image.open(fname).convert('RGB')
        if self.aug:
            if self.nlimit_aug_examples>-1:
                # choose randomly but from a constant set of possibilities (for a given image id)
                choice_id = self.cur_aug_choice_id[index]
                choices_ids = range(index*self.nlimit_aug_examples,(index+1)*self.nlimit_aug_examples)
                seed = choices_ids[choice_id]
                if self.cur_aug_choice_id[index]<self.nlimit_aug_examples-1:
                    self.cur_aug_choice_id[index] +=1
                else:
                    self.cur_aug_choice_id[index] = 0
                 # transforms is affected by random
                random.seed(seed)
            img = self.aug_transforms(img)
        img = transforms.ToTensor()(img)
        img = 255*img
        return [img]
    
def inputs_to_struct_aug(inputs):
    img = inputs[0]
    sample = SimpleNamespace()
    sample.image = img
    return sample

class EMNISTAdjDatasetRandomInstruction(EMNISTAdjDatasetAug):
    def __init__(self, root, nclasses_existence, ndirections, nrows, obj_per_row, genmany, edge_class, aug = False, nexamples = None, split = False):
        super(EMNISTAdjDatasetRandomInstruction, self).__init__(root, nclasses_existence,ndirections,-1,aug,nexamples, split)
        self.nrows = nrows
        self.obj_per_row = obj_per_row
        self.edge_class = edge_class
        if genmany:
            # the dataset was generated assuming we avoid queries where the label is a border as this limits the number of valid images
            self.maxcol = self.obj_per_row-1
            self.mincol = 1
        else:
            self.maxcol = self.obj_per_row
            self.mincol = 0

    def __getitem__(self, index):
        basic_inputs = super().__getitem__(index)
        sample = self.get_raw_sample(index)
        lbl = sample.label_existence
        label_all = sample.label_ordered
        id = sample.id
        # randomly create an instruction and a label
        adj_type = int(np.random.rand()<.5)
        # adj_type = 0
        adj_type_ohe = torch.nn.functional.one_hot(torch.tensor(adj_type), self.ndirections)
        r = np.random.choice(self.nrows)
        if adj_type==0:
            # right
            c = np.random.choice(range(self.maxcol))
            if c==(self.obj_per_row-1):
                lbl_task = self.edge_class 
            else:
                lbl_task = label_all[r,c+1]
        else:
            # left
            c = np.random.choice(range(self.mincol,self.obj_per_row))
            if c==0:
                lbl_task = self.edge_class 
            else:
                lbl_task = label_all[r,c-1]
        char = label_all[r,c]
        char_ohe = torch.nn.functional.one_hot(torch.tensor(char), self.nclasses_existence)
        flag = torch.cat((adj_type_ohe,char_ohe),dim=0)
        flag = flag.float()
        lbl,lbl_task,id,label_all = map(
                torch.tensor, (lbl,lbl_task,id,label_all))
        lbl=lbl.float()
        lbl_task = lbl_task.view((-1))
        return (*basic_inputs, lbl, flag,lbl_task,id,label_all)


class EMNISTAdjDatasetRandomInstructionLimit(EMNISTAdjDatasetAug):
    def __init__(self, root, nclasses_existence, ndirections, nrows, obj_per_row, genmany, edge_class, nlimit_aug_examples=-1, aug = False, nexamples = None, split = False):
        super(EMNISTAdjDatasetRandomInstructionLimit, self).__init__(root, nclasses_existence,ndirections,nlimit_aug_examples,aug,nexamples, split)
        self.nrows = nrows
        self.obj_per_row = obj_per_row
        self.edge_class = edge_class
        # setup the random possibilities in advance since we want to have a constant pool of random examples
        nchoices = 1
        choices_possibilities = []
        if genmany:
            # the dataset was generated assuming we avoid queries where the label is a border as this limits the number of valid images
            maxcol = self.obj_per_row-1
            mincol = 1
            nchoices *= (self.obj_per_row-1)
        else:
            maxcol = self.obj_per_row
            mincol = 0
            nchoices *= self.obj_per_row
        choices_possibilities.append(np.arange(mincol,maxcol))
        nchoices *= nrows
        choices_possibilities.append(np.arange(0,nrows))
        nchoices *= ndirections
        choices_possibilities.append(np.arange(0,ndirections))
        self.nchoices = nchoices
        self.choices_possibilities = choices_possibilities
        self.choices_shapes = [len(lst) for lst in choices_possibilities]
        if self.nlimit_aug_examples>-1:
            # Naively tossing n (nlimit_aug_examples) balls into k (nchoices) bins usually results 
            # in choosing the same bin more than twice so we would get less than n unique examples.
            # That's why we setup everything in advance and choose a permutation of n examples
            choices_ids=np.zeros((self.nexamples,self.nlimit_aug_examples),dtype=int)
            prng = np.random.RandomState(0)
            for i in range(self.nexamples):
                choices_ids[i] = prng.choice(nchoices,self.nlimit_aug_examples,replace = False)
            self.choices_ids = choices_ids
            self.cur_choice_id = np.zeros(self.nexamples,dtype=int)

    def __getitem__(self, index):
        basic_inputs = super().__getitem__(index)
        sample = self.get_raw_sample(index)
        lbl = sample.label_existence
        label_all = sample.label_ordered
        id = sample.id
        if self.nlimit_aug_examples>-1:
            # assuming this is not run concurrently for a given index
            choice_id = self.cur_choice_id[index]
            choice_id = self.choices_ids[index,choice_id]
            if self.cur_choice_id[index]<self.nlimit_aug_examples-1:
                self.cur_choice_id[index] +=1
            else:
                self.cur_choice_id[index] = 0
        else:
            choice_id = random.choice(range(self.nchoices))
        c,r,adj_type = np.unravel_index(choice_id,self.choices_shapes)
        adj_type_ohe = torch.nn.functional.one_hot(torch.tensor(adj_type), self.ndirections)
        if adj_type==0:
            # right
            if c==(self.obj_per_row-1):
                lbl_task = self.edge_class 
            else:
                lbl_task = label_all[r,c+1]
        else:
            # left
            if c==0:
                lbl_task = self.edge_class 
            else:
                lbl_task = label_all[r,c-1]
        char = label_all[r,c]
        char_ohe = torch.nn.functional.one_hot(torch.tensor(char), self.nclasses_existence)
        flag = torch.cat((adj_type_ohe,char_ohe),dim=0)
        flag = flag.float()
        lbl,lbl_task,id,label_all = map(
                torch.tensor, (lbl,lbl_task,id,label_all))
        lbl=lbl.float()
        lbl_task = lbl_task.view((-1))
        return (*basic_inputs, lbl, flag,lbl_task,id,label_all)


def inputs_to_struct_data(inputs):
    *basic_inputs,lbl, flag,lbl_task,id,label_all = inputs
    sample = inputs_to_struct_aug(basic_inputs)
    sample.label_occurence = lbl
    sample.flag = flag
    sample.label_task = lbl_task
    sample.id = id
    sample.label_all = label_all
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


class EMNISTAdjDatasetLabelAdjAllAug(EMNISTAdjDatasetAug):
    
    def __getitem__(self, index):
        basic_inputs = super().__getitem__(index)
        sample = self.get_raw_sample(index)
        lbl = sample.label_existence
        label_all = sample.label_ordered

        not_available_class=48
        # right-of of all characters
        # label_adj_all = calc_label_adj_all(label_all,not_available_class,adj_type = 0)
        lbl_task = sample.label_adj_all
        loss_weight = sample.label_adj_all!=not_available_class
        id = sample.id
        flag = -1 # flag is unused. cannot have a None tensor however
        lbl,flag,lbl_task,id,label_all,loss_weight = map(
                torch.tensor, (lbl,flag,lbl_task,id,label_all,loss_weight))
        lbl=lbl.float()
        loss_weight = loss_weight.float()
        return (*basic_inputs, lbl, flag,lbl_task,id,label_all,loss_weight)
        
def inputs_to_struct_label_adj_all(inputs):
    *basic_inputs,loss_weight = inputs
    sample = inputs_to_struct_data(basic_inputs)
    sample.loss_weight = loss_weight
    return sample
    