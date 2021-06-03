from PIL import Image
import pickle
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from types import SimpleNamespace
from v26.funcs import AutoSimpleNamespace

class AvatarDetailsDatasetBase(data.Dataset):
    def __init__(self, root, nclasses_existence,nfeatures,nexamples = None, split = False,mean_image = None):
        self.root = root
        self.nclasses_existence = nclasses_existence
        self.nfeatures = nfeatures
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

class AvatarDetailsDataset(AvatarDetailsDatasetBase):
    def __getitem__(self, index):
        root = self.get_root_by_index(index)
        fname=os.path.join(root,'%d_img.jpg' % index)
        img = Image.open(fname).convert('RGB')
        fname=os.path.join(root,'%d_seg.jpg' % index)
        seg = Image.open(fname).convert('RGB')
        data_fname=os.path.join(root,'%d_data.pkl' % index)
        with open(data_fname, "rb") as new_data_file:
            lbl,flag,lbl_task,id = pickle.load(new_data_file)
        img,seg = map(
                transforms.ToTensor(), (img,seg))
        img,seg = 255*img,255*seg
        if self.mean_image is not None:
            img -= self.mean_image
            img=img.float()
        flag,lbl,lbl_task,id = map(
                torch.tensor, (flag,lbl,lbl_task,id))
        lbl=lbl.float()
        av_ids = flag[0]
        av_feats = flag[1]
        av_ids_ohe = torch.nn.functional.one_hot(av_ids, self.nclasses_existence)
        av_feats_ohe = torch.nn.functional.one_hot(av_feats, self.nfeatures)
        flag = torch.cat((av_ids_ohe,av_feats_ohe),dim=0)
        flag = flag.float()
        lbl_task = lbl_task.view((-1))
        return img,lbl, flag,seg,lbl_task,id

def inputs_to_struct_basic(inputs):
    img,lbl, flag,seg,lbl_task,id = inputs
    sample = SimpleNamespace()
    sample.image = img
    sample.label_existence = lbl
    sample.flag = flag
    sample.seg = seg
    sample.label_task = lbl_task
    sample.id = id
    return sample

# class AvatarDetailsDatasetPersonFeatures(AvatarDetailsDataset):

#     def __getitem__(self, index):
#         basic_inputs = super().__getitem__(index)
#         sample = self.get_raw_sample(index)
#         person_features = sample.object_based.preson_features
#         preson_features = torch.tensor(person_features)
#         return (*basic_inputs,person_features)


# def inputs_to_struct_person_features(inputs):
#     *basic_inputs,person_features = inputs
#     sample = inputs_to_struct_basic(basic_inputs)
#     sample.person_features = person_features
#     return sample

class AvatarDetailsDatasetLabelAll(AvatarDetailsDataset):

    def __getitem__(self, index):
        basic_inputs = super().__getitem__(index)
        sample = self.get_raw_sample(index)
        label_all = sample.label_all
        label_all = torch.tensor(label_all)
        loss_weight = sample.loss_weight 
        loss_weight = torch.tensor(loss_weight)
        loss_weight = loss_weight.float()
        return (*basic_inputs,label_all,loss_weight)


def inputs_to_struct_label_all(inputs):
    *basic_inputs,label_all,loss_weight = inputs
    sample = inputs_to_struct_basic(basic_inputs)
    sample.label_task = label_all
    sample.loss_weight = loss_weight
    return sample

class AvatarDetailsDatasetRaw(AvatarDetailsDatasetBase):
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
        flag = sample.object_based.flag
        label_task = sample.object_based.det_label
        label_all = sample.label_all
        loss_weight = sample.loss_weight
        person_features = sample.object_based.preson_features
        id = sample.id
        label_existence, flag,label_task,id,label_all, loss_weight,person_features = map(
                torch.tensor, (label_existence, flag,label_task,id,label_all, loss_weight,person_features))
        label_existence=label_existence.float()
        av_ids = flag[0]
        av_feats = flag[1]
        av_ids_ohe = torch.nn.functional.one_hot(av_ids, self.nclasses_existence)
        av_feats_ohe = torch.nn.functional.one_hot(av_feats, self.nfeatures)
        flag = torch.cat((av_ids_ohe,av_feats_ohe),dim=0)
        flag = flag.float()
        label_task = label_task.view((-1))
        loss_weight = loss_weight.float()
        return img,label_existence, flag,seg,label_task,id,label_all, loss_weight

class AvatarDetailsDatasetRawNew(AvatarDetailsDatasetBase):
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
        flag = sample.flag
        label_task = sample.label_task
        label_all = sample.label_all
        loss_weight = sample.loss_weight
        person_features = sample.person_features
        id = sample.id
        label_existence, flag,label_task,id,label_all, loss_weight,person_features = map(
                torch.tensor, (label_existence, flag,label_task,id,label_all, loss_weight,person_features))
        label_existence=label_existence.float()
        av_ids = flag[0]
        av_feats = flag[1]
        av_ids_ohe = torch.nn.functional.one_hot(av_ids, self.nclasses_existence)
        av_feats_ohe = torch.nn.functional.one_hot(av_feats, self.nfeatures)
        flag = torch.cat((av_ids_ohe,av_feats_ohe),dim=0)
        flag = flag.float()
        label_task = label_task.view((-1))
        loss_weight = loss_weight.float()
        return img,label_existence, flag,seg,label_task,id,label_all, loss_weight

def inputs_to_struct_raw(inputs):
    image,label_existence, flag,seg,label_task,id,label_all, loss_weight = inputs
    sample = AutoSimpleNamespace(locals(), image,label_existence, flag,seg,label_task,id,label_all, loss_weight).tons()
    return sample

def inputs_to_struct_raw_label_all(inputs):
    image,label_existence, flag,seg,label_task,id,label_all, loss_weight = inputs
    sample = AutoSimpleNamespace(locals(), image,label_existence, flag,seg,label_task,id,label_all, loss_weight).tons()
    sample.label_task = label_all
    return sample

class AvatarDetailsDatasetDummy(AvatarDetailsDatasetBase):
    def __init__(self, inshape,flag_size,nclasses_existence, nexamples):
        self.inshape = inshape
        self.flag_size = flag_size
        self.nexamples = nexamples
        self.nclasses_existence = nclasses_existence
        
    def __getitem__(self, index):
        img = torch.zeros(self.inshape,dtype=torch.float)
        seg = torch.zeros_like(img)
        label_existence = torch.zeros([self.nclasses_existence],dtype=torch.float)
        label_all = torch.ones((1),dtype=torch.int)
        loss_weight = torch.ones_like(label_all,dtype=torch.float)
        flag = torch.zeros(self.flag_size,dtype = torch.float)
        label_task = torch.zeros((1),dtype=torch.long)
        id = torch.tensor(index)
        label_task = label_task.view((-1))
        return img,label_existence, flag,seg,label_task,id,label_all, loss_weight
