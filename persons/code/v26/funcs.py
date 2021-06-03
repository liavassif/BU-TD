import numpy as np
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=120)
import sys
import os
from pathlib import Path
home = str(Path.home())
sys.path.append(os.path.join(home,"code/other_py"))
from types import SimpleNamespace

import shutil
import six
import pickle
import time
import logging
import matplotlib as mpl
try:
    # only used here for determining matplotlib backend
    import v26.cfg as cfg
    use_gui = cfg.gpu_interactive_queue
except:
    use_gui = False
if use_gui:
    # when running without a graphics server (such as xserver) change to: mpl.use('AGG')
    mpl.use('TkAgg')
    # from mimshow import *
else:
    mpl.use('AGG')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
seed=0
torch.manual_seed(seed)
# if torch.cuda.is_available():
#     torch.backends.cudnn.benchmark = False
#     #Deterministic mode can have a performance impact, depending on your model. This means that due to the deterministic nature of the model, the processing speed (i.e. processed batch items per second) can be lower than when the model is non-deterministic.
#     torch.backends.cudnn.deterministic = True
# np.random.seed(seed)  # Numpy module.
#random.seed(seed)  # Python random module.

orig_relus = False # if True use the same number of ReLUs as in the original tensorflow implementation
orig_bn_eps = False # if True use the batch norm epsilon as in the original tensorflow implementation
if orig_bn_eps:
    bn_eps = 1e-3
else:
    bn_eps = 1e-5


import logging
logging.basicConfig(format=('%(asctime)s ' + '%(message)s'), level=logging.INFO)

logger = logging.getLogger(__name__)

#######################################
#    General functions
#######################################
def reset_logger():
    # there should be a better way than this...
    while len(logger.handlers)>0:
        logger.handlers.pop()

def setup_logger(fname):
    reset_logger()
    # Create the Handler for logging data to a file
    logger_handler = logging.FileHandler(fname)
    logger_handler.setLevel(logging.DEBUG)
    # Create a Formatter for formatting the log messages
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    # Add the Formatter to the Handler
    logger_handler.setFormatter(logger_formatter)

    # logger = logging.getLogger(__name__)
    # Add the Handler to the Logger
    logger.addHandler(logger_handler)
    return logger

def log_init(opts):
    logfname = opts.logfname
    if logfname is not None:
        model_dir = opts.model_dir
        setup_logger(os.path.join(model_dir,logfname))

def print_info(opts):
    import __main__ as main
    try:
        script_fname = main.__file__
        logger.info('Executing file %s' % script_fname)
    except:
        pass
    try:
        import subprocess
        result = subprocess.run(
            ['hostname', ''],
            stdout=subprocess.PIPE,
            shell=True)
        res = result.stdout.decode('utf-8')
        logger.info('Running on host: %s' % res)
    except:
        pass

    logger.info('PyTorch version: %s' % torch.__version__)
    logger.info('cuDNN enabled: %s' % torch.backends.cudnn.enabled)
    logger.info('model_opts: %s',str(opts))

def save_script(opts):
    model_dir = opts.model_dir
    if getattr(opts,'save_script',None) is None:
        save_script=True
    else:
        save_script=opts.save_script

    if save_script:
        import __main__ as main
        try:
            # copy the running script
            script_fname = main.__file__
            if False:
                # fix some permission problems we have on waic, do not copy mode bits
                dst=model_dir
                if os.path.isdir(dst):
                    dst = os.path.join(dst, os.path.basename(script_fname))
                shutil.copyfile(script_fname, dst)
            else:
                dst = shutil.copy(script_fname, model_dir)
                if opts.distributed:
                    # if distributed then also copy the actual script
                    script_base_fname = opts.module + '.py'
                    script_base_fname = os.path.join(os.path.dirname(script_fname),script_base_fname)
                    dst = shutil.copy(script_base_fname, model_dir)

            # copy funcs folder
            mods = [m.__name__ for m in sys.modules.values() if '.funcs' in m.__name__]
            if len(mods)>0:
                mods=mods[0]
                mods = mods.split('.')
                funcs_version=mods[0]
                # might want to use  dirs_exist_ok=True for Python>3.6
                dst = shutil.copytree(funcs_version, os.path.join(model_dir,funcs_version))
        except:
            pass

def pause_image(fig=None):
    plt.draw()
    plt.show(block=False)
    if fig == None:
        fig = plt.gcf()


#    fig.canvas.manager.window.activateWindow()
#    fig.canvas.manager.window.raise_()
    fig.waitforbuttonpress()

def redraw_fig(fig):
    if fig is None:
        return

    # ask the canvas to re-draw itself the next time it
    # has a chance.
    # For most of the GUI backends this adds an event to the queue
    # of the GUI frameworks event loop.
    fig.canvas.draw_idle()
    try:
        # make sure that the GUI framework has a chance to run its event loop
        # and clear any GUI events.  This needs to be in a try/except block
        # because the default implementation of this method is to raise
        # NotImplementedError
        fig.canvas.flush_events()
    except NotImplementedError:
        pass

def tocpu(inputs):
    return [inp.cpu() if inp is not None else None for inp in inputs]

def tonp(inputs):
    inputs = tocpu(inputs)
    return [inp.numpy() if inp is not None else None for inp in inputs]

def detach_tonp(outs):
    outs = [out.detach() if out is not None else None for out in outs]
    outs = tonp(outs)
    return outs

# ns* functions are the same as the usual functions for lists but for namespaces
def ns_tocpu(ns):
    dc = ns.__dict__
    for key in dc:
        if dc[key] is not None:
            dc[key]=dc[key].cpu()

def ns_tonp(ns):
    ns_tocpu(ns)
    dc = ns.__dict__
    for key in dc:
        if dc[key] is not None:
            dc[key]=dc[key].numpy()

def ns_detach_tonp(ns):
    dc = ns.__dict__
    for key in dc:
        if dc[key] is not None:
            dc[key]=dc[key].detach()
    ns_tonp(ns)

class AutoSimpleNamespace(SimpleNamespace):
# https://stackoverflow.com/questions/14903576/easily-dumping-variables-from-to-namespaces-dictionaries-in-python
    def __init__(self, env, *vs):
        vars(self).update(dict([(x, env[x]) for v in vs for x in env if v is env[x]]))

    def tons(self):
        dump_dict = self.__dict__
        ns = SimpleNamespace()
        ns_dict = ns.__dict__
        for key in dump_dict:
            ns_dict[key]=dump_dict[key]
        return ns

def preprocess(inputs):
    inputs=[inp.to(dev) for inp in inputs]
    return inputs

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            if b is not None: # if the batch is None do nothing (iterate over the next one)
                yield (self.func(b))

def print_total_parameters(model):
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{:,}".format(total_parameters))
    return total_parameters

def from_network(inputs,outs,module,inputs_to_struct,convert_to_np = True):
    if convert_to_np:
        inputs = tonp(inputs)
    else:
        inputs = tocpu(inputs)
    outs = module.outs_to_struct(outs)
    if convert_to_np:
        ns_detach_tonp(outs)
    else:
        ns_tocpu(outs)
    samples = inputs_to_struct(inputs)
    return samples,outs

# def get_mean_image(ds,inshape,inputs_to_struct):
#     mean_image= np.zeros(inshape)
#     for i,inputs in enumerate(ds):
#         inputs=tonp(inputs)
#         samples = inputs_to_struct(inputs)
#         mean_image = (mean_image*i + samples.image)/(i+1)
#         if i == nsamples_train - 1:
#             break
#     # mean_image = mean_image.astype(np.int)
#     return mean_image

# set stop_after to None if you want the accurate mean, otherwise set to the number of examples to process
def get_mean_image(dl,inshape,inputs_to_struct,stop_after = 1000):
    mean_image= np.zeros(inshape)
    nimgs = 0
    for inputs in dl:
        inputs=tonp(inputs)
        samples = inputs_to_struct(inputs)
        cur_bs = samples.image.shape[0]
        mean_image = (mean_image*nimgs + samples.image.sum(axis=0))/(nimgs+cur_bs)
        nimgs += cur_bs
        if stop_after and nimgs>stop_after:
            break
    return mean_image

def argmax_by_thresh(a):
    # returns 0 if the first argument is larger than the second one by at least some threshold
    THRESH = 0.01
    index_array = np.argsort(a)
    max_arg = index_array[-1]
    second_highest_arg = index_array[-2]
    max_val = a[max_arg]
    second_highest_val = a[second_highest_arg]
    if max_val>second_highest_val+THRESH:
        return max_arg
    else:
        return -1

def instruct(struct,key):
    return getattr(struct,key,None) is not None

#######################################
#    Model functions
#######################################

from enum import Enum, auto

class FlagAt(Enum):
    BU1 = auto()
    TD = auto()
    BU2 = auto()
    NOFLAG = auto()
    BU1_SIMPLE = auto()
    BU1_NOLAT = auto()

def setup_flag(opts):
    if opts.flag_at is FlagAt.BU2:
        opts.use_bu1_flag = False
        opts.use_td_flag = False
        opts.use_bu2_flag = True
        opts.use_td_loss = False
    elif (opts.flag_at is FlagAt.BU1) or (opts.flag_at is FlagAt.BU1_SIMPLE) or (opts.flag_at is FlagAt.BU1_NOLAT):
        opts.use_bu1_flag = True
        opts.use_td_flag = False
        opts.use_bu2_flag = False
        if (opts.flag_at is FlagAt.BU1_SIMPLE) or (opts.flag_at is FlagAt.BU1_NOLAT):
            opts.use_lateral_butd = False
            opts.use_lateral_tdbu = False
        if opts.flag_at is FlagAt.BU1_SIMPLE:
            opts.use_td_loss = False
    elif opts.flag_at is FlagAt.TD:
        opts.use_bu1_flag = False
        opts.use_td_flag = True
        opts.use_bu2_flag = False
    elif opts.flag_at is FlagAt.NOFLAG:
        opts.use_bu1_flag = False
        opts.use_td_flag = False
        opts.use_bu2_flag = False
        opts.use_td_loss = False

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout,kernel_size,stride=1,padding=1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin,bias=False)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1,bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

#conv2d_fun = nn.Conv2d
conv2d_fun = depthwise_separable_conv

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution"""
    return conv2d_fun(in_planes,out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3up(in_planes, out_planes, upsample_size=1):
    """upsample then 3x3 convolution"""
    layer = conv3x3(in_planes, out_planes)
    if upsample_size>1:
        layer = nn.Sequential(nn.Upsample(scale_factor=upsample_size, mode='bilinear',align_corners=False),
                layer)
    return layer

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,out_planes, kernel_size=1, stride=stride, bias=False)

class MultiLabelHead(nn.Module):

    def __init__(self, opts):
        super(MultiLabelHead, self).__init__()
        layers = []
        for k in range(len(opts.nclasses)):
            filters = opts.nclasses[k]
            k_layers = []
            infilters = opts.nfilters[-1]
            for i in range(opts.ntaskhead_fc-1):
                k_layers += [nn.Linear(infilters,infilters),opts.norm_fun(infilters,dims=1),opts.activation_fun()]

            # add last FC: plain
            k_layers += [nn.Linear(infilters,filters)]
            if len(k_layers)>1:
                k_layers = nn.Sequential(*k_layers)
            else:
                k_layers = k_layers[0]
            layers.append(k_layers)

        self.layers = nn.ModuleList(layers)

    def forward(self, inputs):
        x= inputs
        x = torch.flatten(x, 1)
        outs = []
        for layer in self.layers:
            y = layer(x)
            outs.append(y)
        return torch.stack(outs,dim=-1)

class OccurenceHead(nn.Module):

    def __init__(self, opts):
        super(OccurenceHead, self).__init__()
        filters = opts.nclasses_existence
        infilters = opts.nfilters[-1]
        self.fc = nn.Linear(infilters,filters)

    def forward(self, inputs):
        x= inputs
        x = torch.flatten(x, 1)
        x = self.fc(x)
#        x = nn.Sigmoid()(x)
        return x

class ImageHead(nn.Module):

    def __init__(self, opts):
        super(ImageHead, self).__init__()
        image_planes = opts.inshape[0]
        upsample_size = opts.strides[0]
        infilters = opts.nfilters[0]
        self.conv = conv3x3up(infilters, image_planes, upsample_size)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x

class Hadamard(nn.Module):

    def __init__(self, lateral_per_neuron, filters):
        super(Hadamard, self).__init__()
        self.lateral_per_neuron = lateral_per_neuron
        self.filters = filters
        # Create a trainable weight variable for this layer.
        if self.lateral_per_neuron:
            # not implemented....
            shape = input_shape[1:]
        else:
            shape=[self.filters,1,1]
        self.weights = nn.Parameter(torch.Tensor(*shape))  # define the trainable parameter
#        nn.init.constant_(self.weights.data, 1)  # init your weights here...
        nn.init.xavier_uniform_(self.weights)  # init your weights here...

    def forward(self, inputs):
        return inputs * self.weights

class SideAndComb(nn.Module):
    def __init__(self, lateral_per_neuron, filters, norm_layer, activation_fun):
        super(SideAndComb, self).__init__()
        self.side = Hadamard(lateral_per_neuron, filters)
        self.norm = norm_layer(filters)
        if not orig_relus:
            self.relu1 = activation_fun()
        self.relu2 = activation_fun()

    def forward(self, inputs):
        x, lateral = inputs

        side_val = self.side(lateral)
        side_val = self.norm(side_val)
        if not orig_relus:
            side_val = self.relu1(side_val)
        x = x + side_val
        x = self.relu2(x)
        return x

def NoNorm(num_channels, dims=2):
    return nn.Identity()

def GroupNorm(num_groups):
    f=lambda num_channels, dims=2: nn.GroupNorm(num_groups,num_channels)
    return f

# don't use population statistics as we share these batch norm layers across
# the BU pillars, where apparently they receive completely different statistics
# leading to wrong estimations when evaluating
def BatchNormNoStats(num_channels, dims=2):
    if dims==2:
        norm = nn.BatchNorm2d(num_channels,track_running_stats=False)
    else:
        norm = nn.BatchNorm1d(num_channels,track_running_stats=False)
    return norm

def BatchNorm(num_channels, dims=2):
    if dims==2:
        norm = nn.BatchNorm2d(num_channels,eps=bn_eps,track_running_stats=True)
    else:
        norm = nn.BatchNorm1d(num_channels,eps=bn_eps,track_running_stats=True)
    return norm

def InstanceNorm(num_channels, dims=2):
    if dims==2:
        norm = nn.InstanceNorm2d(num_channels,track_running_stats=False)
    else:
        norm = nn.Identity()
    return norm

def LocalResponseNorm():
    size = 2
    return nn.LocalResponseNorm(size)

class SideAndCombSharedBase():
    def __init__(self, lateral_per_neuron, filters):
        super(SideAndCombSharedBase, self).__init__()
        self.side = Hadamard(lateral_per_neuron, filters)
        self.filters = filters

class SideAndCombShared(nn.Module):
    def __init__(self, shared, norm_layer,activation_fun):
        super(SideAndCombShared, self).__init__()
        self.side = shared.side
        self.norm = norm_layer(shared.filters)
        if not orig_relus:
            self.relu1 = activation_fun()
        self.relu2 = activation_fun()

    def forward(self, inputs):
        x, lateral = inputs

        side_val = self.side(lateral)
        side_val = self.norm(side_val)
        if not orig_relus:
            side_val = self.relu1(side_val)
        x = x + side_val
        x = self.relu2(x)
        return x

class BasicBlockLatSharedBase():
    expansion = 1

    def __init__(self, inplanes, planes, stride, use_lateral):
        super(BasicBlockLatSharedBase, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        downsample = None
        if stride != 1 or inplanes != planes * BasicBlockLatSharedBase.expansion:
            downsample = conv1x1(inplanes, planes * BasicBlockLatSharedBase.expansion, stride)
        self.downsample = downsample
        self.stride = stride
        self.use_lateral = use_lateral
        if self.use_lateral:
            self.lat1 = SideAndCombSharedBase(lateral_per_neuron=False,filters=inplanes)
            self.lat2 = SideAndCombSharedBase(lateral_per_neuron=False,filters=planes)
            self.lat3 = SideAndCombSharedBase(lateral_per_neuron=False,filters=planes)
        self.inplanes = inplanes
        self.planes = planes

class BasicBlockLatShared(nn.Module):

    def __init__(self, shared, norm_layer, activation_fun):
        super(BasicBlockLatShared, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        planes = shared.planes
        self.conv1 = nn.Sequential(shared.conv1,norm_layer(planes),activation_fun())
        self.conv2 = nn.Sequential(shared.conv2,norm_layer(planes),activation_fun())
        if shared.downsample is not None:
            downsample = nn.Sequential(shared.downsample,norm_layer(planes))
        else:
            downsample = None
        self.downsample = downsample
        self.stride = shared.stride
        self.use_lateral = shared.use_lateral
        if self.use_lateral:
            self.lat1 = SideAndCombShared(shared.lat1,norm_layer,activation_fun)
            self.lat2 = SideAndCombShared(shared.lat2,norm_layer,activation_fun)
            self.lat3 = SideAndCombShared(shared.lat3,norm_layer,activation_fun)
        if orig_relus:
            self.relu = activation_fun()

    def forward(self, inputs):
        x, laterals_in = inputs
        if laterals_in is not None:
            lateral1_in, lateral2_in, lateral3_in = laterals_in
        laterals_out = []

        if laterals_in is not None:
            x = self.lat1((x,lateral1_in))
        laterals_out.append(x)
        inp = x

        x = self.conv1(x)
        if laterals_in is not None:
            x = self.lat2((x,lateral2_in))
        laterals_out.append(x)

        x = self.conv2(x)
        if laterals_in is not None:
            x = self.lat3((x,lateral3_in))
        laterals_out.append(x)

        if self.downsample is not None:
            identity = self.downsample(inp)
        else:
            identity = inp

        x = x + identity
        if orig_relus:
            x = self.relu(x)

        return x, laterals_out

class ResNetLatSharedBase():

    def __init__(self, opts):
        super(ResNetLatSharedBase, self).__init__()
        self.activation_fun = opts.activation_fun
        self.use_lateral = opts.use_lateral_tdbu #incoming lateral
        stride = opts.strides[0]
        filters = opts.nfilters[0]
        inplanes = opts.inshape[0]
        inshape=np.array(opts.inshape)
        self.use_bu1_flag = opts.use_bu1_flag
        if self.use_bu1_flag:
            lastAdded_shape = opts.inshape
            flag_scale = 2
            self.flag_shape = [-1,1, lastAdded_shape[1]//flag_scale, lastAdded_shape[2]//flag_scale]
            bu1_bot_neurons = int(np.product(self.flag_shape[1:]))
            self.bu1_bot_neurons = bu1_bot_neurons
            self.h_flag_bu = nn.Linear(opts.flag_size,bu1_bot_neurons)
            self.h_flag_bu_resized = nn.Upsample(scale_factor=flag_scale, mode='bilinear',align_corners=False)
            inplanes += 1

        inshapes=[]
        self.conv1 = conv2d_fun(inplanes, filters, kernel_size=7, stride=stride, padding=3,
                                bias=False)
        self.inplanes = filters
        inshape = np.array([filters, inshape[1]//stride, inshape[2]//stride])
        inshapes.append(inshape)
        # for BU2 use the final TD output as an input lateral. Note that this should be done even when not using laterals
        self.bot_lat = SideAndCombSharedBase(lateral_per_neuron=False,filters=filters)

        layers = [] # groups. each group has n blocks
        for k in range(1, len(opts.strides)):
            nblocks = opts.ns[k]
            stride = opts.strides[k]
            filters = opts.nfilters[k]
            layers.append(self._make_layer(filters, nblocks, stride=stride))
            inshape = np.array([filters, inshape[1]//stride, inshape[2]//stride])
            inshape_lst=[]
            for _ in range(nblocks):
                inshape_lst.append(inshape)
            inshapes.append(inshape_lst)

        self.alllayers = layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        filters = opts.nfilters[-1]
        if self.use_lateral:
            self.top_lat = SideAndCombSharedBase(lateral_per_neuron=False,filters=filters)
        inshape = np.array([filters,1,1])
        inshapes.append(inshape)
        self.inshapes=inshapes

        self.use_bu2_flag = opts.use_bu2_flag
        if self.use_bu2_flag:
            top_filters = opts.nfilters[k]
            self.top_filters = top_filters
            self.h_flag_bu2 = nn.Linear(opts.flag_size,top_filters)
            self.h_top_bu2 = nn.Linear(top_filters*2,top_filters)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        layers.append(BasicBlockLatSharedBase(self.inplanes, planes, stride,self.use_lateral))
        self.inplanes = planes * BasicBlockLatSharedBase.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlockLatSharedBase(self.inplanes, planes,1,self.use_lateral))

        return layers

def get_laterals(laterals,layer_id,block_id):
    if laterals is None:
        return None
    if len(laterals)>layer_id:
        layer_laterals = laterals[layer_id]
        if type(layer_laterals)==list and len(layer_laterals)>block_id:
            return layer_laterals[block_id]
        else:
            return layer_laterals
    return None

def init_module_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

class ResNetLatShared(nn.Module):

    def __init__(self, opts, shared):
        super(ResNetLatShared, self).__init__()
        self.norm_layer = opts.norm_fun
        self.activation_fun = opts.activation_fun
        self.inshapes = shared.inshapes
        self.use_lateral = shared.use_lateral #incoming lateral
        filters = opts.nfilters[0]
        self.use_bu1_flag = opts.use_bu1_flag
        if self.use_bu1_flag:
            # flag at BU1. It is shared across all the BU towers
            self.h_flag_bu = nn.Sequential(shared.h_flag_bu,self.norm_layer(shared.bu1_bot_neurons, dims=1),self.activation_fun())
            self.flag_shape = shared.flag_shape
            self.h_flag_bu_resized = shared.h_flag_bu_resized

        self.conv1 = nn.Sequential(shared.conv1, self.norm_layer(filters), self.activation_fun())
        self.bot_lat = SideAndCombShared(shared.bot_lat,self.norm_layer, self.activation_fun)

        layers = []
        for shared_layer in shared.alllayers:
            layers.append(self._make_layer(shared_layer))

        self.alllayers = nn.ModuleList(layers)
        self.avgpool = shared.avgpool
        if self.use_lateral:
            self.top_lat = SideAndCombShared(shared.top_lat,self.norm_layer, self.activation_fun)

        if not instruct(opts,'use_top_flag'):
            use_top_flag = False
        else:
            use_top_flag  = opts.use_top_flag
        self.use_top_flag = use_top_flag
        if self.use_top_flag:
            # flag at BU2. It is not shared across the BU towers
            self.top_filters = shared.top_filters
            self.h_flag_bu2 = nn.Sequential(shared.h_flag_bu2, self.norm_layer(self.top_filters, dims=1),self.activation_fun())
            self.h_top_bu2 = nn.Sequential(shared.h_top_bu2,self.norm_layer(self.top_filters,dims=1),self.activation_fun())

        # TODO: this should only be called once for all shared instances...
        init_module_weights(self.modules())


    def _make_layer(self, blocks):
        norm_layer = self.norm_layer
        layers = []
        for shared_block in blocks:
            layers.append(BasicBlockLatShared(shared_block, norm_layer, self.activation_fun))

        return nn.ModuleList(layers)


    def forward(self, inputs):
        x, flags, laterals_in = inputs
        if self.use_bu1_flag:
            f = self.h_flag_bu(flags)
            f = f.view(self.flag_shape)
            f = self.h_flag_bu_resized(f)
            x = torch.cat((x, f), dim = 1)

        laterals_out = []
        x = self.conv1(x)
        lateral_layer_id = 0
        lateral_in = get_laterals(laterals_in,lateral_layer_id,None)
        if lateral_in is not None:
            x = self.bot_lat((x,lateral_in))
        laterals_out.append(x)

        for layer_id,layer in enumerate(self.alllayers):
            layer_lats_out = []
            for block_id,block in enumerate(layer):
                lateral_layer_id = layer_id+1
                cur_lat_in = get_laterals(laterals_in,lateral_layer_id,block_id)
                x, block_lats_out = block((x,cur_lat_in))
                layer_lats_out.append(block_lats_out)

            laterals_out.append(layer_lats_out)

        x = self.avgpool(x)
        lateral_in = get_laterals(laterals_in,lateral_layer_id+1,None)
        if self.use_lateral and lateral_in is not None:
            x = self.top_lat((x,lateral_in))

        if self.use_top_flag:
            flag_bu2 = self.h_flag_bu2(flags)
            flag_bu2 = flag_bu2.view((-1,self.top_filters,1,1))
            x = torch.cat((x, flag_bu2),dim=1)
            x = torch.flatten(x,1)
            x = self.h_top_bu2(x)
            x = x.view((-1,self.top_filters,1,1))

        laterals_out.append(x)

        return x,laterals_out

class BasicBlockTDLat(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, norm_layer, activation_fun, use_lateral):
        super(BasicBlockTDLat, self).__init__()
        if use_lateral:
            self.lat1 = SideAndComb(False,inplanes, norm_layer, activation_fun)
            self.lat2 = SideAndComb(False,inplanes, norm_layer, activation_fun)
            self.lat3 = SideAndComb(False,planes, norm_layer, activation_fun)
        self.conv1 = nn.Sequential(conv3x3(inplanes, inplanes),norm_layer(inplanes))
        self.relu1 = activation_fun()
        self.conv2 = nn.Sequential(conv3x3up(inplanes, planes, stride),norm_layer(planes))
        if orig_relus:
            self.relu2 = activation_fun()
        self.relu3 = activation_fun()
        upsample = None
        outplanes = planes * BasicBlockTDLat.expansion
        if stride != 1:
            upsample = nn.Sequential(
                nn.Upsample(scale_factor = stride, mode='bilinear',align_corners=False),
                conv1x1(inplanes, outplanes, stride=1),
                norm_layer(outplanes)
            )
        elif inplanes != outplanes:
            upsample = nn.Sequential(
                conv1x1(inplanes, outplanes, stride=1),
                norm_layer(outplanes)
            )

        self.upsample = upsample
        self.stride = stride

    def forward(self, inputs):
        x, laterals_in = inputs
        if laterals_in is not None:
            lateral1_in, lateral2_in, lateral3_in = laterals_in
        laterals_out = []

        if laterals_in is not None:
            x = self.lat1((x,lateral1_in))
        laterals_out.append(x)
        inp = x

        x = self.conv1(x)
        x = self.relu1(x)
        if laterals_in is not None:
            x = self.lat2((x,lateral2_in))
        laterals_out.append(x)

        x = self.conv2(x)
        if orig_relus:
            x = self.relu2(x)
        if laterals_in is not None:
            x = self.lat3((x,lateral3_in))
        laterals_out.append(x)

        if self.upsample is not None:
            identity = self.upsample(inp)
        else:
            identity = inp

        x = x + identity
        x = self.relu3(x)

        return x, laterals_out[::-1]

class ResNetTDLat(nn.Module):

    def __init__(self, opts):
        super(ResNetTDLat, self).__init__()
        block = BasicBlockTDLat
        self.use_lateral = opts.use_lateral_butd
        self.activation_fun = opts.activation_fun
        self.use_td_flag = opts.use_td_flag
        self.norm_layer = opts.norm_fun

        top_filters = opts.nfilters[-1]
        self.top_filters=top_filters
        self.inplanes = top_filters
        if opts.use_td_flag:
            self.h_flag_td = nn.Sequential(nn.Linear(opts.flag_size,top_filters),self.norm_layer(top_filters,dims=1),self.activation_fun())
            self.h_top_td = nn.Sequential(nn.Linear(top_filters*2,top_filters),self.norm_layer(top_filters,dims=1),self.activation_fun())

        upsample_size = opts.avg_pool_size # before avg pool we have 7x7x512
        self.top_upsample = nn.Upsample(scale_factor=upsample_size, mode='bilinear',align_corners=False)
#        if self.use_lateral:
#           self.top_lat = SideAndComb(lateral_per_neuron=False,filters=top_filters)

        layers = []
        for k in range(len(opts.strides)-1, 0, -1):
            nblocks = opts.ns[k]
            stride = opts.strides[k]
            filters = opts.nfilters[k-1]
            layers.append(self._make_layer(block, filters, nblocks, stride=stride))

        self.alllayers = nn.ModuleList(layers)
        filters = opts.nfilters[0]
        if self.use_lateral:
            self.bot_lat = SideAndComb(False,filters, self.norm_layer, self.activation_fun)
        self.use_final_conv = opts.use_final_conv
        if self.use_final_conv:
            # here we should have performed another convolution to match BU conv1, but
            # we don't, as that was the behaviour in TF. Unless use_final_conv=True
            conv1 = conv2d_fun(filters, filters, kernel_size=7, stride=1, padding=3,
                                    bias=False)
            self.conv1 = nn.Sequential(conv1, self.norm_layer(filters), self.activation_fun())

        init_module_weights(self.modules())


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self.norm_layer
        layers = []
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes, 1, norm_layer, self.activation_fun, self.use_lateral))
        layers.append(block(self.inplanes, planes, stride, norm_layer, self.activation_fun, self.use_lateral))
        self.inplanes = planes * block.expansion

        return nn.ModuleList(layers)

    def forward(self, inputs):
        bu_out, flag, laterals_in = inputs
        laterals_out = []

        if self.use_td_flag:
            top_td = self.h_flag_td(flag)
            top_td = top_td.view((-1,self.top_filters,1,1))
            top_td_embed = top_td
            h_side_top_td = bu_out
            top_td = torch.cat((h_side_top_td, top_td),dim=1)
            top_td = torch.flatten(top_td,1)
            top_td = self.h_top_td(top_td)
            top_td = top_td.view((-1,self.top_filters,1,1))
            x = top_td
        else:
            x = bu_out

        laterals_out.append(x)

        x = self.top_upsample(x)
        if laterals_in is None or not self.use_lateral:
            for layer in self.alllayers:
                layer_lats_out = []
                for block in layer:
                    x, block_lats_out = block((x,None))
                    layer_lats_out.append(block_lats_out)

                reverse_layer_lats_out = layer_lats_out[::-1]
                laterals_out.append(reverse_layer_lats_out)
        else:
            reverse_laterals_in = laterals_in[::-1]

            for layer,lateral_in in zip(self.alllayers,reverse_laterals_in[1:-1]):
                layer_lats_out = []
                reverse_lateral_in=lateral_in[::-1]
                for block,cur_lat_in in zip(layer,reverse_lateral_in):
                    reverse_cur_lat_in = cur_lat_in[::-1]
                    x, block_lats_out = block((x,reverse_cur_lat_in))
                    layer_lats_out.append(block_lats_out)

                reverse_layer_lats_out = layer_lats_out[::-1]
                laterals_out.append(reverse_layer_lats_out)

            lateral_in = reverse_laterals_in[-1]
            x = self.bot_lat((x,lateral_in))

        if self.use_final_conv:
            x = self.conv1(x)
        laterals_out.append(x)

        outs = [x,laterals_out[::-1]]
        if self.use_td_flag:
            outs+=[top_td_embed,top_td]
        return outs

class BUModel(nn.Module):

    def __init__(self, opts):
        super(BUModel, self).__init__()
        bu_shared = ResNetLatSharedBase(opts)
        self.trunk=ResNetLatShared(opts,bu_shared)

    def forward(self, inputs):
        trunk_out, laterals_out= self.trunk(inputs)
        return trunk_out, laterals_out

class TDModel(nn.Module):

    def __init__(self, opts):
        super(TDModel, self).__init__()
        self.trunk = ResNetTDLat(opts)

    def forward(self, inputs):
        td_outs = self.trunk(inputs)
        return td_outs

class BUTDModel(nn.Module):

    def forward(self, inputs):
        samples = self.inputs_to_struct(inputs)
        images = samples.image
        flags = samples.flag
        model_inputs = [images,flags, None]
        bu_out, bu_laterals_out = self.bumodel1(model_inputs)
        if self.use_bu1_loss:
            occurence_out = self.occhead(bu_out)
        else:
            occurence_out = None
        model_inputs = [bu_out, flags]
        if self.use_lateral_butd:
            model_inputs += [bu_laterals_out]
        else:
            model_inputs += [None]
        td_outs = self.tdmodel(model_inputs)
        td_out, td_laterals_out, *td_rest = td_outs
        if self.use_td_loss:
            td_head_out = self.imagehead(td_out)
        model_inputs = [images, flags]
        if self.use_lateral_tdbu:
            model_inputs += [td_laterals_out]
        else:
            # when not using laterals we only use td_out as a lateral
            model_inputs += [[td_out]]
        bu2_out, bu2_laterals_out = self.bumodel2(model_inputs)
        task_out = self.taskhead(bu2_out)
        outs = [occurence_out, task_out, bu_out, bu2_out]
        if self.use_td_loss:
            outs+= [td_head_out]
        if self.tdmodel.trunk.use_td_flag:
            td_top_embed, td_top = td_rest
            outs+= [td_top_embed]
        return outs

    def outs_to_struct(self,outs):
        occurence_out, task_out, bu_out, bu2_out, *rest = outs
        outs_ns = SimpleNamespace(occurence=occurence_out,task=task_out,bu=bu_out,bu2=bu2_out)
        if self.use_td_loss:
            td_head_out, *rest = rest
            outs_ns.td_head = td_head_out
        if self.tdmodel.trunk.use_td_flag:
            td_top_embed = rest[0]
            outs_ns.td_top_embed = td_top_embed
        return outs_ns

class BUTDModelShared(BUTDModel):

    def __init__(self, opts):
        super(BUTDModelShared, self).__init__()
        self.use_bu1_loss = opts.use_bu1_loss
        if self.use_bu1_loss:
            self.occhead = OccurenceHead(opts)
        self.taskhead = MultiLabelHead(opts)
        self.use_td_loss = opts.use_td_loss
        if self.use_td_loss:
            self.imagehead=ImageHead(opts)
        bu_shared = ResNetLatSharedBase(opts)
        self.bumodel1=ResNetLatShared(opts,bu_shared)
        opts.use_top_flag = opts.use_bu2_flag
        self.bumodel2=ResNetLatShared(opts,bu_shared)

        pre_top_shape = bu_shared.inshapes[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
#        opts.avg_pool_size = (7,14)
        self.tdmodel=TDModel(opts)
        self.use_bu1_flag = opts.use_bu1_flag
        self.use_lateral_butd = opts.use_lateral_butd
        self.use_lateral_tdbu = opts.use_lateral_tdbu
        self.inputs_to_struct = opts.inputs_to_struct

class BUTDModelDuplicate(BUTDModel):
    def __init__(self, opts):
        super(BUTDModelDuplicate, self).__init__()
        self.occhead = OccurenceHead(opts)
        self.taskhead = MultiLabelHead(opts)
        self.use_td_loss = opts.use_td_loss
        if self.use_td_loss:
            self.imagehead=ImageHead(opts)
        bu_shared = ResNetLatSharedBase(opts)
        self.bumodel1=ResNetLatShared(opts,bu_shared)
        opts.use_top_flag = opts.use_bu2_flag # TODO: fix this as this will not work in duplicate
        self.bumodel2=self.bumodel1

        pre_top_shape = bu_shared.inshapes[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
#        opts.avg_pool_size = (7,14)
        self.tdmodel=TDModel(opts)
        self.use_bu1_flag = opts.use_bu1_flag
        self.use_lateral_butd = opts.use_lateral_butd
        self.use_lateral_tdbu = opts.use_lateral_tdbu
        self.inputs_to_struct = opts.inputs_to_struct

class BUTDModelSeparate(BUTDModel):
    def __init__(self, opts):
        super(BUTDModelSeparate, self).__init__()
        self.occhead = OccurenceHead(opts)
        self.taskhead = MultiLabelHead(opts)
        self.use_td_loss = opts.use_td_loss
        if self.use_td_loss:
            self.imagehead=ImageHead(opts)
        bu_shared1 = ResNetLatSharedBase(opts)
        self.bumodel1=ResNetLatShared(opts,bu_shared1)
        opts.use_top_flag = opts.use_bu2_flag
        bu_shared2 = ResNetLatSharedBase(opts)
        self.bumodel2=ResNetLatShared(opts,bu_shared2)

        pre_top_shape = bu_shared1.inshapes[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
#        opts.avg_pool_size = (7,14)
        self.tdmodel=TDModel(opts)
        self.use_bu1_flag = opts.use_bu1_flag
        self.use_lateral_butd = opts.use_lateral_butd
        self.use_lateral_tdbu = opts.use_lateral_tdbu
        self.inputs_to_struct = opts.inputs_to_struct

class BUModelSimple(nn.Module):

    def __init__(self, opts):
        super(BUModelSimple, self).__init__()
        self.occhead = OccurenceHead(opts)
        self.taskhead = MultiLabelHead(opts)
        self.bumodel=BUModel(opts)
        pre_top_shape = self.bumodel.trunk.inshapes[-2][-1]
        opts.avg_pool_size = tuple(pre_top_shape[1:].tolist())
        self.inputs_to_struct = opts.inputs_to_struct

    def forward(self, inputs):
        samples = self.inputs_to_struct(inputs)
        images = samples.image
        flags = samples.flag
        model_inputs = [images,flags, None]
        bu_out, _ = self.bumodel(model_inputs)
        occurence_out = self.occhead(bu_out)
        task_out = self.taskhead(bu_out)
        return occurence_out, task_out, bu_out

    def outs_to_struct(self,outs):
        occurence_out, task_out, bu_out = outs
        outs_ns = SimpleNamespace(occurence=occurence_out,task=task_out,bu=bu_out)
        return outs_ns

##########################################
#    Baseline functions - not really used
##########################################
class BUModelRawOcc(nn.Module):

    def __init__(self, opts):
        super(BUModelRawOcc, self).__init__()
        self.occhead = OccurenceHead(opts)
        self.bumodel=BUModelRaw(self.occhead,opts)

    def forward(self, inputs):
        images, labels, flags, segs, y_adjs, ids = inputs
        model_inputs = images
        task_out = self.bumodel(model_inputs)
        return task_out

class BUModelRaw(nn.Module):

    def __init__(self, head, opts):
        super(BUModelRaw, self).__init__()
        self.trunk = ResNet(BasicBlock,opts)
        self.head = head

    def forward(self, inputs):
        x= self.trunk(inputs)
        x = self.head(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs):
        x = inputs

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, opts):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        stride = opts.strides[0]
        filters = opts.nfilters[0]
        inplanes = opts.inshape[0]
        self.conv1 = nn.Conv2d(inplanes, filters, kernel_size=7, stride=stride, padding=3,
                                bias=False)
        self.inplanes = filters
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        layers = []
        for k in range(1, len(opts.strides)):
            nblocks = opts.ns[k]
            stride = opts.strides[k]
            filters = opts.nfilters[k]
            layers.append(self._make_layer(block, filters, nblocks, stride=stride))

        self.alllayers = nn.ModuleList(layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer, self.activation_fun))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer, self.activation_fun))

        return nn.ModuleList(layers)

    def forward(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #        x = self.maxpool(x)

        for layer in self.alllayers:
            for block in layer:
                x = block(x)

        x = self.avgpool(x)

        return x


#######################################
#    Train functions
#######################################
def get_multi_gpu_learning_rate(learning_rates_mult,num_gpus,scale_batch_size,ubs):
    # In pytorch gradients are summed across multi-GPUs (and not averaged) so
    # there is no need to change the learning rate when changing from a single GPU to multiple GPUs.
    # However when increasing the batch size (not because of multi-GPU, i.e. when scale_batch_size>1),
    # we need to increase the learning rate as usual
    clr = False
    if clr:
        learning_rates_mult *= scale_batch_size
    else:
        if ubs > 1:
            warmup_epochs = 5
            initial_lr = np.linspace(
                learning_rates_mult[0]/num_gpus, scale_batch_size * learning_rates_mult[0], warmup_epochs)
            learning_rates_mult = np.concatenate((initial_lr, scale_batch_size * learning_rates_mult))
    return learning_rates_mult

def save_model_and_md(model_fname,metadata,epoch, opts):
    tmp_model_fname = model_fname + '.tmp'
    logger.info('Saving model to %s' % model_fname)
    torch.save({
        'epoch': epoch,
        'model_state_dict': opts.model.state_dict(),
        'optimizer_state_dict': opts.optimizer.state_dict(),
        'scheduler_state_dict': opts.scheduler.state_dict(),
        'metadata': metadata,
        }, tmp_model_fname)
    os.rename(tmp_model_fname, model_fname)
    logger.info('Saved model')

def load_model(opts,model_latest_fname,gpu=None):
    if gpu is None:
        checkpoint = torch.load(model_latest_fname)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu)
        checkpoint = torch.load(model_latest_fname, map_location=loc)
    # checkpoint = torch.load(model_latest_fname)
    opts.model.load_state_dict(checkpoint['model_state_dict'])
    opts.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    opts.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint

class MeasurementsBase():
    '''a class for measuring train/test statistics such as accuracy'''
    def __init__(self, opts):
        self.opts = opts
        # self.reset()
        self.names = ['Loss']

    def init_results(self):
        self.n_measurements=len(self.names)
        epochs = 1
        self.results = np.full((epochs,self.n_measurements),np.nan)


    # update metrics for current batch and epoch (cumulative)
    # here we update the basic metric (loss). Subclasses should also call update_metric()
    def update(self, inputs, outs, loss):
        cur_batch_size = inputs[0].shape[0]
        self.loss += loss * cur_batch_size
        self.nexamples += cur_batch_size
        self.metrics_cur_batch = [loss * cur_batch_size]
        self.cur_batch_size  = cur_batch_size

    # update next metric for current batch
    def update_metric(self, metric, batch_sum):
        self.metrics_cur_batch += [batch_sum]
        metric += batch_sum

    def get_history(self):
        return np.array(self.metrics)/self.nexamples

    # store the epoch's metrics
    def add_history(self,epoch):
        if epoch+1>len(self.results):
            more = np.full(((epoch+1),self.n_measurements),np.nan) # resize array by 2
            self.results = np.concatenate((self.results,more))
        self.results[epoch,:] = self.get_history()

    def print_data(self,data):
        str=''
        for name,metric in zip (self.names, data):
            str+='{}: {:.2f}, '.format(name,metric)
        str=str[:-2]
        return str

    def print_batch(self):
        data = np.array(self.metrics_cur_batch)/self.cur_batch_size
        return self.print_data(data)

    def print_epoch(self):
        data = self.get_history()
        return self.print_data(data)

    def plot(self,fig, subplots_axes):
        n_measurements=len(self.metrics)
        for measurementi,name in enumerate(self.names):
            #ax = subplot(1, n_measurements, measurementi + 1)
            ax = subplots_axes[measurementi]
            ax.plot(self.results[:, measurementi])
            ax.set_title(name)

    def reset(self):
        self.loss = np.array(0.0)
        self.nexamples = 0
        self.metrics = [self.loss]

    def add_name(self, name):
        self.names += [name]

def set_datasets_measurements(datasets,measurements_class,model_opts,model):
    for the_dataset in datasets:
        the_dataset.create_measurement(measurements_class,model_opts,model)

class DatasetInfo():
    '''encapsulates a (train/test/validation) dataset with its appropriate train or test function and Measurement class'''
    
    def __init__(self,istrain,ds,nbatches,name,checkpoints_per_epoch=1):
        self.dataset = ds
        self.nbatches = nbatches
        self.istrain = istrain
        self.checkpoints_per_epoch = checkpoints_per_epoch
        if self.istrain and checkpoints_per_epoch>1:
            # when checkpoints_per_epoch>1 we make each epoch smaller
            self.nbatches = self.nbatches//checkpoints_per_epoch
        if istrain:
            self.batch_fun = train_step
        else:
            self.batch_fun = test_step
        self.name = name
        self.dataset_iter = None
        self.needinit = True

    def create_measurement(self,measurements_class,model_opts,model):
        self.measurements = measurements_class(model_opts,model)

    def reset_iter(self):
        self.dataset_iter = iter(self.dataset)

    def do_epoch(self,epoch,opts):
        logger.info(self.name)
        nbatches_report = 10
        aborted = False
        self.measurements.reset()
        cur_batches = 0
        if self.needinit or self.checkpoints_per_epoch==1:
            self.reset_iter()
            self.needinit = False
            if self.istrain and opts.distributed:
                opts.train_sampler.set_epoch(epoch)
                # TODO: when aborted save cur_batches. next, here do for loop and pass over cur_batches
                # and use train_sampler.set_epoch(epoch // checkpoints_per_epoch)
        start_time = time.time()
        for inputs in self.dataset_iter:
            cur_loss,outs = self.batch_fun(inputs,opts)
            with torch.no_grad():
                # so that accuracies calculation will not accumulate gradients
                self.measurements.update(inputs, outs, cur_loss.item())
            cur_batches += 1
            template = 'Epoch {} step {}/{} {} ({:.1f} estimated minutes/epoch)'
            if cur_batches % nbatches_report == 0:
                duration = time.time() - start_time
                start_time = time.time()
                # print(duration,self.nbatches)
                estimated_epoch_minutes = duration / 60 * self.nbatches / nbatches_report
                logger.info(
                    template.format(epoch + 1, cur_batches, self.nbatches,
                                    self.measurements.print_batch(),
                                    estimated_epoch_minutes))


            if True:
                if self.istrain and cur_batches > self.nbatches:
                    aborted = True
                    break

        if not aborted:
            self.needinit = True
        self.measurements.add_history(epoch)

def train_step(inputs,opts):
    opts.model.train()
    outs = opts.model(inputs)
    loss = opts.loss_fun(inputs, outs)
    opts.optimizer.zero_grad()
    loss.backward()
    opts.optimizer.step()
    return loss,outs

def test_step(inputs,opts):
    opts.model.eval()
    with torch.no_grad():
        outs = opts.model(inputs)
        loss = opts.loss_fun(inputs, outs)
    return loss,outs

def fit(opts,the_datasets):
    '''iterate over the datasets and train (or test) them'''
    if opts.first_node:
        logger.info('train_opts: %s',str(opts))
    optimizer=opts.optimizer
    scheduler  = opts.scheduler
    datasets_name = [dataset.name for dataset in the_datasets]

    nb_epochs=opts.EPOCHS

    model_dir = opts.model_dir

    model_ext='.pt'
    model_basename = 'model'
    model_latest_fname = model_basename + '_latest' + model_ext
    model_latest_fname = os.path.join(model_dir,model_latest_fname)

    if not instruct(opts,'save_details'):
        save_details = SimpleNamespace()
        # only save by maximum accuracy value
        save_details.optimum = -np.inf
        #save_details.save_cmp_fun = np.argmax
        save_details.save_cmp_fun = argmax_by_thresh
        save_details.epoch_save_idx = -1 # last metric: accuracy
        save_details.dataset_id=1 # from the test dataset
    else:
        save_details = opts.save_details

    optimum = save_details.optimum

    # restore saved model
    last_epoch = -1
    model_found = False
    if opts.load_model_if_exists:
        if os.path.exists(model_latest_fname):
            logger.info('Loading model: %s' % model_latest_fname)
            checkpoint = load_model(opts,model_latest_fname,opts.gpu)
            metadata= checkpoint['metadata']
            for the_dataset in the_datasets:
                the_dataset.measurements.results = metadata[the_dataset.name]
            optimum = metadata['optimum']
            last_epoch = metadata['epoch']
            if opts.distributed:
                # synchronize point so all distributed processes would have the same weights
                import torch.distributed as dist
                dist.barrier()
            logger.info('restored model with optimum %f' % optimum)
            logger.info('continuing from epoch: %d' % (last_epoch + 2))


    fig = None
    st_epoch = last_epoch + 1
    end_epoch = nb_epochs
    if instruct(opts,'abort_after_epochs') and opts.abort_after_epochs>0:
        end_epoch = st_epoch+opts.abort_after_epochs

    for epoch in range(st_epoch, end_epoch):
        if opts.first_node:
            logger.info('Epoch {} learning rate: {}'.format(epoch+1, optimizer.param_groups[0]['lr']))
        # if opts.distributed:
        #     opts.train_sampler.set_epoch(epoch)

        for the_dataset in the_datasets:
            the_dataset.do_epoch(epoch,opts)

        logger.info('Epoch {} done'.format(epoch+1))
        for the_dataset in the_datasets:
            logger.info(
                'Epoch {}, {} {}'.format(epoch + 1, the_dataset.name, the_dataset.measurements.print_epoch()))

        if epoch<nb_epochs-1:
            # learning rate scheduler
            scheduler.step()

        # save model
        if opts.first_node:
            # When using distributed data parallel, one optimization is to save the model in only one process, reducing write overhead. This is correct because all processes start from the same parameters and gradients are synchronized in backward passes, and hence optimizers should keep setting parameters to the same values 
            if opts.save_model:
                save_by_dataset = the_datasets[save_details.dataset_id]
                measurements=np.array(save_by_dataset.measurements.results)
                new_optimum = False
                epoch_save_value = measurements[epoch,save_details.epoch_save_idx]
                if save_details.save_cmp_fun(
                    [epoch_save_value, optimum]) == 0:
                    optimum = epoch_save_value
                    new_optimum = True
                    logger.info('New optimum: %f' % optimum)

                metadata = dict()
                metadata['epoch'] = epoch
                for the_dataset in the_datasets:
                    measurements=np.array(the_dataset.measurements.results)
                    metadata[the_dataset.name] = measurements
                metadata['optimum'] = optimum

                model_latest_fname = model_basename + '_latest' + model_ext
                model_latest_fname = os.path.join(model_dir,model_latest_fname)
                save_model_and_md(model_latest_fname,metadata,epoch,opts)
                if new_optimum:
                    model_fname = model_basename+'%d'% (epoch+1) + model_ext
                    model_fname = os.path.join(model_dir, model_fname)
                    shutil.copyfile(model_latest_fname, model_fname)
                    logger.info('Saved model to %s' % model_fname)
                    # save_model_and_md(model_fname,metadata,epoch,opts)


        # plot
        if opts.first_node:
            if epoch == st_epoch:
                n_measurements=len(the_datasets[0].measurements.metrics)
                fig, subplots_axes = plt.subplots(1, n_measurements,figsize=(5, 3))
                if n_measurements==1:
                    subplots_axes=[subplots_axes]
            else:
                plt.figure(fig.number)
                for ax in subplots_axes:
                    ax.clear()
            for the_dataset in the_datasets:
                the_dataset.measurements.plot(fig,subplots_axes)
            plt.legend(datasets_name)
            if use_gui:
                plt.show(block=False)
    #        draw()
            redraw_fig(fig)
            fig.savefig(os.path.join(model_dir, 'net-train.png'))
    logger.info('Done fit')
