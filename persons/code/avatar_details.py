# %% general initialization
import os
import v26.cfg as cfg
# running interactively uses a single GPU and plots the training results in a window
interactive_session = False
cfg.gpu_interactive_queue = interactive_session
if interactive_session:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from v26.funcs import *

ngpus_per_node = torch.cuda.device_count()
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--hyper', default=-1, type=int)
parser.add_argument('--only_cont', action='store_true')
parser.add_argument('--SGD', action='store_true')
parser.add_argument('-lr', default=1e-3 / 2, type=float)
parser.add_argument('-bs', default=10, type=int)
parser.add_argument('-wd', default=0.0001, type=float)
parser.add_argument('--checkpoints-per-epoch', default=1, type=int)
parser.add_argument('-e',
                    '--extended',
                    action='store_true',
                    help='Use the extended set instead of the sufficient set')
if interactive_session:
    sys.argv = ['']
args = parser.parse_args()
args.distributed = False
args.hyper_search = args.hyper > -1
if args.extended:
    base_tf_records_dir = 'extended'
else:
    base_tf_records_dir = 'sufficient'
if args.hyper_search:
    index = args.hyper
    import itertools
    if args.SGD:
        lrs = [0.001, 0.0001]
        bss = [10]
        wds = [0.0001, 0.0002]
        cmd_options = list(itertools.product(lrs, bss, wds))
        cmd_options = np.array(cmd_options)
    else:
        lrs = np.array([0.0005, 0.001, 0.002])
        bss = [10]
        wds = [0.0001, 0.0002]
        cmd_options = list(itertools.product(lrs, bss, wds))
        cmd_options = np.array(cmd_options)
    args.lr, args.bs, args.wd = cmd_options[index]
    args.bs = int(args.bs)

ENABLE_LOGGING = True
data_dir = '../data'
avatars_dir = os.path.join(data_dir, 'avatars')
new_avatars_dir = os.path.join(avatars_dir, 'samples')
base_samples_dir = os.path.join(new_avatars_dir, base_tf_records_dir)
data_fname = os.path.join(base_samples_dir, 'conf')
results_dir = os.path.join(data_dir, 'results')
flag_at = FlagAt.TD
# when True use a dummy dataset instead of a real one (for debugging)
dummyds = False

# %% load samples
if not dummyds:
    with open(data_fname, "rb") as new_data_file:
        nsamples_train, nsamples_test, nsamples_val, nfeatures, nclasses, nclasses_existence, ntypes, img_channels, IMAGE_SIZE = pickle.load(
            new_data_file)
else:
    nsamples_train = 200
    nsamples_test = 200
    nclasses_existence = 6
    nfeatures = 7
    ntypes = [9]
    IMAGE_SIZE = [224, 448]
    img_channels = 3
# %% dataset
inshape = (img_channels, *IMAGE_SIZE)
flag_size = nclasses_existence + nfeatures

num_gpus = ngpus_per_node
batch_size = args.bs
scale_batch_size = 1
ubs = scale_batch_size  # unified batch scale
if num_gpus > 1:
    ubs = ubs * num_gpus

from torch.utils.data import DataLoader
if dummyds:
    from v26.avatar_dataset import AvatarDetailsDatasetDummy as dataset, inputs_to_struct_raw as inputs_to_struct

    def flag_to_comp(flag):
        return 1, 1

    train_ds = dataset(inshape, flag_size, nclasses_existence, nsamples_train)
    test_ds = dataset(inshape, flag_size, nclasses_existence, nsamples_test)
    val_ds = dataset(inshape, flag_size, nclasses_existence, nsamples_test)
    nsamples_val = len(
        val_ds
    )  # validation set is only sometimes present so nsamples_val is not always available
else:
    if flag_at is FlagAt.NOFLAG:
        from v26.avatar_dataset import AvatarDetailsDatasetRawNew as dataset, inputs_to_struct_raw_label_all as inputs_to_struct
    else:
        from v26.avatar_dataset import AvatarDetailsDatasetRawNew as dataset, inputs_to_struct_raw as inputs_to_struct
    train_ds = dataset(os.path.join(base_samples_dir, 'train'),
                       nclasses_existence,
                       nfeatures,
                       nsamples_train,
                       split=True)
    normalize_image = True
    if normalize_image:
        # just for getting the mean image
        train_dl = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=args.workers,
                              shuffle=False,
                              pin_memory=True)
        mean_image = get_mean_image(train_dl, inshape, inputs_to_struct)
        train_ds = dataset(os.path.join(base_samples_dir, 'train'),
                           nclasses_existence,
                           nfeatures,
                           nsamples_train,
                           split=True,
                           mean_image=mean_image)
    else:
        mean_image = None
    test_ds = dataset(os.path.join(base_samples_dir, 'test'),
                      nclasses_existence,
                      nfeatures,
                      nsamples_test,
                      split=True,
                      mean_image=mean_image)
    val_ds = dataset(os.path.join(base_samples_dir, 'val'),
                     nclasses_existence,
                     nfeatures,
                     split=True,
                     mean_image=mean_image)
    nsamples_val = len(
        val_ds
    )  # validation set is only sometimes present so nsamples_val is not always available

    def flag_to_comp(flag):
        avatar_id = flag[:nclasses_existence].nonzero()[0][0]
        feature_id = flag[nclasses_existence:].nonzero()[0][0]
        return avatar_id, feature_id


train_sampler = None
batch_size = batch_size * ubs

train_dl = DataLoader(train_ds,
                      batch_size=batch_size,
                      num_workers=args.workers,
                      shuffle=(train_sampler is None),
                      pin_memory=True,
                      sampler=train_sampler)
# train_dl = DataLoader(train_ds, batch_size=batch_size,num_workers=0,shuffle=False,pin_memory=True, sampler=train_sampler)
test_dl = DataLoader(test_ds,
                     batch_size=batch_size,
                     num_workers=args.workers,
                     shuffle=False,
                     pin_memory=True)
val_dl = DataLoader(val_ds,
                    batch_size=batch_size,
                    num_workers=args.workers,
                    shuffle=False,
                    pin_memory=True)

nbatches_train = len(train_dl)
nbatches_val = len(val_dl)
nbatches_test = len(test_dl)

train_dataset = WrappedDataLoader(train_dl, preprocess)
test_dataset = WrappedDataLoader(test_dl, preprocess)
val_dataset = WrappedDataLoader(val_dl, preprocess)

the_train_dataset = DatasetInfo(True, train_dataset, nbatches_train, 'Train',
                                args.checkpoints_per_epoch)
the_test_dataset = DatasetInfo(False, test_dataset, nbatches_test, 'Test')
the_datasets = [the_train_dataset, the_test_dataset]
if nsamples_val > 0:
    the_val_dataset = DatasetInfo(False, val_dataset, nbatches_val,
                                  'Validation')
    the_datasets += [the_val_dataset]
if False:
    inputs = train_ds[0]
    samples = inputs_to_struct(inputs)
# %% model options
model_opts = SimpleNamespace()
model_opts.data_dir = data_dir
model_opts.normalize_image = normalize_image
model_opts.flag_at = flag_at
model_opts.nclasses_existence = nclasses_existence
if model_opts.flag_at is FlagAt.NOFLAG:
    model_opts.nclasses = ntypes
else:
    model_opts.nclasses = [ntypes[0]]

model_opts.flag_size = flag_size
model_opts.norm_fun = BatchNorm
model_opts.activation_fun = nn.ReLU
model_opts.use_td_loss = False
model_opts.use_bu1_loss = True
model_opts.use_bu2_loss = True
model_opts.use_lateral_butd = True
model_opts.use_lateral_tdbu = True
model_opts.use_final_conv = False
model_opts.ntaskhead_fc = 1
setup_flag(model_opts)

#based on ResNet 18
model_opts.nfilters = [64, 64, 128, 256, 512]
model_opts.strides = [2, 2, 2, 2, 2]
# filter sizes
model_opts.ks = [7, 3, 3, 3, 3]
model_opts.ns = [0, 2, 2, 2, 2]
model_opts.inshape = inshape
if model_opts.flag_at is FlagAt.BU1_SIMPLE:
    model_opts.ns = 3 * np.array(model_opts.ns)  # results in 56 layers

flag_str = str(model_opts.flag_at).replace('.', '_').lower()
num_gpus = torch.cuda.device_count()
dummytype = '_dummyds' * dummyds
base_model_dir = 'avatar_details_pyt_v26_%s_sgd%d%s' % (flag_str, args.SGD,
                                                        dummytype)
if not model_opts.use_td_loss and model_opts.use_bu1_loss:
    base_model_dir = base_model_dir + '_two_losses'
model_dir = os.path.join(results_dir,
                         base_model_dir + '_%s' % (base_tf_records_dir))
if args.hyper_search:
    model_dir = os.path.join(model_dir, 'cmdo%d' % args.hyper)
    if args.only_cont and not os.path.exists(model_dir):
        # return
        sys.exit()

model_opts.inputs_to_struct = inputs_to_struct
# just for logging. changes must be made in code in order to take effect
model_opts.lateral_per_neuron = False
model_opts.separable = True

model_opts.model_dir = model_dir
if ENABLE_LOGGING:
    model_opts.logfname = 'log.txt'
else:
    print('Logging disabled')
    model_opts.logfname = None

os.makedirs(model_dir, exist_ok=True)

model_opts.distributed = args.distributed

log_init(model_opts)
print_info(model_opts)
save_script(model_opts)

if args.hyper_search:
    logger.info('sysargv %s' % sys.argv)
    logger.info('using command options lr,bs,wd:%f,%f,%f' %
                (args.lr, args.bs, args.wd))

# %% create model
if model_opts.flag_at is FlagAt.BU1_SIMPLE:
    model = BUModelSimple(model_opts)
else:
    model = BUTDModelShared(model_opts)

if not torch.cuda.is_available():
    logger.info('using CPU, this will be slow')
else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()


# %% loss and metrics
def get_model_outs(model, outs):
    if type(model) is torch.nn.DataParallel or type(
            model) is torch.nn.parallel.DistributedDataParallel:
        return model.module.outs_to_struct(outs)
    else:
        return model.outs_to_struct(outs)


def multi_label_accuracy_base(outs, samples, nclasses):
    cur_batch_size = samples.image.shape[0]
    preds = torch.zeros((cur_batch_size, len(nclasses)),
                        dtype=torch.int).to(dev, non_blocking=True)
    for k in range(len(nclasses)):
        taskk_out = outs.task[:, :, k]
        predsk = torch.argmax(taskk_out, axis=1)
        preds[:, k] = predsk
    label_task = samples.label_task
    task_accuracy = (preds == label_task).float()
    return preds, task_accuracy


def multi_label_accuracy(outs, samples, nclasses):
    preds, task_accuracy = multi_label_accuracy_base(outs, samples, nclasses)
    task_accuracy = task_accuracy.mean(axis=1)  # per single example
    return preds, task_accuracy


def multi_label_accuracy_weighted_loss(outs, samples, nclasses):
    preds, task_accuracy = multi_label_accuracy_base(outs, samples, nclasses)
    loss_weight = samples.loss_weight
    task_accuracy = task_accuracy * loss_weight
    task_accuracy = task_accuracy.sum(axis=1) / loss_weight.sum(
        axis=1)  # per single example
    return preds, task_accuracy


class Measurements(MeasurementsBase):
    def __init__(self, opts, model):
        super(Measurements, self).__init__(opts)
        # self.reset()
        self.model = model
        self.opts = opts
        if self.opts.use_bu1_loss:
            super().add_name('Occurence Acc')

        if self.opts.use_bu2_loss:
            super().add_name('Task Acc')

        self.init_results()

    def update(self, inputs, outs, loss):
        super().update(inputs, outs, loss)
        outs = get_model_outs(model, outs)
        samples = inputs_to_struct(inputs)
        if self.opts.use_bu1_loss:
            occurence_pred = outs.occurence > 0
            occurence_accuracy = (
                occurence_pred == samples.label_existence).type(
                    torch.float).mean(axis=1)
            super().update_metric(self.occurence_accuracy,
                                  occurence_accuracy.sum().cpu().numpy())

        if self.opts.use_bu2_loss:
            preds, task_accuracy = self.opts.task_accuracy(
                outs, samples, self.opts.nclasses)
            super().update_metric(self.task_accuracy,
                                  task_accuracy.sum().cpu().numpy())

    def reset(self):
        super().reset()
        if self.opts.use_bu1_loss:
            self.occurence_accuracy = np.array(0.0)
            self.metrics += [self.occurence_accuracy]

        if self.opts.use_bu2_loss:
            self.task_accuracy = np.array(0.0)
            self.metrics += [self.task_accuracy]


def multi_label_loss_base(outs, samples, nclasses):
    losses_task = torch.zeros((samples.label_task.shape)).to(dev,
                                                             non_blocking=True)
    for k in range(len(nclasses)):
        taskk_out = outs.task[:, :, k]
        label_taskk = samples.label_task[:, k]
        loss_taskk = loss_task_multi_label(taskk_out, label_taskk)
        losses_task[:, k] = loss_taskk
    return losses_task


def multi_label_loss(outs, samples, nclasses):
    losses_task = multi_label_loss_base(outs, samples, nclasses)
    loss_task = losses_task.mean(
    )  # a single valued result for the whole batch
    return loss_task


def multi_label_loss_weighted_loss(outs, samples):
    losses_task = multi_label_loss_base(outs, samples)
    loss_weight = samples.loss_weight
    losses_task = losses_task * loss_weight
    loss_task = losses_task.sum() / loss_weight.sum(
    )  # a single valued result for the whole batch
    return loss_task


def loss_fun(inputs, outs):
    # nn.CrossEntropyLoss on GPU is not deterministic. However using CPU doesn't seem to help either...
    outs = get_model_outs(model, outs)
    samples = inputs_to_struct(inputs)
    losses = []
    if model_opts.use_bu1_loss:
        loss_occ = loss_occurence(outs.occurence, samples.label_existence)
        losses.append(loss_occ)

    if model_opts.use_td_loss:
        loss_seg_td = loss_seg(outs.td_head, samples.seg)
        loss_bu1_after_convergence = 1
        loss_td_after_convergence = 100
        ratio = loss_bu1_after_convergence / loss_td_after_convergence
        losses.append(ratio * loss_seg_td)

    if model_opts.use_bu2_loss:
        loss_task = model_opts.bu2_loss(outs, samples, model_opts.nclasses)
        losses.append(loss_task)

    loss = torch.sum(torch.stack(losses))
    return loss


loss_occurence = torch.nn.BCEWithLogitsLoss(reduction='mean').to(dev)
loss_seg = torch.nn.MSELoss(reduction='mean').to(dev)
loss_task_op = nn.CrossEntropyLoss(reduction='mean').to(dev)
loss_task_multi_label = nn.CrossEntropyLoss(reduction='none').to(dev)

set_datasets_measurements(the_datasets, Measurements, model_opts, model)

if model_opts.flag_at is FlagAt.NOFLAG:
    model_opts.bu2_loss = multi_label_loss_weighted_loss
    model_opts.task_accuracy = multi_label_accuracy_weighted_loss
else:
    model_opts.bu2_loss = multi_label_loss
    model_opts.task_accuracy = multi_label_accuracy

# %% fit
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

train_opts = SimpleNamespace()
train_opts.model = model
train_opts.weight_decay = args.wd
train_opts.initial_lr = args.lr
learning_rates_mult = np.ones(300)
learning_rates_mult = get_multi_gpu_learning_rate(learning_rates_mult,
                                                  num_gpus, scale_batch_size,
                                                  ubs)
if args.checkpoints_per_epoch > 1:
    learning_rates_mult = np.repeat(learning_rates_mult,
                                    args.checkpoints_per_epoch)
train_opts.batch_size = batch_size
train_opts.nbatches_train = nbatches_train  # just for logging
train_opts.nbatches_val = nbatches_val  # just for logging
train_opts.nbatches_test = nbatches_test  # just for logging

train_opts.num_gpus = num_gpus
train_opts.EPOCHS = len(learning_rates_mult)
train_opts.learning_rates_mult = learning_rates_mult
train_opts.load_model_if_exists = True
train_opts.model_dir = model_opts.model_dir
train_opts.save_model = True
train_opts.abort_after_epochs = 0
if args.SGD:
    optimizer = optim.SGD(model.parameters(),
                          lr=train_opts.initial_lr,
                          momentum=0.9,
                          weight_decay=train_opts.weight_decay)
else:
    optimizer = optim.Adam(model.parameters(),
                           lr=train_opts.initial_lr,
                           weight_decay=train_opts.weight_decay)
train_opts.optimizer = optimizer
train_opts.loss_fun = loss_fun
lmbda = lambda epoch: train_opts.learning_rates_mult[epoch]
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
train_opts.scheduler = scheduler
train_opts.checkpoints_per_epoch = args.checkpoints_per_epoch
train_opts.train_sampler = train_sampler
train_opts.distributed = args.distributed
train_opts.first_node = True
train_opts.gpu = None

fit(train_opts, the_datasets)
if not interactive_session:
    sys.exit()
    # return


#load_model(train_opts,os.path.join(model_dir,'model_latest.pt'));
# %% visualize predictions
def from_network_transpose(samples, outs):
    if normalize_image:
        samples.image += mean_image
    samples.image = samples.image.transpose(0, 2, 3, 1)
    samples.seg = samples.seg.transpose(0, 2, 3, 1)
    if model_opts.use_td_loss:
        outs.td_head = outs.td_head.transpose(0, 2, 3, 1)
    return samples, outs


import matplotlib.patches as patches


def get_bounding_box(mask):
    if len(mask.shape) > 2:
        bool_mask = mask[:, :, 1]
    else:
        bool_mask = mask
    h, w = bool_mask.shape
    rows, cols = bool_mask.nonzero()
    margin = 5
    stx = min(cols)
    sty = min(rows)
    endx = max(cols)
    endy = max(rows)
    stx -= margin
    sty -= margin
    endx += margin
    endy += margin
    stx = max(0, stx)
    sty = max(0, sty)
    endx = min(endx, w)
    endy = min(endy, h)
    return [stx, sty, endx, endy]


features_strings = [
    'Avatar', 'Tilt', 'Background type', 'Clothes type', 'Glasses type',
    'Hair type', 'Mustache type'
]
ds_iter = iter(train_dataset)
inputs = next(ds_iter)
loss, outs = test_step(inputs, train_opts)
samples, outs = from_network(inputs, outs, model.module, inputs_to_struct)
samples, outs = from_network_transpose(samples, outs)
preds = np.array(outs.occurence > 0, dtype=np.float)
fig = plt.figure(figsize=(15, 4))
if model_opts.use_td_loss:
    n = 3
else:
    n = 2
for k in range(len(samples.image)):
    fig.clf()
    fig.tight_layout()
    ax = plt.subplot(1, n, 1)
    ax.axis('off')
    # we only have existence information about all the avatars, without order
    present = str(samples.label_existence[k].nonzero()[0].tolist())
    tit = 'Present: %s\n' % present
    fl = samples.flag[k]
    avatar_id, feature_id = flag_to_comp(fl)

    if model_opts.flag_at is FlagAt.NOFLAG:
        pred = outs.task[k].argmax(axis=0)
        pred = np.array(pred).reshape((-1, nfeatures))
        predicted_feature_value = pred[avatar_id][feature_id]
        gt = samples.label_task[k]
        gt = np.array(gt).reshape((-1, nfeatures))
        feature_value = gt[avatar_id][feature_id]
    else:
        feature_value = samples.label_task[k][0]
        predicted_feature_value = outs.task[k].argmax()

    ins = 'Instruction: Avatar %d, %s' % (avatar_id,
                                          features_strings[feature_id])
    tit = tit + ins
    plt.imshow(samples.image[k].astype(np.uint8))
    plt.title(tit)

    ax = plt.subplot(1, n, 2)
    ax.axis('off')
    plt.imshow(samples.image[k].astype(np.uint8))
    curseg = samples.seg[k].squeeze()
    if False:
        curseg_min = curseg.min()
    else:
        # somewhat hacky way to find the background: assuming it is the most dominant feature
        curseg_ch0 = curseg[:, :, 0]
        vals, counts = np.unique(curseg_ch0, return_counts=True)
        curseg_min = vals[counts.argmax()]
    mask = curseg > curseg_min
    if len(np.unique(mask)) > 1:
        [stx, sty, endx, endy] = get_bounding_box(mask)
        ax.add_patch(
            patches.Rectangle((stx, sty),
                              endx - stx,
                              endy - sty,
                              linewidth=2,
                              edgecolor='g',
                              facecolor='none'))

    gt_str = 'Ground Truth: %s = %d' % (features_strings[feature_id],
                                        feature_value)

    pred_str = 'Prediction: %s = %d' % (features_strings[feature_id],
                                        predicted_feature_value)
    predicted_existing_avatar_ids = preds[k]
    if feature_value == predicted_feature_value:
        font = {'color': 'blue'}
    else:
        font = {'color': 'red'}
    if model_opts.use_td_loss:
        tit_str = gt_str
        plt.title(tit_str)
    else:
        tit_str = gt_str + '\n' + pred_str
        plt.title(tit_str, fontdict=font)
    if model_opts.use_td_loss:
        ax = plt.subplot(1, n, 3)
        ax.axis('off')
        image_tdk = np.array(outs.td_head[k])
        image_tdk = image_tdk - np.min(image_tdk)
        image_tdk = image_tdk / np.max(image_tdk)
        plt.imshow(image_tdk)
        plt.title(pred_str, fontdict=font)
    print(k)
    print(predicted_existing_avatar_ids)
    #    savefig(os.path.join(avatars_dir, 'examples%d.png'% k), dpi=90, bbox_inches='tight' )
    pause_image()
# %% percent correct
accs_id = [[] for i in range(nfeatures)]
npersons = 6
perc = np.zeros((npersons, nfeatures))
lens = np.zeros((npersons, nfeatures))
for inputs in val_dataset:
    loss, outs = test_step(inputs, train_opts)
    samples, outs = from_network(inputs, outs, model.module, inputs_to_struct)
    for k in range(len(samples.image)):
        fl = samples.flag[k]
        avatar_id, feature_id = flag_to_comp(fl)

        if model_opts.flag_at is FlagAt.NOFLAG:
            pred = outs.task[k].argmax(axis=0)
            pred = np.array(pred).reshape((-1, nfeatures))
            predicted_feature_value = pred[avatar_id][feature_id]
            gt = samples.label_task[k]
            gt = np.array(gt).reshape((-1, nfeatures))
            feature_value = gt[avatar_id][feature_id]
        else:
            feature_value = samples.label_task[k][0]
            predicted_feature_value = outs.task[k].argmax()
        acc_id = int(predicted_feature_value == feature_value)
        accs_id[feature_id].append(acc_id)
        perc[avatar_id][feature_id] += acc_id
        lens[avatar_id][feature_id] += 1

#len_features=[len(lst) for lst in accs_id]
#PC_features=[np.mean(lst) for lst in accs_id]
PC2 = perc / lens
print(PC2)
print(np.nanmean(PC2))
b = PC2.mean(axis=0)
print(b)
print(b[[1, 3, 4, 5, 6]].mean())

# for just the accuracy:
# the_val_dataset.do_epoch(0,train_opts)
# print(the_val_dataset.measurements.print_epoch())
