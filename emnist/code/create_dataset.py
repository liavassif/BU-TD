from types import SimpleNamespace
import os
import sys
import skimage.transform
from skimage import color
import numpy as np
import pickle
from PIL import Image
import datetime
import argparse
# for copying the generating script
import __main__ as mainmod
import shutil

#%% command line options
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-dir',
    default='../data',
    type=str,
    help=
    'root dir for dataset generation and location of the input Avatars Raw file'
)
parser.add_argument('-t',
                    '--threads',
                    default=10,
                    type=int,
                    help='the number of threads used (default: 10)')
parser.add_argument(
    '-e',
    '--extended',
    action='store_true',
    help='Create the extended set instead of the sufficient set')
parser.add_argument('-n',
                    '--nchars',
                    default=6,
                    type=int,
                    help='the number of chars in each image (default: 6)')


# %% augmentation
def get_batch_base(imdb, batch_range, opts):
    batch_images = imdb.images[batch_range]
    batch_labels = imdb.labels[batch_range]
    batch_segs = imdb.segs[batch_range]

    aug_data = opts.dataset_info.aug_data
    augment_type = 'aug_package'
    if aug_data.augment:
        aug_seed = aug_data.aug_seed
        if augment_type == 'aug_package':
            aug = aug_data.aug
            aug.seed_(aug_seed)

            batch_images = aug.augment_images(batch_images)
            if aug_data.aug_seg:
                aug_nn_interpolation = aug_data.aug_nn_interpolation
                aug_nn_interpolation.seed_(aug_seed)
                batch_segs_without_add_color = aug_nn_interpolation.augment_images(
                    batch_segs)
                aug_with_add_color = aug_data.aug
                aug_with_add_color.seed_(aug_seed)

                batch_segs_with_add_color = aug_with_add_color.augment_images(
                    batch_segs)
                mask = batch_segs_without_add_color > 0
                batch_masks = mask

                # only add random color for the avatar segmentation, not the background
                # segmentation
                batch_segs_with_add_color[
                    ~mask] = batch_segs_without_add_color[~mask]
                batch_segs = batch_segs_with_add_color

            aug_data.aug_seed += 1

    result = SimpleNamespace()
    result.images = batch_images
    result.labels = batch_labels
    result.segs = batch_segs
    result.size = len(batch_images)
    return result


def get_aug_data(IMAGE_SIZE):
    aug_data = SimpleNamespace()
    aug_data.aug_seed = 0
    # augmentation init
    from imgaug import augmenters as iaa
    from imgaug import parameters as iap
    import imgaug as ia
    color_add_range = int(0.2 * 255)
    rotate_deg = 2
    aug = iaa.Sequential(
        [
            iaa.Affine(
                translate_percent={
                    "x": (-0.1, 0.1),
                    "y": (-0.05, 0.05)
                },
                rotate=(-rotate_deg, rotate_deg),
                mode='edge',
                name='affine'),  # translate by -20 to +20 percent (per axis))
        ],
        random_state=0)

    aug_data.aug_seg = True
    if aug_data.aug_seg:
        aug_nn_interpolation = aug.deepcopy()
        aff_aug = aug_nn_interpolation.find_augmenters_by_name('affine')[0]
        aff_aug.order = iap.Deterministic(0)
        aug_data.aug_nn_interpolation = aug_nn_interpolation

    # this will be used to add random color to the segmented avatar but not the segmented background
    aug.append(iaa.Add(
        (-color_add_range,
         color_add_range)))  # only for the image not the segmentation

    aug_data.aug = aug
    aug_data.image_size = IMAGE_SIZE
    return aug_data


# %% storage
def store_sample_memory(sample, samples):
    samples.append(sample)


def store_sample_disk_pytorch(sample, cur_samples_dir, folder_split,
                              folder_size):
    samples_dir = cur_samples_dir
    i = sample.id
    if folder_split:
        samples_dir = os.path.join(cur_samples_dir, '%d' % (i // folder_size))
        if not os.path.exists(samples_dir):
            os.makedirs(samples_dir, exist_ok=True)
    img = sample.image
    seg = sample.seg
    img_fname = os.path.join(samples_dir, '%d_img.jpg' % i)
    c = Image.fromarray(img)
    c.save(img_fname)
    seg_fname = os.path.join(samples_dir, '%d_seg.jpg' % i)
    c = Image.fromarray(seg)
    c.save(seg_fname)
    del sample.image, sample.seg, sample.infos
    data_fname = os.path.join(samples_dir, '%d_raw.pkl' % i)
    with open(data_fname, "wb") as new_data_file:
        pickle.dump(sample, new_data_file)


# %% EMNIST dataset
# This is the dataset used in the paper. It is most likely similar although not identical to the PyTorch version
def convert_raw_data_tensorflow(emnist_preprocess):
    emnist_download_dir = emnist_preprocess.download_dir
    if not os.path.exists(emnist_download_dir):
        print(
            'Note that this worked using an older version of Tensorflow (1.x). Most likely it doesn\'t work now. Use the PyTorch function instead'
        )
        print(
            'For Tensorflow you should manually download and extract EMNIST from https://www.nist.gov/itl/products-and-services/emnist-dataset'
        )
        print('Extract it into %s' % emnist_download_dir)
        sys.exit(-1)
    else:
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets(emnist_download_dir, one_hot=False)
        images_train = mnist.train.images
        images_test = mnist.test.images
        labels_train = mnist.train.labels
        labels_test = mnist.test.labels

        with open(emnist_preprocess.data_fname, "wb") as new_data_file:
            pickle.dump((images_train, images_test, labels_train, labels_test),
                        new_data_file)


def convert_raw_data_pytorch(emnist_preprocess):
    emnist_download_dir = emnist_preprocess.download_dir
    emnist_split_type = 'balanced'
    import torchvision
    labels = [[] for _ in range(2)]
    imgs = [[] for _ in range(2)]
    for k, train in enumerate((True, False)):
        emnist_data = torchvision.datasets.EMNIST(emnist_download_dir,
                                                  emnist_split_type,
                                                  train=train,
                                                  download=True)
        for img, lbl in emnist_data:
            labels[k].append(lbl)
            img = np.array(img)
            img = img.reshape(-1)
            imgs[k].append(img)
    images_train, images_test = map(np.array, imgs)
    labels_train, labels_test = map(np.array, labels)
    with open(emnist_preprocess.data_fname, "wb") as new_data_file:
        pickle.dump((images_train, images_test, labels_train, labels_test),
                    new_data_file)


# load the raw EMNIST dataset. If it doesn't exist download and process it
def load_raw_data(emnist_preprocess):
    data_fname = emnist_preprocess.data_fname
    if not os.path.exists(data_fname):
        print('Converting EMNIST raw data using %s' %
              emnist_preprocess.convertor)
        if emnist_preprocess.convertor == 'pytorch':
            convert_raw_data_pytorch(emnist_preprocess)
        else:
            convert_raw_data_tensorflow(emnist_preprocess)
        print('Done converting EMNIST raw data')

    with open(data_fname, "rb") as new_data_file:
        images_train, images_test, labels_train, labels_test = pickle.load(
            new_data_file)
    xs = np.concatenate((images_train, images_test))
    ys = np.concatenate((labels_train, labels_test))

    LETTER_SIZE = 28
    images = xs.reshape(len(xs), LETTER_SIZE, LETTER_SIZE)
    images = images.transpose((0, 2, 1))
    labels = ys
    total_bins = 8
    IMAGE_SIZE = [LETTER_SIZE * 4, LETTER_SIZE * total_bins]

    return images, labels, total_bins, LETTER_SIZE, IMAGE_SIZE


# class to letter dictionary
def get_cl2let(emnist_preprocess):
    # the mapping file is provided with the EMNIST dataset
    with open(emnist_preprocess.mapping_fname, "r") as text_file:
        lines = text_file.read().split('\n')
        cl2letmap = [line.split() for line in lines]
        cl2letmap = cl2letmap[:-1]
        cl2let = {int(mapi[0]): chr(int(mapi[1])) for mapi in cl2letmap}
        cl2let[47] = 'Border'
        cl2let[48] = 'NA'
        return cl2let


# %% create samples
# create a sample image and label from the lightweight example
def gen_sample(sample_id, is_train, aug_data, grayscale_as_rgb, images_raw,
               labels_raw, nclasses, PERSON_SIZE, IMAGE_SIZE, edge_class,
               example, augment_sample, not_available_class):
    # start by creating the image background
    image = 0 * np.ones(IMAGE_SIZE, dtype=np.float32)

    sample_nchars = len(example.chars)
    sample = SimpleNamespace()
    # collect the necessary chars in the infos Namespace and create the sample image
    infos = []
    for char in example.chars:
        char_id = char.id
        s_id = char.s_id
        ims = images_raw
        lbs = labels_raw
        im = ims[s_id]
        label = lbs[s_id]
        scale = char.scale
        sz = scale * np.array([PERSON_SIZE, PERSON_SIZE])
        sz = sz.astype(np.int)
        h, w = sz
        im = skimage.transform.resize(im, sz, mode='constant')
        stx = char.location_x
        endx = stx + w
        sty = char.location_y
        endy = sty + h
        # this is a view into the image and when it changes the image also changes
        part = image[sty:endy, stx:endx]
        mask = im.copy()
        #        mask[mask > 0.1] = 1
        mask[mask > 0] = 1
        mask[mask < 1] = 0
        rng = mask > 0
        part[rng] = mask[rng] * im[rng] + (1 - mask[rng]) * part[rng]
        info = SimpleNamespace()
        info.char_id = char_id
        info.s_id = s_id
        info.label = label
        info.im = im
        info.mask = mask
        info.stx = stx
        info.endx = endx
        info.sty = sty
        info.endy = endy
        info.edge_to_the_right = char.edge_to_the_right
        infos.append(info)

    # create two of the labels: existence and ordered
    label = np.zeros(nclasses)
    for info in infos:
        label[info.char_id] = 1
    label_existence = np.array(label)

    # the characters in order as seen in the image
    row = list()
    rows = []
    for k in range(len(infos)):
        info = infos[k]
        char = info.char_id
        row.append(char)
        if info.edge_to_the_right:
            rows.append(row)
            row = list()
    label_ordered = np.array(rows)

    # create segmentation of the char we are interested in (object-based attention)
    query_part_id = example.query_part_id
    info = infos[query_part_id]
    char = info.char_id
    seg = np.zeros_like(image)
    # this is a view into the image and when it changes the image also changes
    part = seg[info.sty:info.endy, info.stx:info.endx]
    mask = info.mask
    im = info.im
    rng = mask > 0
    part[rng] = mask[rng] * im[rng] + (1 - mask[rng]) * part[rng]

    # instruction and task label
    query_part_id = example.query_part_id
    info = infos[query_part_id]
    char = info.char_id
    adj_type = example.adj_type
    rows, obj_per_row = label_ordered.shape
    r, c = (label_ordered == char).nonzero()
    # there should only be a single match. use it
    r = r[0]
    c = c[0]
    # find the adjacent char
    if adj_type == 0:
        # right
        if c == (obj_per_row - 1):
            label_task = edge_class
        else:
            label_task = label_ordered[r, c + 1]
    else:
        # left
        if c == 0:
            label_task = edge_class
        else:
            label_task = label_ordered[r, c - 1]
    flag = np.array([adj_type, char])

    # even for grayscale images, store them as 3 channels RGB like
    if grayscale_as_rgb:
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            seg = seg[:, :, np.newaxis]
        image = 255 * np.concatenate((image, image, image), axis=2)
        seg = 255 * np.concatenate((seg, seg, seg), axis=2)

    image = image.astype(np.uint8)
    seg = seg.astype(np.uint8)

    if is_train and augment_sample:
        # augment
        imdb = SimpleNamespace()
        imdb.images = image[np.newaxis]
        imdb.segs = seg[np.newaxis]
        imdb.labels = label_existence[np.newaxis]
        batch_range = range(1)
        opts = SimpleNamespace()
        opts.dataset_info = SimpleNamespace()
        opts.dataset_info.aug_data = aug_data
        batch_data = get_batch_base(imdb, batch_range, opts)

        image = batch_data.images[0]
        seg = batch_data.segs[0]

    sample.infos = infos
    sample.image = image
    sample.id = sample_id
    sample.label_existence = label_existence.astype(np.int)
    sample.label_ordered = label_ordered
    sample.query_part_id = query_part_id
    sample.seg = seg
    sample.label_task = label_task
    sample.flag = flag
    sample.is_train = is_train
    return sample


# Create several samples from lightweight examples. This function can be considered as a 'job' which can run in parallel
def gen_samples(job_id, range_start, range_stop, examples, storage_dir,
                ds_type, nclasses, storage_type, job_chunk_size, edge_class,
                img_channels, grayscale_as_rgb, augment_sample,
                not_available_class, sanity, folder_split, folder_size,
                emnist_preprocess):
    images, labels, total_bins, LETTER_SIZE, IMAGE_SIZE = load_raw_data(
        emnist_preprocess)

    storage_disk = not storage_type == 'mem'
    if not storage_disk:
        memsamples = []

    aug_data = None
    is_train = ds_type == 'train'
    if is_train:
        rnd_shift = 0
        if augment_sample:
            # create a separate augmentation per job since we always update aug_data.aug_seed
            aug_data = get_aug_data(IMAGE_SIZE)
            aug_data.aug_seed = range_start
            aug_data.augment = True
    else:
        # both validation and training use the same job id. make the random generator different
        rnd_shift = 100000

    # divide the job into several smaller parts and run them sequentially
    nbatches = np.ceil((range_stop - range_start) / job_chunk_size)
    ranges = np.arange(range_start, range_stop, job_chunk_size)
    if ranges[-1] != range_stop:
        ranges = ranges.tolist()
        ranges.append(range_stop)
    rel_id = 0
    for k in range(len(ranges) - 1):
        range_start = ranges[k]
        range_stop = ranges[k + 1]
        print('%s: job %d. processing: %s-%d-%d' %
              (datetime.datetime.now(), job_id, ds_type, range_start,
               range_stop - 1))
        if storage_type == 'pytorch':
            cur_samples_dir = os.path.join(storage_dir, ds_type)
            if not os.path.exists(cur_samples_dir):
                os.makedirs(cur_samples_dir, exist_ok=True)
            print('%s: storing in: %s' %
                  (datetime.datetime.now(), cur_samples_dir))

        sys.stdout.flush()

        for samid in range(range_start, range_stop):
            sample = gen_sample(samid, is_train, aug_data, grayscale_as_rgb,
                                images, labels, nclasses, LETTER_SIZE,
                                IMAGE_SIZE, edge_class, examples[rel_id],
                                augment_sample, not_available_class)
            if sample is None:
                continue

            if storage_type == 'pytorch':
                store_sample_disk_pytorch(sample, cur_samples_dir,
                                          folder_split, folder_size)
            elif storage_type == 'mem':
                store_sample_memory(sample, memsamples)

            rel_id += 1

    print('%s: Done' % (datetime.datetime.now()))
    if not storage_disk:
        return memsamples
    else:
        return


# %% main
def main():
    cmd_args = parser.parse_args()
    njobs = cmd_args.threads
    storage_type = 'pytorch'
    # Use multiprocessing on this machine
    local_multiprocess = njobs > 1
    # each 'job' processes several chunks. Each chunk is of 'storage_batch_size' samples
    job_chunk_size = 1000

    sanity = 0  # for debugging
    if sanity:
        storage_type = 'mem'
        local_multiprocess = False

    storage_disk = not storage_type == 'mem'

    data_dir = cmd_args.data_dir
    emnist_dir = os.path.join(data_dir, 'emnist')
    # split the generated data set to 1000 files per folder
    folder_split = True
    folder_size = 1000
    if storage_disk:
        # the name of the dataset to create
        base_storage_dir = '%d_' % cmd_args.nchars
        if cmd_args.extended:
            base_storage_dir += 'extended'
        else:
            base_storage_dir += 'sufficient'
        if storage_type == 'pytorch':
            new_emnist_dir = os.path.join(emnist_dir, 'samples')
            base_samples_dir = os.path.join(new_emnist_dir, base_storage_dir)
            if not os.path.exists(base_samples_dir):
                os.makedirs(base_samples_dir, exist_ok=True)
            storage_dir = base_samples_dir

        data_fname = os.path.join(storage_dir, 'conf')
    else:
        storage_dir = None

    augment_sample = True

    # obtain the EMNIST dataset
    emnist_preprocess = SimpleNamespace()
    emnist_preprocess.convertor = 'pytorch'
    if emnist_preprocess.convertor == 'pytorch':
        download_dir = os.path.join(emnist_dir, 'emnist_raw')
        raw_data_fname = os.path.join(emnist_dir, 'emnist-pyt.pkl')
    else:
        download_dir = '../data/emnist/gzip'
        raw_data_fname = os.path.join(emnist_dir, 'emnist-tf.pkl')
    emnist_preprocess.data_fname = raw_data_fname
    emnist_preprocess.download_dir = download_dir
    emnist_preprocess.mapping_fname = os.path.join(
        emnist_dir, "emnist-balanced-mapping.txt")
    _, labels, total_bins, LETTER_SIZE, IMAGE_SIZE = load_raw_data(
        emnist_preprocess)

    grayscale_as_rgb = True
    img_channels = 3

    nclasses_existence = len(set(labels))
    edge_class = nclasses_existence
    not_available_class = edge_class + 1
    max_class = not_available_class + 1
    ntypes = (max_class) * np.ones(nclasses_existence, dtype=np.int32)

    # if True, generate multiple examples (image,label) pairs from each image, else generate a single example
    single_feat_to_generate = False
    # generate multiple examples (image,label) pairs from each image
    ngenerate = None
    total_rows = 4
    obj_per_row = 6
    sample_nchars = cmd_args.nchars
    nsamples_test = 2000

    if sample_nchars == 24:
        # Generate 'ngenerate' examples from each image for a left-of query and the same for a right-of query
        # The total number of training examples would be 10,000*20*2 for the extended set and 10,000*5*2 for the sufficient
        train_iterate_all_directions = True
        nsamples_train = 10000
        if cmd_args.extended:
            # extended
            ngenerate = 20
        else:
            # sufficient
            ngenerate = 5
    elif sample_nchars == 6:
        # Generate 5 examples from each image, randomly selecting each time either a left-of query or a right-of query
        # The total number of training examples would be 10,000*5 for the extended set and 2,000*5 for the sufficient
        ngenerate = 5
        train_iterate_all_directions = False
        if cmd_args.extended:
            # extended
            nsamples_train = 10000
        else:
            # sufficient
            nsamples_train = 2000

    if sanity:
        train_val_ratio = .95
        nsamples = 100
        nsamples_train = int(train_val_ratio * nsamples)
        nsamples_test = nsamples - nsamples_train
        ngenerate = 2

    generalize = True
    add_non_gen = True
    if not generalize:
        add_non_gen = False

    use_only_valid_classes = True
    if use_only_valid_classes:
        # remove some characters which are very similar to other characters
        # ['O', 'F', 'q', 'L', 't', 'g', 'C', 'S', 'I', 'B', 'Y', 'n', 'b', 'X', 'r', 'H', 'P', 'G']
        invalid = [
            24, 15, 44, 21, 46, 41, 12, 28, 18, 11, 34, 43, 37, 33, 45, 17, 25,
            16
        ]
        all_classes = np.arange(0, nclasses_existence)
        valid_classes = np.setdiff1d(all_classes, invalid)
    else:
        valid_classes = all_classes

    # We don't exclude (char,border) or (border,char) pairs as this drastically limits the available valid training examples
    # Therefore, do not query near borders
    valid_queries = np.arange(sample_nchars)
    near_border = np.arange(obj_per_row - 1, sample_nchars, obj_per_row)
    valid_queries_right = np.setdiff1d(valid_queries, near_border)
    valid_queries = np.arange(sample_nchars)
    near_border = np.arange(0, sample_nchars, obj_per_row)
    valid_queries_left = np.setdiff1d(valid_queries, near_border)

    ndirections = 2
    avail_adj_types = range(ndirections)  # 0: right, 1: left
    minscale = .5
    maxscale = 1
    minshift = 0
    maxshift = .2 * LETTER_SIZE

    image_ids = set()

    ds_types = ['test', 'train']
    nexamples_vec = [nsamples_test, nsamples_train]

    if add_non_gen:
        nsamples_val = nsamples_test
        nexamples_vec.append(nsamples_val)
        ds_types.append('val')
    else:
        nsamples_val = 0
    if not storage_disk:
        memsamples = [[] for _ in range(len(ds_types))]

    if generalize:
        # Exclude part of the training data. Validation set is from the train ditribution. Test is only the excluded data (combinatorial generalization)

        # How many strings (of nsample_chars) to exclude from training
        # For 24 characters, 1 string excludes about 2.4% pairs of consequtive characters, 4 strings: 9.3%, 23 strings: 42%, 37: 52%
        # For 6 characters, 1 string excludes about 0.6% pairs of consequtive characters, 77 strings: 48%, 120: 63%, 379: 90%
        # (these numbers were simply selected in order to achieve a certain percentage)
        ntest_strings = 1

        nsamples_test = 500

        # generate once the same test strings
        np.random.seed(777)
        valid_pairs = np.zeros((nclasses_existence, nclasses_existence))
        for i in valid_classes:
            for j in valid_classes:
                if j != i:
                    valid_pairs[i, j] = 1
        test_chars_list = []
        # it is enough to validate only right-of pairs even when we query for left of as it is symmetric
        for i in range(ntest_strings):
            test_chars = np.random.choice(valid_classes,
                                          sample_nchars,
                                          replace=False)
            print('test chars:', test_chars)
            test_chars_list.append(test_chars)
            test_chars = test_chars.reshape((-1, obj_per_row))
            for row in test_chars:
                for pi in range(obj_per_row - 1):
                    cur_char = row[pi]
                    adj_char = row[pi + 1]
                    valid_pairs[cur_char, adj_char] = 0
        avail_ratio = valid_pairs.sum() / (len(valid_classes) *
                                           (len(valid_classes) - 1))
        exclude_percentage = (100 * (1 - avail_ratio))
        print('Excluding %d strings, %f percentage of pairs' %
              (ntest_strings, exclude_percentage))

    # %% create train/test/val sets
    # first create all the examples, which are lightweight (without the actual images), then send them to parallel jobs
    # in order to create samples from them
    for k, (ds_type, cur_nexamples) in enumerate(zip(ds_types, nexamples_vec)):
        prng = np.random.RandomState(k)
        is_train = ds_type == 'train'
        is_val = ds_type == 'val'
        is_test = ds_type == 'test'
        not_test = not is_test

        # create lightweight examples information
        examples = []
        for i in range(cur_nexamples):
            if generalize:
                if not_test:
                    found = False
                    while not found:
                        # faster generation of an example than a random sample
                        found_hide_many = False
                        while not found_hide_many:
                            sample_chars = []
                            cur_char = prng.choice(valid_classes, 1)[0]
                            sample_chars.append(cur_char)
                            for pi in range(sample_nchars - 1):
                                cur_char_adjs = valid_pairs[cur_char].nonzero(
                                )[0]
                                cur_char_adjs = np.setdiff1d(
                                    cur_char_adjs, sample_chars
                                )  # create a permutation: choose each character at most once
                                if len(cur_char_adjs) > 0:
                                    cur_adj = prng.choice(cur_char_adjs, 1)[0]
                                    cur_char = cur_adj
                                    sample_chars.append(cur_char)
                                    if len(sample_chars) == sample_nchars:
                                        found_hide_many = True
                                else:
                                    break

                        found = True
                else:
                    test_chars_idx = prng.randint(ntest_strings)
                    sample_chars = test_chars_list[test_chars_idx]

            else:
                sample_chars = prng.choice(valid_classes,
                                           sample_nchars,
                                           replace=False)
            image_id = []
            s_ids = []
            for char in sample_chars:
                valid = labels == char
                lbs = labels[valid]
                s_id = prng.randint(0, len(lbs))
                s_id = valid.nonzero()[0][s_id]
                image_id.append(s_id)
                s_ids.append(s_id)
            image_id_hash = str(image_id)
            if image_id_hash in image_ids:
                continue

            image_ids.add(image_id_hash)

            # place the chars on the image
            chars = []
            for samplei in range(sample_nchars):
                char = SimpleNamespace()
                char.s_id = s_ids[samplei]
                char.id = sample_chars[samplei]
                scale = prng.rand() * (maxscale - minscale) + minscale
                new_size = int(scale * LETTER_SIZE)
                char.s_id = s_ids[samplei]
                char.id = sample_chars[samplei]
                char.scale = scale
                shift = prng.rand(2) * (maxshift - minshift) + minshift
                y, x = shift.astype(np.int)
                origr, origc = np.unravel_index(samplei,
                                                [total_rows, obj_per_row])
                c = origc + 1  # start from column 1 instead of 0
                r = origr
                imageh, imagew = IMAGE_SIZE
                stx = c * LETTER_SIZE + x
                stx = max(0, stx)
                stx = min(stx, imagew - new_size)
                sty = r * LETTER_SIZE + y
                sty = max(0, sty)
                sty = min(sty, imageh - new_size)
                char.location_x = stx
                char.location_y = sty
                char.edge_to_the_right = origc == obj_per_row - 1 or samplei == sample_nchars - 1
                chars.append(char)

            print(i)

            if train_iterate_all_directions and is_train:
                adj_types = avail_adj_types
            else:
                adj_types = [prng.choice(avail_adj_types)]

            # generate a single or multiple examples for each generated configuration
            for adj_type in adj_types:
                if adj_type == 0:
                    valid_queries = valid_queries_right
                else:
                    valid_queries = valid_queries_left

                if single_feat_to_generate or is_test or is_val:
                    query_part_ids = [prng.choice(valid_queries)]
                else:
                    cur_ngenerate = ngenerate

                    if cur_ngenerate == -1:
                        query_part_ids = valid_queries
                    else:
                        query_part_ids = prng.choice(valid_queries,
                                                     cur_ngenerate,
                                                     replace=False)
                for query_part_id in query_part_ids:
                    example = SimpleNamespace()
                    example.sample = sample_chars
                    example.query_part_id = query_part_id
                    example.adj_type = adj_type
                    example.chars = chars
                    examples.append(example)

        cur_nexamples = len(examples)
        if is_train:
            nsamples_train = cur_nexamples
        elif is_test:
            nsamples_test = cur_nexamples
        else:
            nsamples_val = cur_nexamples
        print('total of %d examples' % cur_nexamples)

        # divide all the examples across several jobs. Each job generates samples from examples
        cur_njobs = min(njobs,
                        np.ceil((cur_nexamples) / job_chunk_size).astype(int))
        ranges = np.linspace(0, cur_nexamples, cur_njobs + 1).astype(int)
        # in case there are fewer ranges than jobs
        ranges = np.unique(ranges)
        all_args = []
        if sanity:
            jobs_range = [0]
        else:
            jobs_range = range(len(ranges) - 1)
        for job_id in jobs_range:
            range_start = ranges[job_id]
            range_stop = ranges[job_id + 1]
            args = (job_id, range_start, range_stop,
                    examples[range_start:range_stop], storage_dir, ds_type,
                    nclasses_existence, storage_type, job_chunk_size,
                    edge_class, img_channels, grayscale_as_rgb, augment_sample,
                    not_available_class, sanity, folder_split, folder_size,
                    emnist_preprocess)
            all_args.append(args)
            if not local_multiprocess:
                cur_memsamples = gen_samples(*args)
                if not storage_disk:
                    memsamples[k].extend(cur_memsamples)

        if local_multiprocess:
            from multiprocessing import Pool
            with Pool(cur_njobs) as p:
                p.starmap(gen_samples, all_args)

    print('done')
    cl2let = get_cl2let(emnist_preprocess)
    if not local_multiprocess and not storage_disk:
        return memsamples

    # store general configuration
    if storage_disk:
        with open(data_fname, "wb") as new_data_file:
            pickle.dump(
                (nsamples_train, nsamples_test, nsamples_val,
                 nclasses_existence, img_channels, LETTER_SIZE, IMAGE_SIZE,
                 ntypes, edge_class, not_available_class, total_rows,
                 obj_per_row, sample_nchars, ngenerate, ndirections,
                 exclude_percentage, valid_classes, cl2let), new_data_file)

        # copy the generating script
        script_fname = mainmod.__file__
        dst = shutil.copy(script_fname, storage_dir)
        return None


# %%
if __name__ == "__main__":
    ret = main()
