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
    rotate_deg = 10
    aug = iaa.Sequential(
        [
            iaa.Affine(
                translate_percent={
                    "x": (-0.1, 0.1),
                    "y": (-0.1, 0.1)
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


def load_raw_data(raw_data_fname):
    # load each Person's raw data from which we'll generate samples
    # each person has an image, mask and a label of its feature values.
    # Each label is ['Avatar ID', 'Tilt', 'Background type', 'Clothes type', 'Glasses type' ,'Hair type', 'Mustache type']
    new_data_file = open(raw_data_fname, "rb")
    images, masks, labels, npersons = pickle.load(new_data_file)
    total_bins = 4
    PERSON_SIZE = 112
    IMAGE_SIZE = [PERSON_SIZE * 2, PERSON_SIZE * total_bins]

    images_raw = images
    masks_raw = masks
    labels_raw = labels
    return images_raw, masks_raw, labels_raw, npersons, total_bins, PERSON_SIZE, IMAGE_SIZE


# %% create samples


# create a sample image and label from the lightweight example
def gen_sample(sample_id, is_train, aug_data, grayscale,
               use_natural_background, bg_images, grayscale_as_rgb,
               img_channels, total_bins, images_raw, labels_raw, masks_raw,
               PERSON_SIZE, IMAGE_SIZE, example, augment_sample,
               max_value_features, npersons):
    # start by creating the image background
    if use_natural_background:
        bg_index = example.bg_index
        image = bg_images[bg_index].copy()
        assert (image.max() <= 1)
    else:
        bg_color = example.bg_color
        if grayscale:
            image = bg_color * np.ones(IMAGE_SIZE, dtype=np.float32)
        else:
            image = bg_color * np.ones(
                (IMAGE_SIZE[0], IMAGE_SIZE[1], img_channels), dtype=np.uint8)
    sample_npersons = len(example.persons)
    sample = SimpleNamespace()
    # collect the necessary persons in the infos Namespace and create the sample image
    infos = []
    for person in example.persons:
        person_id = person.id
        s_id = person.s_id
        ims = images_raw
        msks = masks_raw
        lbs = labels_raw
        im = ims[s_id]
        label = lbs[s_id]
        mask = msks[s_id]
        scale = person.scale
        sz = scale * np.array([PERSON_SIZE, PERSON_SIZE])
        sz = sz.astype(np.int)
        h, w = sz
        im = skimage.transform.resize(im, sz, mode='constant')
        if grayscale:
            im = color.rgb2gray(im)
        mask = skimage.transform.resize(mask, sz, mode='constant')
        stx = person.location_x
        endx = stx + w
        sty = person.location_y
        endy = sty + h
        # this is a view into the image and when it changes the image also changes
        part = image[sty:endy, stx:endx]
        if not grayscale:
            mask = mask[:, :, np.newaxis]
            mask = np.concatenate((mask, mask, mask), axis=2)
        rng = mask > 0
        part[rng] = mask[rng] * im[rng] + (1 - mask[rng]) * part[rng]
        info = SimpleNamespace()
        info.person_id = person_id
        info.s_id = s_id
        info.label = label
        info.im = im
        info.mask = mask
        info.stx = stx
        info.endx = endx
        info.sty = sty
        info.endy = endy
        infos.append(info)

    # create the labels
    label = np.tile(max_value_features, (npersons, 1))
    loss_weight_not_exist = np.zeros_like(max_value_features)
    loss_weight_not_exist[0] = 1
    loss_weight = np.tile(loss_weight_not_exist, (npersons, 1))
    for info in infos:
        lbl = info.label
        person_id = lbl[0]
        label[person_id] = lbl
        loss_weight[person_id] = 1
    label = np.array(label)
    label_all = label.flatten()
    loss_weight = loss_weight.flatten()

    label_existence = np.zeros(npersons)
    for info in infos:
        label_existence[info.person_id] = 1
    label_existence = np.array(label_existence)

    # create segmentation of the example we are interested in (object-based attention)
    query_part_id = example.query_part_id
    det_id = example.det_id
    seg = np.zeros_like(image)
    if query_part_id == npersons:
        person_id = example.non_existing_person
        person_features = max_value_features
    else:
        info = infos[query_part_id]
        person_id = info.person_id
        person_features = info.label

        # this is a view into the image and when it changes the image also changes
        part = seg[info.sty:info.endy, info.stx:info.endx]
        mask = info.mask
        im = info.im
        rng = mask > 0
        part[rng] = mask[rng] * im[rng] + (1 - mask[rng]) * part[rng]

    label_task = person_features[det_id]

    # create the instruction for the task
    flag = [person_id, det_id]
    flag = np.array(flag)

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
    sample.label_all = label_all
    sample.label_existence = label_existence.astype(np.int)
    sample.loss_weight = loss_weight
    sample.query_part_id = query_part_id
    sample.seg = seg
    sample.person_features = person_features
    sample.label_task = label_task
    sample.flag = flag
    sample.is_train = is_train
    return sample


# Create several samples from lightweight examples. This function can be considered as a 'job' which can run in parallel
def gen_samples(job_id, range_start, range_stop, examples, storage_dir,
                ds_type, storage_type, raw_data_fname, job_chunk_size,
                img_channels, augment_sample, max_value_features, grayscale,
                use_natural_background, bg_fnames, grayscale_as_rgb,
                folder_split, folder_size):

    images_raw, masks_raw, labels_raw, npersons, total_bins, PERSON_SIZE, IMAGE_SIZE = load_raw_data(
        raw_data_fname)

    bg_images = []
    if use_natural_background:
        for fname in bg_fnames:
            im = imageio.imread(fname)
            im = skimage.transform.resize(im, IMAGE_SIZE, mode='constant')
            if grayscale and len(im.shape) == 3:
                im = color.rgb2gray(im)
            bg_images.append(im)

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
            sample = gen_sample(samid, is_train, aug_data, grayscale,
                                use_natural_background, bg_images,
                                grayscale_as_rgb, img_channels, total_bins,
                                images_raw, labels_raw, masks_raw, PERSON_SIZE,
                                IMAGE_SIZE, examples[rel_id], augment_sample,
                                max_value_features, npersons)
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
    avatars_dir = os.path.join(data_dir, 'avatars')
    raw_data_fname = os.path.join(avatars_dir, 'avatars_6_raw.pkl')
    # split the generated data set to 1000 files per folder
    folder_split = True
    folder_size = 1000
    if storage_disk:
        # the name of the dataset to create
        if cmd_args.extended:
            base_storage_dir = 'extended'
        else:
            base_storage_dir = 'sufficient'
        if storage_type == 'pytorch':
            new_avatars_dir = os.path.join(avatars_dir, 'samples')
            base_samples_dir = os.path.join(new_avatars_dir, base_storage_dir)
            if not os.path.exists(base_samples_dir):
                os.makedirs(base_samples_dir, exist_ok=True)
            storage_dir = base_samples_dir

        data_fname = os.path.join(storage_dir, 'conf')
    else:
        storage_dir = None

    augment_sample = True

    _, _, labels_raw, npersons, total_bins, PERSON_SIZE, IMAGE_SIZE = load_raw_data(
        raw_data_fname)
    nfeatures = len(labels_raw[0])

    grayscale = True
    grayscale_as_rgb = True
    img_channels = 3
    use_natural_background = False
    bg_fnames = []
    if use_natural_background:
        # Can also use backgrounds from Caltech-256 Object Category Dataset (not supplied)
        bg_folders = [
            '../data/images/256_ObjectCategories/241.waterfall',
            '../data/images/256_ObjectCategories/257.clutter'
        ]
        for folder in bg_folders:
            filenames = [os.path.join(folder, f) for f in os.listdir(folder)]
            bg_fnames.extend(filenames)

        if sanity:
            # for quick loading
            bg_fnames = bg_fnames[:10]

    same_max_value = True  # use the same value as the maximum for all features
    npersons_per_example = 2
    generalize = True  # if True exclude 7% of the training data and test only the excluded data (combinatorial generalization)
    gen_hide_many = False  # if True exclude 40% of the training data (combinatorial generalization)
    add_non_gen = True  # add validation set
    if not generalize:
        add_non_gen = False

    # if True, generate multiple examples (image,label) pairs from each image, else generate a single example
    single_feat_to_generate = False
    query_npersons = 2  # if single_feat_to_generate is False then query only query_npersons from the image

    if cmd_args.extended:
        nsamples = 6000  # from each sample multiple examples (images/labels) are created
    else:
        nsamples = 2000  # from each sample multiple examples (images/labels) are created
    train_val_ratio = .8
    use_constant_ntest = True
    nsamples_train = int(train_val_ratio * nsamples)
    if gen_hide_many:
        nsamples_test = 50 * 10  # 10 systematic gen query types, 50 examples per query type
    else:
        if use_constant_ntest:
            nsamples_test = 590
        else:
            nsamples_test = nsamples - nsamples_train
    if sanity:
        nsamples_train = 80
        nsamples_test = 20

    max_value_features = np.max(labels_raw,
                                axis=0)  # the maximum value of each feature
    max_value_features += 1  # does not exist type
    max_value_features = np.array(max_value_features)
    if same_max_value:
        max_value_features = np.max(max_value_features) * np.ones_like(
            max_value_features)

    nclasses = nfeatures * npersons
    nclasses_existence = npersons
    nvalue_features = max_value_features + 1  # values start from 0
    ntypes = np.tile(nvalue_features, (npersons, 1)).flatten()

    valid_features = [
        1, 3, 4, 5, 6
    ]  #ignore ID and background. background is not used in the dataset

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

        # person_x_pairs is the excluded feature and value for a person: [id, (feature id, feature value), (feature id, feature value)...]
        # Features: 'Avatar ID', 'Tilt', 'Background type', 'Clothes type', 'Glasses type' ,'Hair type', 'Mustache type'
        if gen_hide_many:
            person_1_pairs = [1, (5, 1), (3, 3)]
            person_2_pairs = [2, (4, 3), (3, 4)]
            person_3_pairs = [3, (6, 0)]
            person_4_pairs = [4, (3, 2), (4, 4)]
            person_5_pairs = [5, (6, 1)]
            person_0_pairs = [0, (5, 0)]
            test_details = [
                person_1_pairs, person_2_pairs, person_3_pairs, person_4_pairs,
                person_5_pairs, person_0_pairs
            ]
        else:
            person_1_pairs = [1, (5, 1)]
            person_2_pairs = [2, (4, 3)]
            person_3_pairs = [3, (6, 0)]
            if npersons_per_example == 2:
                test_details = [person_1_pairs, person_2_pairs]
            else:
                test_details = [person_1_pairs, person_2_pairs, person_3_pairs]

        # only test on the excluded features
        available_test_details = dict()
        available_test_persons = []
        for person_num in range(len(test_details)):
            person_pairs = test_details[person_num]
            person_id = person_pairs[0]
            available_test_persons.append(person_id)

            person_cons = person_pairs[1:]
            available_test_details[person_id] = person_cons

        # find the valid feature combinations for train
        person_train_valid = [[] for i in range(npersons)]
        for person_id in range(npersons):
            train_valid = labels_raw[:, 0] == person_id
            person_train_valid[person_id] = train_valid

        # find the valid feature combinations for test
        person_test_valid = [[] for i in range(npersons)]
        for person_num in range(len(test_details)):
            person_pairs = test_details[person_num]
            person_id = person_pairs[0]
            person_cons = person_pairs[1:]
            test_valid = np.full(len(labels_raw), False, dtype=bool)
            for detail in person_cons:
                pair_valid = labels_raw[:, 0] == person_id
                det_id, det_value = detail
                pair_valid = np.all(
                    [pair_valid, labels_raw[:, det_id] == det_value], axis=0)

                test_valid = np.any([test_valid, pair_valid], axis=0)

            person_test_valid[person_id] = test_valid
            person_train_valid[person_id] = np.all(
                [person_train_valid[person_id], ~test_valid], axis=0)

        tot_avail = [
            person_train_valid[person_id].sum()
            for person_id in range(npersons)
        ]
        print('number of train images per avatar to choose from')
        print(tot_avail)
        print('total available ratio:')
        tot_avail_ratio = sum(tot_avail) / len(labels_raw)
        print(tot_avail_ratio)

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
            # generate a valid sample with valid labels
            if (generalize and not_test) or (not generalize):
                chosen_persons = prng.choice(npersons,
                                             npersons_per_example,
                                             replace=False)
            else:
                chosen_persons = prng.choice(available_test_persons,
                                             npersons_per_example,
                                             replace=False)

            image_id = [None for _ in range(npersons)]
            s_ids = []
            persons_labels = dict()
            for person_id in chosen_persons:
                if generalize:
                    if not_test:
                        valid = person_train_valid[person_id]
                    else:
                        valid = person_test_valid[person_id]
                else:
                    valid = labels_raw[:, 0] == person_id

                lbs = labels_raw[valid]
                s_id = prng.randint(0, len(lbs))
                s_id = valid.nonzero()[0][s_id]
                label = labels_raw[s_id]
                image_id[person_id] = label
                s_ids.append(s_id)
                persons_labels[person_id] = label

            image_id_hash = str(image_id)
            if image_id_hash in image_ids:
                # avoid generating the same persons' configuration twice
                continue

            # create an example from the the chosen persons
            image_ids.add(image_id_hash)
            if use_natural_background:
                bg_index = prng.randint(0, len(bg_fnames))
            else:
                if grayscale:
                    bg_color = prng.randint(0, 255 + 1) / 255
                else:
                    bg_color = prng.randint(0, 255 + 1)
            imageh, imagew = IMAGE_SIZE
            minscale = .5
            maxscale = 1

            # place the avatars freely on the image but without occlusions
            success = False
            while not success:
                success = True
                persons = []
                valid_location_map = np.full(IMAGE_SIZE, False, dtype=bool)
                sty = int(PERSON_SIZE / 2)
                endy = int(IMAGE_SIZE[0] - PERSON_SIZE / 2)
                stx = int(PERSON_SIZE / 2)
                endx = int(IMAGE_SIZE[1] - PERSON_SIZE / 2)
                valid_location_map[sty:endy, stx:endx] = True
                for samplei in range(len(chosen_persons)):
                    rows, cols = valid_location_map.nonzero()
                    if len(rows) == 0 or len(cols) == 0:
                        success = False
                        continue
                    idx = prng.choice(len(rows))
                    row = rows[idx]
                    col = cols[idx]
                    #from save_image import save_image
                    #save_image(valid_location_map.astype(np.int8),'a4.png')
                    scale = prng.rand() * (maxscale - minscale) + minscale
                    cur_stx = int(col - scale * PERSON_SIZE / 2)
                    cur_endx = int(col + scale * PERSON_SIZE / 2)
                    cur_sty = int(row - scale * PERSON_SIZE / 2)
                    cur_endy = int(row + scale * PERSON_SIZE / 2)
                    val_sty = max(0, int(cur_sty - PERSON_SIZE / 2))
                    val_endy = min(imageh, int(cur_endy + PERSON_SIZE / 2))
                    val_stx = max(0, int(cur_stx - PERSON_SIZE / 2))
                    val_endx = min(imagew, int(cur_endx + PERSON_SIZE / 2))
                    valid_location_map[val_sty:val_endy,
                                       val_stx:val_endx] = False
                    person = SimpleNamespace()
                    person.s_id = s_ids[samplei]
                    person.id = chosen_persons[samplei]
                    person.location_x = cur_stx
                    person.location_y = cur_sty
                    person.scale = scale
                    persons.append(person)

            print(i)

            # generate a single or multiple examples for each generated configuration
            if single_feat_to_generate:
                query_part_ids = [prng.randint(0, len(chosen_persons))]
            else:
                query_part_ids = prng.choice(npersons_per_example,
                                             query_npersons,
                                             replace=False)
                query_part_ids.sort()

            for query_part_id in query_part_ids:
                if (generalize and not_test) or (not generalize):
                    if single_feat_to_generate or is_val or is_test:
                        detail_ids = [prng.choice(valid_features)]
                    else:
                        detail_ids = valid_features
                else:
                    # generalize and test
                    person_id = persons[query_part_id].id
                    test_details = available_test_details[person_id]
                    label = persons_labels[person_id]
                    match = np.full(nfeatures, False, dtype=bool)
                    for detail in test_details:
                        det_id, det_value = detail
                        match[det_id] = label[det_id] == det_value

                    detail_ids = match.nonzero()[0]
                    if single_feat_to_generate:
                        detail_ids = [prng.choice(detail_ids, 1)]

                for det_id in detail_ids:
                    example = SimpleNamespace()
                    example.persons = persons
                    example.query_part_id = query_part_id
                    example.det_id = det_id
                    if use_natural_background:
                        example.bg_index = bg_index
                    else:
                        example.bg_color = bg_color
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
                    storage_type, raw_data_fname, job_chunk_size, img_channels,
                    augment_sample, max_value_features, grayscale,
                    use_natural_background, bg_fnames, grayscale_as_rgb,
                    folder_split, folder_size)
            all_args.append(args)
            if not local_multiprocess:
                cur_memsamples = gen_samples(*args)
                if not storage_disk:
                    memsamples[k].extend(cur_memsamples)

        if local_multiprocess:
            from multiprocessing import Pool
            with Pool(cur_njobs) as p:
                p.starmap(gen_samples, all_args)

    # store general configuration
    if storage_disk:
        with open(data_fname, "wb") as new_data_file:
            pickle.dump((nsamples_train, nsamples_test, nsamples_val,
                         nfeatures, nclasses, nclasses_existence, ntypes,
                         img_channels, IMAGE_SIZE), new_data_file)

        # copy the generating script
        script_fname = mainmod.__file__
        dst = shutil.copy(script_fname, storage_dir)

    print('done')


if __name__ == '__main__':
    main()
