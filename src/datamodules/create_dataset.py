from torch.utils.data import Dataset
import numpy as np
import torch
import os
import SimpleITK as sitk
import torchio as tio
import pickle
sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
from multiprocessing import Manager

# this controls the split.
RANDOM_SEED = 1


class ResizeIntensity(tio.Transform):
    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = target_shape

    def apply_transform(self, subject):
        # Iterate through each image in the subject
        for image_name, image in subject.get_images_dict(intensity_only=False).items():
            if image.type == tio.INTENSITY:
                # Save the original shape
                original_shape = image.spatial_shape
                subject['original_shape'] = original_shape

                # Resize the image to the target shape
                resize_transform = tio.Resize(self.target_shape)
                image.set_data(resize_transform(image.data))

            else:
                # For LabelMap, do not resize
                pass

        return subject

class UndoResizeIntensity(tio.Transform):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        # Iterate through each image in the subject
        for image_name, image in subject.get_images_dict(intensity_only=False).items():
            if image.type == tio.INTENSITY:
                # Check if the original shape is stored in the metadata
                if 'original_shape' in image:
                    original_shape = image['original_shape']

                    # Resize the image back to the original shape
                    resize_transform =tio.Resize(original_shape)
                    image.set_data(resize_transform(image.data))

                else:
                    raise ValueError(f"Original shape not found for image {image_name}")

            else:
                # For LabelMap, do not resize
                pass


class CropToBrain(tio.Transform):
    def apply_transform(self, subject):
        # Find the ScalarImage and compute cropping bounds
        scalar_image = None
        for image in subject.get_images(intensity_only=True):
            if isinstance(image, tio.ScalarImage):
                scalar_image = image
                break

        if scalar_image is None:
            raise ValueError("No ScalarImage found in the subject.")

        data = scalar_image.data

        non_zero_indices = np.where(data != 0)
        min_indices = np.min(non_zero_indices, axis=1)
        max_indices = np.max(non_zero_indices, axis=1)

        for image_name, image in subject.get_images_dict(intensity_only=False).items():
            data = image.data
            cropped_data= data[:,
                                            min_indices[1]:max_indices[1]+1,
                                            min_indices[2]:max_indices[2]+1,
                                            min_indices[3]:max_indices[3]+1]
            if isinstance(image, tio.ScalarImage):
                cropped_image = tio.ScalarImage(tensor=cropped_data)
            elif isinstance(image, tio.LabelMap):
                cropped_image = tio.LabelMap(tensor=cropped_data)
            else:
                raise TypeError(f'{type(image)} is not known')
            subject[image_name] = cropped_image

        return subject

class MRIProcessor:
    def __init__(self, root_dir, image_new_size):
        self.root_dir = root_dir
        self.image_new_size = image_new_size
    def load_images(self):
        subjects = []
        transforms = tio.Compose([
            tio.Resample((1, 1, 1)),  # Resample to 1mm^3
            tio.RescaleIntensity(percentiles=(0.05, 99.5)),
            tio.ToCanonical(),  # Convert to canonical orientation: RAS
            CropToBrain(),
            ResizeIntensity(self.image_new_size)
        ])
        for subdir, _, files in os.walk(self.root_dir):
            if len(files) == 0:
                continue
            for filename in files:
                if filename.endswith('.nii') or filename.endswith('.nii.gz') and not ('seg' in filename):
                    image_path = os.path.join(subdir, filename)
                    source_label = os.path.basename(subdir)
                    subject_dict = {
                        'vol': tio.ScalarImage(image_path),
                        'source_label': source_label,
                        'id': filename
                    }

                    name = filename.split('t1')[0] if len(filename.split('t1')) > 1 else filename.split('T1')[0] 
                    label_file_path = os.path.join(subdir , name + 'seg.nii.gz')
                    if os.path.exists(label_file_path):
                        subject_dict['seg'] = tio.LabelMap(label_file_path)
                    subject = tio.Subject(**subject_dict)
                    transformed_subject = transforms(subject)
                    subjects.append(transformed_subject)
                    print(f'done with file: {filename}')
        return subjects


    @staticmethod
    def split_data(subjects, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        generator = torch.Generator().manual_seed(RANDOM_SEED)
        total_samples = len(subjects)
        random_indexes = torch.randperm(total_samples, generator=generator)
        subjects = list(map(subjects.__getitem__, random_indexes))
        train_samples = int(total_samples * train_ratio)
        val_samples = int(total_samples * val_ratio)
        reminders = total_samples - train_samples - val_samples
        if test_ratio == 0:
            val_samples += reminders

        train_data = subjects[:train_samples]
        val_data = subjects[train_samples:train_samples + val_samples]
        test_data = subjects[train_samples + val_samples:]

        return train_data, val_data, test_data

    @staticmethod
    def save_split_info(train_data, val_data, test_data, file_path):
        split_info = {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }
        with open(file_path, 'wb') as f:
            pickle.dump(split_info, f)

    @staticmethod
    def load_split_info(file_path):
        with open(file_path, 'rb') as f:
            split_info = pickle.load(f)
        return split_info["train"], split_info["val"], split_info["test"]


#this method to handle the special cases of this project. I think this is only needed for abnormality evaluation
    def modify_subjects_for_this_project(self):
        subjects = self.load_images()

        for subject in subjects:
            if 'seg' in subject:
                subject['vol_orig'] = subject['vol']
                subject['seg_orig'] = subject['seg']
                subject['age'] = 60
                subject['ID'] = subject['id']
                subject['label'] = torch.tensor([0])
                subject['stage'] = 'val'
                subject['seg_available'] = False
                subject['mask_orig'] = subject['vol_orig']

        return subjects


def Train(csv, cfg, preload=True):
    subjects = []
    for _, sub in csv.iterrows():
        subject_dict = {
            'vol': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            'age': sub.age,
            'ID': sub.img_name,
            'label': sub.label,
            'Dataset': sub.setname,
            'stage': sub.settype,
            'path': sub.img_path
        }
        if sub.mask_path != None:  # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path, reader=sitk_reader)
        else:  # if we don't have masks, we create a mask from the image
            subject_dict['mask'] = tio.LabelMap(tensor=tio.ScalarImage(sub.img_path, reader=sitk_reader).data > 0)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)

        ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg))
        ds = vol2slice(ds, cfg, slice=slice_ind, seq_slices=seq_slices)
    return ds


def Eval(csv, cfg):
    subjects = []
    for _, sub in csv.iterrows():
        if sub.mask_path is not None and tio.ScalarImage(sub.img_path, reader=sitk_reader).shape != tio.ScalarImage(
                sub.mask_path, reader=sitk_reader).shape:
            print(
                f'different shapes of vol and mask detected. Shape vol: {tio.ScalarImage(sub.img_path, reader=sitk_reader).shape}, shape mask: {tio.ScalarImage(sub.mask_path, reader=sitk_reader).shape} \nsamples will be resampled to the same dimension')

        subject_dict = {
            'vol': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            'vol_orig': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            # we need the image in original size for evaluation
            'age': sub.age,
            'ID': sub.img_name,
            'label': sub.label,
            'Dataset': sub.setname,
            'stage': sub.settype,
            'seg_available': False,
            'path': sub.img_path}
        if sub.seg_path != None:  # if we have segmentations
            subject_dict['seg'] = tio.LabelMap(sub.seg_path, reader=sitk_reader),
            subject_dict['seg_orig'] = tio.LabelMap(sub.seg_path,
                                                    reader=sitk_reader)  # we need the image in original size for evaluation
            subject_dict['seg_available'] = True
        if sub.mask_path != None:  # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path, reader=sitk_reader)
            subject_dict['mask_orig'] = tio.LabelMap(sub.mask_path,
                                                     reader=sitk_reader)  # we need the image in original size for evaluation
        else:
            tens = tio.ScalarImage(sub.img_path, reader=sitk_reader).data > 0
            subject_dict['mask'] = tio.LabelMap(tensor=tens)
            subject_dict['mask_orig'] = tio.LabelMap(tensor=tens)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg))
    return ds


## got it from https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/12
class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, subject):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (subject)


class preload_wrapper(Dataset):
    def __init__(self, ds, cache, augment=None):
        self.cache = cache
        self.ds = ds
        self.augment = augment

    def reset_memory(self):
        self.cache.reset()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        if self.cache.is_cached(index):
            subject = self.cache.get(index)
        else:
            subject = self.ds.__getitem__(index)
            self.cache.cache(index, subject)
        if self.augment:
            subject = self.augment(subject)
        return subject


class vol2slice(Dataset):
    def __init__(self, ds, cfg, onlyBrain=False, slice=None, seq_slices=None):
        self.ds = ds
        self.onlyBrain = onlyBrain
        self.slice = slice
        self.seq_slices = seq_slices
        self.counter = 0
        self.ind = None
        self.cfg = cfg

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        subject = self.ds.__getitem__(index)
        if self.onlyBrain:
            start_ind = None
            for i in range(subject['vol'].data.shape[-1]):
                if subject['mask'].data[0, :, :, i].any() and start_ind is None:  # only do this once
                    start_ind = i
                if not subject['mask'].data[0, :, :,
                       i].any() and start_ind is not None:  # only do this when start_ind is set
                    stop_ind = i
            low = start_ind
            high = stop_ind
        else:
            low = 0
            high = subject['vol'].data.shape[-1]
        if self.slice is not None:
            self.ind = self.slice
            if self.seq_slices is not None:
                low = self.ind
                high = self.ind + self.seq_slices
                self.ind = torch.randint(low, high, size=[1])
        else:
            if self.cfg.get('unique_slice', False):  # if all slices in one batch need to be at the same location
                if self.counter % self.cfg.batch_size == 0 or self.ind is None:  # only change the index when changing to new batch
                    self.ind = torch.randint(low, high, size=[1])
                self.counter = self.counter + 1
            else:
                self.ind = torch.randint(low, high, size=[1])

        subject['ind'] = self.ind

        subject['vol'].data = subject['vol'].data[..., self.ind]
        # subject['mask'].data = subject['mask'].data[..., self.ind]

        return subject


def get_transform(cfg):  # only transforms that are applied once before preloading
    transforms = tio.Compose([
        tio.Resample((1, 1, 1)),  # Resample to 1mm^3
        tio.RescaleIntensity(percentiles=(0.05, 99.5)),
        tio.ToCanonical()  # Convert to canonical orientation: RAS
    ])

    return transforms


def get_augment(cfg):  # augmentations that may change every epoch
    augmentations = []

    # individual augmentations
    if cfg.get('random_bias', False):
        augmentations.append(tio.RandomBiasField(p=0.25))
    if cfg.get('random_motion', False):
        augmentations.append(tio.RandomMotion(p=0.1))
    if cfg.get('random_noise', False):
        augmentations.append(tio.RandomNoise(p=0.5))
    if cfg.get('random_ghosting', False):
        augmentations.append(tio.RandomGhosting(p=0.5))
    if cfg.get('random_blur', False):
        augmentations.append(tio.RandomBlur(p=0.5))
    if cfg.get('random_gamma', False):
        augmentations.append(tio.RandomGamma(p=0.5))
    if cfg.get('random_elastic', False):
        augmentations.append(tio.RandomElasticDeformation(p=0.5))
    if cfg.get('random_affine', False):
        augmentations.append(tio.RandomAffine(p=0.5))
    if cfg.get('random_flip', False):
        augmentations.append(tio.RandomFlip(p=0.5))

    # policies/groups of augmentations
    if cfg.get('aug_intensity', False):  # augmentations that change the intensity of the image rather than the geometry
        augmentations.append(tio.RandomGamma(p=0.5))
        augmentations.append(tio.RandomBiasField(p=0.25))
        augmentations.append(tio.RandomBlur(p=0.25))
        augmentations.append(tio.RandomGhosting(p=0.5))

    augment = tio.Compose(augmentations)
    return augment


def sitk_reader(path):
    image_nii = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if not 'mask' in str(path) and not 'seg' in str(path):  # only for volumes / scalar images
        image_nii = sitk.CurvatureFlow(image1=image_nii, timeStep=0.125, numberOfIterations=3)
    vol = sitk.GetArrayFromImage(image_nii).transpose(2, 1, 0)
    return vol, None
