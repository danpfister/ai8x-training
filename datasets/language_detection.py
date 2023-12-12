###################################################################################################
#
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Classes and functions used to utilize Speech Commands dataset.
"""
import errno
import hashlib
import os
import tarfile
import urllib
import urllib.error
import urllib.request
import warnings

import numpy as np
import torch
from torch.utils.model_zoo import tqdm  # type: ignore # tqdm exists in model_zoo
from torchvision import transforms

import librosa
import librosa.display
from PIL import Image

import ai8x


class LanguageDetection(torch.utils.data.Dataset):
    """

    Args:
        root (string): Root directory of dataset where ``SpeechCom/processed/train.pt``
            ``SpeechCom/processed/val.pt`` and  ``SpeechCom/processed/test.pt`` exist.
        classes(array): List of keywords to be used.
        d_type(string): Option for the created dataset. ``train`` is to create dataset
            from ``training.pt``, ``val`` is to create dataset from ``val.pt``, ``test``
            is to create dataset from ``test.pt``.
        n_augment(int, optional): Number of samples added to the dataset from each sample
            by random modifications, i.e. stretching, shifting and random noise addition.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    fs = 16000
    training_file = 'train.pt'
    test_file = 'test.pt'
    validation_file = 'val.pt'

    class_dict = {'en': 0, 'de': 1, 'es': 2}

    def __init__(self, root, classes, d_type, n_augment=0, transform=None, download=False):
        self.root = root
        self.classes = classes
        self.d_type = d_type
        self.transform = transform
        self.n_augment = n_augment

        if download: self.__download()

        if self.d_type == 'train':
            data_file = self.training_file
        elif self.d_type == 'test':
            data_file = self.test_file
        elif self.d_type == 'val':
            data_file = self.validation_file
        else:
            print(f'Unknown data type: {d_type}')
            return

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        #self.__filter_classes()

    @property
    def raw_folder(self):
        """Folder for the raw data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        """Folder for the processed data.
        """
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def __gen_datasets(self):
        print('Generating dataset from raw data samples.')
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            lst = os.listdir(self.raw_folder)
            labels = [d for d in lst if os.path.isdir(os.path.join(self.raw_folder, d)) and
                      d[0].isalpha()]
            train_images = []
            val_images = []
            test_images = []
            train_labels = []
            val_labels = []
            test_labels = []
            for i, label in enumerate(labels):
                print(f'\tProcessing the label: {label}. {i+1} of {len(labels)}')
                records = os.listdir(os.path.join(self.raw_folder, label))
                records = sorted(records)
                for record in records:
                    record_pth = os.path.join(self.raw_folder, label, record)
                    y, _ = librosa.load(record_pth, offset=0, sr=22050)

                    mfcc = compute_mfcc(audio=y, sr=22050)
                    
                    if mfcc is not None:
                        if hash(record) % 10 < 7:
                            train_images.append(mfcc)
                            train_labels.append(label)
                        elif hash(record) % 10 < 9:
                            val_images.append(mfcc)
                            val_labels.append(label)
                        else:
                            test_images.append(mfcc)
                            test_labels.append(label)

            print(f"number of train images elements: {len(train_images)} and each has shape: {train_images[0].shape}")

            train_images = torch.from_numpy(np.array(train_images))
            val_images = torch.from_numpy(np.array(val_images))
            test_images = torch.from_numpy(np.array(test_images))

            label_dict = dict(zip(labels, range(len(self.class_dict))))
            train_labels = torch.from_numpy(np.array([label_dict[ll] for ll in train_labels]))
            val_labels = torch.from_numpy(np.array([label_dict[ll] for ll in val_labels]))
            test_labels = torch.from_numpy(np.array([label_dict[ll] for ll in test_labels]))

            train_set = (train_images, train_labels)
            val_set = (val_images, val_labels)
            test_set = (test_images, test_labels)

            torch.save(train_set, os.path.join(self.processed_folder, self.training_file))
            torch.save(val_set, os.path.join(self.processed_folder, self.validation_file))
            torch.save(test_set, os.path.join(self.processed_folder, self.test_file))

        print('Dataset created!')

    def __download(self):
        '''
        if self.__check_exists():
            return
        '''

        self.__makedir_exist_ok(self.raw_folder)
        self.__makedir_exist_ok(self.processed_folder)

        self.__gen_datasets()

    def __check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) |
                os.path.exists(os.path.join(self.processed_folder, self.test_file)) |
                os.path.exists(os.path.join(self.processed_folder, self.validation_file)))

    def __makedir_exist_ok(self, dirpath):
        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index].numpy(), int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (2, 1, 0)))

        #print(f"get item shape: {img.size}")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


# functions to convert audio data to image by mel spectrogram technique and augment data.


def compute_mfcc(audio, sr):
    """Converts audio to an image form by taking mel spectrogram.
    """
    duration = librosa.get_duration(y=audio, sr=sr)
    desired_duration = 5
    if duration < desired_duration:
        num_repeats = int(np.ceil(desired_duration / duration))
        repeated_audio = np.tile(audio, num_repeats)
        audio = repeated_audio[:int(sr*desired_duration)]
    elif duration > desired_duration:
        audio = audio[:int(sr*desired_duration)]

    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=64, n_fft=2048, hop_length=1024, n_mels=64)
    delta = librosa.feature.delta(mfcc, order=1)
    deltadelta = librosa.feature.delta(mfcc, order=2)

    all = np.stack((mfcc, delta, deltadelta), axis=0).astype(np.uint8)

    return all


def load_audio_file(file_path):
    """Loads audio data from specified file location.
    """
    input_length = 16000
    audio = librosa.core.load(file_path)[0]  # sr=16000
    if len(audio) > input_length:
        audio = audio[:input_length]
    else:
        audio = np.pad(audio, (0, max(0, input_length - len(audio))), "constant")
    return audio


def add_white_noise(audio, noise_var_coeff):
    """Adds zero mean Gaussian noise to image with specified variance.
    """
    coeff = noise_var_coeff * np.mean(np.abs(audio))
    noisy_audio = audio + coeff * np.random.randn(len(audio))
    return noisy_audio


def shift(audio, shift_sec, fs):
    """Shifts audio.
    """
    shift_count = int(shift_sec * fs)
    return np.roll(audio, shift_count)


def stretch(audio, rate=1.):
    """Stretches audio with specified ratio.
    """
    input_length = 16000
    audio2 = librosa.effects.time_stretch(audio, rate=rate)
    if len(audio2) > input_length:
        audio2 = audio2[:input_length]
    else:
        audio2 = np.pad(audio2, (0, max(0, input_length - len(audio2))), "constant")

    return audio2


def augment(audio, fs, verbose=False):
    """Augments audio by adding random noise, shift and stretch ratio.
    """
    random_noise_var_coeff = np.random.uniform(0, 1)
    random_shift_time = np.random.uniform(-0.1, 0.1)
    random_stretch_coeff = np.random.uniform(0.8, 1.3)

    aug_audio = stretch(audio, random_stretch_coeff)
    aug_audio = shift(aug_audio, random_shift_time, fs)
    aug_audio = add_white_noise(aug_audio, random_noise_var_coeff)
    if verbose:
        print(f'random_noise_var_coeff: {random_noise_var_coeff:.2f}\nrandom_shift_time: \
                {random_shift_time:.2f}\nrandom_stretch_coeff: {random_stretch_coeff:.2f}')
    return aug_audio


def augment_multiple(audio, fs, n_augment, verbose=False):
    """Calls `augment` function for n_augment times for given audio data.
    Finally the original audio is added to have (n_augment+1) audio data.
    """
    aug_audio = [augment(audio, fs, verbose) for i in range(n_augment)]
    aug_audio.insert(0, audio)
    return aug_audio


def languagedetect_get_datasets(data, load_train=True, load_test=True, num_classes=6):
    """
    Load the SpeechCom v0.02 dataset
    (https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz).

    The dataset originally includes 30 keywords. A dataset is formed with 7 classes which includes
    6 of the original keywords ('up', 'down', 'left', 'right', 'stop', 'go') and the rest of the
    dataset is used to form the last class, i.e class of the others.
    The dataset is split into training, validation and test sets. 80:10:10 training:validation:test
    split is used by default.

    Data is augmented to 5x duplicate data by randomly stretch, shift and randomly add noise where
    the stretching coefficient, shift amount and noise variance are randomly selected between
    0.8 and 1.3, -0.1 and 0.1, 0 and 1, respectively.
    """
    (data_dir, args) = data

    if num_classes == 3:
        classes = ['en', 'de', 'es']
    else:
        raise ValueError(f'Unsupported num_classes {num_classes}')

    transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=args)
    ])

    if load_train:
        train_dataset = LanguageDetection(root=data_dir, classes=classes, d_type='train', n_augment=0,
                                  transform=transform, download=True)
    else:
        train_dataset = None

    if load_test:
        test_dataset = LanguageDetection(root=data_dir, classes=classes, d_type='val', n_augment=0,
                                 transform=transform, download=True)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset

def language_detection_get_datasets(data, load_train=True, load_test=True):
    return languagedetect_get_datasets(data, load_train, load_test, num_classes=3)


datasets = [
    {
        'name': 'LanguageDetection',
        'input': (3, 64, 108),
        'output': (0, 1, 2),
        'weight': (1, 1, 1),
        'loader': language_detection_get_datasets,
    },
]
