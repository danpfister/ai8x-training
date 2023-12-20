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
import random

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

    fs = 22050
    training_file = 'train.pt'
    test_file = 'test.pt'
    validation_file = 'val.pt'

    class_dict = {'de': 0, 'en': 1, 'es': 2}

    def __init__(self, root, classes, d_type, n_augment=0, transform=None, download=False):
        self.root = root
        self.classes = classes
        self.d_type = d_type
        self.transform = transform
        self.n_augment = n_augment
        self.debug = False # if True, dataset is processed everytime at launch

        if download: self.__download()

        if self.d_type == 'train':
            data_file = self.training_file
        elif self.d_type == 'test':
            data_file = self.test_file
        elif self.d_type == 'val':
            raise Exception("val accessed")
            data_file = self.validation_file
        else:
            print(f'Unknown data type: {d_type}')
            return

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

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
            languages = os.listdir(self.raw_folder)
            languages = sorted(languages)
            train_images = []
            val_images = []
            test_images = []
            train_labels = []
            val_labels = []
            test_labels = []
            for i, language in enumerate(languages):
                print(f'\tProcessing the label: {language}. {i+1} of {len(languages)}')
                audio_names = os.listdir(os.path.join(self.raw_folder, language))
                audio_names = random.sample(audio_names, 1000)
                for j, audio_name in enumerate(audio_names):
                    print(f"\t\tProcessing audio file {j+1} of {len(audio_names)}")
                    audio_file_path = os.path.join(self.raw_folder, language, audio_name)
                    audio, _ = librosa.load(audio_file_path, offset=0, sr=self.fs)
                    
                    if self.n_augment != 0: augmented_audios = augment_multiple(audio=audio, fs=self.fs, n_augment=self.n_augment)
                    else: augmented_audios = [audio]

                    for augmented_audio in augmented_audios:
                        mfcc = compute_mfcc(audio=augmented_audio, sr=self.fs)
                        
                        if hash(audio_name) % 10 < 8:
                            train_images.append(mfcc)
                            train_labels.append(language)
                        else:
                            test_images.append(mfcc)
                            test_labels.append(language)
                        '''
                        elif hash(audio_name) % 10 < 9:
                            val_images.append(mfcc)
                            val_labels.append(language)
                        '''

            assert len(train_images) == len(train_labels)
            assert len(val_images) == len(val_labels)
            assert len(test_images) == len(test_labels)
            
            if self.debug: print(f"data has shape {train_images[0].shape}")
            if self.debug:
                print(f"train has {len(train_images)} samples")
                print(f"\tof which {train_labels.count('de')} are de")
                print(f"\tof which {train_labels.count('en')} are en")
                print(f"\tof which {train_labels.count('es')} are es")
                print(f"val has {len(val_images)} samples")
                print(f"\tof which {val_labels.count('de')} are de")
                print(f"\tof which {val_labels.count('en')} are en")
                print(f"\tof which {val_labels.count('es')} are es")
                print(f"test has {len(test_images)} samples")
                print(f"\tof which {test_labels.count('de')} are de")
                print(f"\tof which {test_labels.count('en')} are en")
                print(f"\tof which {test_labels.count('es')} are es")

            train_images = torch.from_numpy(np.array(train_images))
            val_images = torch.from_numpy(np.array(val_images))
            test_images = torch.from_numpy(np.array(test_images))

            # change this if some classes are not used
            label_dict = self.class_dict # dict(zip(languages, range(len(self.class_dict))))
            if self.debug: print(f"mapping languages according to {label_dict}")
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
        if self.__check_exists() and not self.debug:
            print("processed files already exist")
            return

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

        img = 255 * (img - img.min()) / (img.max() - img.min())

        # img = np.random.rand(3, 16, 251)
        # stored as CxHxW but PIL expects HxWxC
        img = Image.fromarray(np.transpose(img, (1, 2, 0)), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        # if self.debug: print(f"got item of shape {img.shape} of label {target}")

        return img, target


# functions to convert audio data to image by mel spectrogram technique and augment data.


def compute_mfcc(audio, sr):
    """fixes audio length to 5 seconds and computes mfcc, delta and deltadelta of mfcc 

    Args:
        audio (_type_): audio
        sr (_type_): sample rate

    Returns:
        np.ndarray: mfcc, delta and deltadelta stacked to shape (3x16x251)
    """
    duration = librosa.get_duration(y=audio, sr=sr)
    desired_duration = 10
    if duration < desired_duration:
        num_repeats = int(np.ceil(desired_duration / duration))
        repeated_audio = np.tile(audio, num_repeats)
        audio = repeated_audio[:int(sr*desired_duration)]
    elif duration > desired_duration:
        audio = audio[:int(sr*desired_duration)]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=16, hop_length=int(sr*0.02),
                   n_fft=2048, win_length=2048, window='hann',
                   center=True, pad_mode='constant')
    delta = librosa.feature.delta(mfcc, order=1)
    deltadelta = librosa.feature.delta(mfcc, order=2)

    stacked = np.stack((mfcc, delta, deltadelta), axis=0)

    return stacked


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
        test_dataset = LanguageDetection(root=data_dir, classes=classes, d_type='test', n_augment=0,
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
        'input': (3, 16, 251),
        'output': (0, 1, 2),
        'weight': (1, 1, 1),
        'loader': language_detection_get_datasets,
    },
]
