import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
import  torchvision
import numpy as np
import nibabel as nib
import glob
# import torchio as tio
join = os.path.join

HGG = []
LGG = []
for i in range(0, 260):
    HGG.append(str(i).zfill(3))
for i in range(336, 370):
    HGG.append(str(i).zfill(3))
for i in range(260, 336):
    LGG.append(str(i).zfill(3))

mask_array = np.array([[True, False, False], [False, True, False], [False, False, True],
                      [True, True, False], [True, False, True], [False, True, True],
                      [True, True, True]])

class NPC_loadall_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=2): #The name of the file that lists the training data

        patients_dir = glob.glob(join(root, '*.npy')) # find all files in the vol directory that end with _vol.npy
        patients_dir.sort(key=lambda x: x.split(os.path.sep)[-1][:-8])  # without the _vol.npy suffix.
        print('###############', len(patients_dir))
        n_patients = len(patients_dir)
        pid_idx = np.arange(n_patients)
        np.random.seed(0)
        np.random.shuffle(pid_idx)

        volpaths = []
        class_labels = pd.read_excel(r'Train_Total.xlsx', sheet_name="Sheet1", header=None)
        class_labels = dict(zip(class_labels.iloc[:, 0], class_labels.iloc[:, 1]))
        self.class_labels = []
        for idx in pid_idx:
            volpaths.append(patients_dir[idx])
            self.class_labels.append(class_labels[patients_dir[idx].split(os.path.sep)[-1].strip('.npy')])
        datalist = [x.split(os.path.sep)[-1].strip('.npy') for x in volpaths]


        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls

        if modal == 't1ce':
            self.modal_ind = np.array([0])
        elif modal == 't1':
            self.modal_ind = np.array([1])
        elif modal == 't2':
            self.modal_ind = np.array([2])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.load(volpath)
        y = self.class_labels[index]
        y = np.array(y)
        x, y = x[None, ...], y[None, ...]

        x = self.transforms([x])[0]

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        # _, H, W, Z, _ = np.shape(x)
        y = np.reshape(y, (-1))
        y = y.astype(int)
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, -1))
        # yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask_idx = np.random.choice(7, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        return x, y, mask, name

    def __len__(self):
        return len(self.volpaths)


class NPC_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=2, label=None):

        patients_dir = glob.glob(join(root, '*.npy'))
        patients_dir.sort(key=lambda x: x.split(os.path.sep)[-1][:-8])
        n_patients = len(patients_dir)
        pid_idx = np.arange(n_patients)

        volpaths = []
        class_labels = pd.read_excel(r'Test_Total.xlsx', sheet_name="Sheet1", header=None, dtype={0: str})
        class_labels = dict(zip(class_labels.iloc[:, 0], class_labels.iloc[:, 1]))
        self.class_labels = []
        for idx in pid_idx:
            if label is not None:
                if label != class_labels[patients_dir[idx].split(os.path.sep)[-1].strip('.npy')]:
                    continue
            volpaths.append(patients_dir[idx])
            # self.class_labels.append(class_labels[int(patients_dir[idx].split(os.path.sep)[-1].strip('.npy'))])
            self.class_labels.append(class_labels[patients_dir[idx].split(os.path.sep)[-1].strip('.npy')])
        datalist = [x.split(os.path.sep)[-1].strip('.npy') for x in volpaths]


        self.num_cls = num_cls
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        print(self.names)

        if modal == 't1ce':
            self.modal_ind = np.array([0])
        elif modal == 't1':
            self.modal_ind = np.array([1])
        elif modal == 't2':
            self.modal_ind = np.array([2])
        elif modal == 't1t2':
            self.modal_ind = np.array([1, 2])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]

        x = np.load(volpath)
        y = self.class_labels[index]
        y = np.array(y)

        x, y = x[None, ...], y[None, ...]
        x = self.transforms([x])[0]

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))
        y = np.reshape(y, (-1))

        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        # yo = torch.squeeze(torch.from_numpy(yo), dim=0)
        return x, y, name

    def __len__(self):
        return len(self.volpaths)

