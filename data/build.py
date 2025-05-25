import logging
import random
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
import h5py
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from config import parse_option
args, config = parse_option()

def applyPCA(X, numComponents=48):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, num_classes, num_samples_per_class):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class

        self.class_indices = [[] for _ in range(num_classes)]
        for idx, (_, label) in enumerate(dataset):
            self.class_indices[label].append(idx)

        self.class_indices = [np.array(indices) for indices in self.class_indices]
        self.current_class_indices = [0] * num_classes

    def __iter__(self):
        num_batches = len(self.dataset) // (self.num_classes * self.num_samples_per_class)
        for _ in range(num_batches):
            batch = []
            for class_idx in range(self.num_classes):
                if self.current_class_indices[class_idx] + self.num_samples_per_class > len(
                        self.class_indices[class_idx]):
                    np.random.shuffle(self.class_indices[class_idx])
                    self.current_class_indices[class_idx] = 0
                start_idx = self.current_class_indices[class_idx]
                end_idx = start_idx + self.num_samples_per_class
                batch.extend(self.class_indices[class_idx][start_idx:end_idx])
                self.current_class_indices[class_idx] += self.num_samples_per_class
            yield batch

    def __len__(self):
        return len(self.dataset) // (self.num_classes * self.num_samples_per_class)

def get_data(dataset_name='Augsburg_HSI_SAR'):

    if dataset_name == 'Augsburg_HSI_SAR':
        logging.info('Augsburg_HSI_SAR......')
        hsi = sio.loadmat('./datasets/Augsburg/HSI_data.mat')['HSI_data'].astype(np.float32)
        labels = sio.loadmat('./datasets/Augsburg/All_Label.mat')['All_Label'].astype(np.int64)
        sar = sio.loadmat('./datasets/Augsburg/SAR_data.mat')['SAR_data'].astype(np.float32)

        if config.DATA.N_PCA != -1:
            hsi, _ = applyPCA(hsi, config.DATA.N_PCA)

        data_list = [hsi, sar]


    elif dataset_name == 'Trento':
        logging.info('Trento......')
        hsi = sio.loadmat('./datasets/Trento/HSI_data.mat')['HSI_data'].astype(np.float32)
        labels = sio.loadmat('./datasets/Trento/All_Label.mat')['All_Label'].astype(np.int64)
        lidar = sio.loadmat('./datasets/Trento/LiDAR_data.mat')['LiDAR_data'].astype(np.float32)

        if config.DATA.N_PCA != -1:
            hsi, _ = applyPCA(hsi, config.DATA.N_PCA)

        lidar = lidar.reshape(lidar.shape[0], lidar.shape[1], -1)
        data_list = [hsi, lidar]


    elif dataset_name == 'Houston':
        logging.info('Houston......')
        hsi = sio.loadmat('./datasets/Houston/HSI_data.mat')['HSI_data'].astype(np.float32)
        labels = sio.loadmat('./datasets/Houston/All_Label.mat')['All_Label'].astype(np.int64)
        lidar = sio.loadmat('./datasets/Houston/LiDAR_data.mat')['LiDAR_data'].astype(np.float32)

        if config.DATA.N_PCA != -1:
            hsi, _ = applyPCA(hsi, config.DATA.N_PCA)

        lidar = lidar.reshape(lidar.shape[0], lidar.shape[1], -1)
        data_list = [hsi, lidar]

    else:
        print('No such dataset')

    return data_list, labels

def StandardScaler_img(data):
    data_ = np.zeros_like(data)
    scaler = StandardScaler()
    for i in range(data.shape[-1]):
        data_[:, :, i] = scaler.fit_transform(data[:, :, i])
    return data_


def build_patch_data(Data, gt=None, patchsize=9, num_samples_per_class=40):

    gt_flatten = np.reshape(gt, -1)
    train_mask = np.zeros_like(gt_flatten)
    for j in range(gt.max()):
        pos = np.where(gt_flatten == j + 1)[0]
        pos_id = random.sample(range(len(pos)), num_samples_per_class)
        train_mask[np.array(pos[pos_id])] = 1
    train_mask = np.reshape(train_mask, (gt.shape[0], gt.shape[1]))
    train_mask = train_mask.astype(np.bool_)
    TR_map = np.multiply(train_mask, gt)
    TE_map = gt


    data_pad = []
    for data in Data: ## 都是3个维度的
        bands = data.shape[-1]
        temp = data[:, :, 0]
        pad_width = np.floor(patchsize / 2)
        pad_width = np.int32(pad_width)
        temp2 = np.pad(temp, pad_width, 'symmetric')
        [h_pad, w_pad] = temp2.shape
        data_pad_ = np.empty((h_pad, w_pad, bands), dtype='float32')
        for i in range(bands):
            temp = data[:, :, i]
            pad_width = np.floor(patchsize / 2)
            pad_width = np.int32(pad_width)
            temp2 = np.pad(temp, pad_width, 'symmetric')
            data_pad_[:, :, i] = temp2
        data_pad.append(data_pad_)


    data_patch_train = [np.empty((sum(sum(TR_map != 0)), data.shape[-1], patchsize, patchsize), dtype='float32') for data in Data]
    data_patch_test = [np.empty((sum(sum(TE_map != 0)), data.shape[-1], patchsize, patchsize), dtype='float32') for data in Data]


    ind1, ind2 = np.where(TR_map != 0)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i, data_pad_ in enumerate(data_pad):
        for j in range(len(ind1)):
            patch = data_pad_[(ind3[j] - pad_width):(ind3[j] + pad_width + 1),
                    (ind4[j] - pad_width):(ind4[j] + pad_width + 1), :]

            patch = np.reshape(patch, (patchsize * patchsize, -1))
            patch = np.transpose(patch)
            patch = np.reshape(patch, (-1, patchsize, patchsize))

            data_patch_train[i][j, :, :, :] = patch


    ind1, ind2 = np.where(TE_map != 0)
    ind3 = ind1 + pad_width
    ind4 = ind2 + pad_width
    for i, data_pad_ in enumerate(data_pad):
        for j in range(len(ind1)):
            patch = data_pad_[(ind3[j] - pad_width):(ind3[j] + pad_width + 1),
                    (ind4[j] - pad_width):(ind4[j] + pad_width + 1), :]

            patch = np.reshape(patch, (patchsize * patchsize, -1))
            patch = np.transpose(patch)
            patch = np.reshape(patch, (-1, patchsize, patchsize))
            data_patch_test[i][j, :, :, :] = patch


    return data_patch_train, data_patch_test, TR_map, TE_map

class MyDataset(Dataset):
    def __init__(self, data_patch, label_map):
        super(MyDataset, self).__init__()
        self.label_map = label_map[label_map != 0]
        self.data_patch = data_patch


    def __getitem__(self, idx):
        return ([torch.from_numpy(data[idx]) for data in self.data_patch],
                torch.tensor(self.label_map[idx] - 1).to(torch.long))

    def __len__(self):
        return len(self.label_map)

def build_loader():
    config.defrost()
    data_tuple, labels = get_data(dataset_name=config.DATA.DATASET_NAME)
    data_patch_train, data_patch_test, TR_map, TE_map = build_patch_data(
        Data=data_tuple, gt=labels, patchsize=config.DATA.PATCH_SIZE, num_samples_per_class=config.DATA.Num_Samples_Per_Class)
    train_dataset = MyDataset(data_patch=data_patch_train, label_map=TR_map)
    test_dataset = MyDataset(data_patch=data_patch_test, label_map=TE_map)

    num_samples_per_class = config.DATA.BATCH_SIZE // config.DATA.NUM_CLASSES ## batchsize = NUM_CLASSES  *  N
    sampler = BalancedBatchSampler(train_dataset, config.DATA.NUM_CLASSES, num_samples_per_class)


    data_loader_train = torch.utils.data.DataLoader( # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        dataset=train_dataset,
        batch_sampler=sampler,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size= 512,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
    )
    config.freeze()
    return data_loader_train, data_loader_test, labels




