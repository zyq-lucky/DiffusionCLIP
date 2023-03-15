from torch.utils.data import Dataset
import lmdb
from io import BytesIO
from PIL import Image
import torchvision.transforms as tfs
import os
from glob import glob
import cv2
import numpy as np
import torch


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError("Cannot open lmdb dataset", path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
    
class CelebA(Dataset):
    def __init__(self, image_root, transform=None, mode='train', animal_class='dog', img_size=256):
        super().__init__()
        # fd = open(os.path.join(image_root, 'val.txt'))
        # lines = fd.readlines()
        # fd.close()
        
        self.image_paths = glob(os.path.join(image_root, '*.png'))
        self.transform = transform
        self.img_size = img_size

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = image_path.replace('single','mask').replace('stroke','mask')
        x = Image.open(image_path)
        x = x.resize((self.img_size, self.img_size),Image.NEAREST)
        label = cv2.imread(label_path,0)
        label =  cv2.resize(label,(self.img_size, self.img_size),cv2.INTER_NEAREST)
        label[label < 200] = 0
        kernel = np.ones((3, 3), dtype=np.uint8)
        label = cv2.erode(label, kernel, iterations=1)
        # label = Image.open(label_path).convert("L")
        # label = label.resize((self.img_size, self.img_size),)
        if self.transform is not None:
            x = self.transform(x)
            # label = tfs.ToTensor()(label)
            label = torch.from_numpy(label).float().unsqueeze(0) / 255.0
        return {'img':x,'name':image_path.split('/')[-1],'label':label}

    def __len__(self):
        return len(self.image_paths)


################################################################################

def get_celeba_dataset(data_root, config):
    train_transform = tfs.Compose([tfs.ToTensor(),
                                   tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                 inplace=True)])

    test_transform = tfs.Compose([tfs.ToTensor(),
                                  tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                                inplace=True)])

    # train_dataset = MultiResolutionDataset(os.path.join(data_root, 'LMDB_train'),
    #                                        train_transform, config.data.image_size)
    # test_dataset = MultiResolutionDataset(os.path.join(data_root, 'LMDB_test'),
    #                                       test_transform, config.data.image_size)

    train_dataset = CelebA(os.path.join(data_root),
                                           train_transform, config.data.image_size)
    test_dataset = CelebA(os.path.join(data_root),
                                          test_transform, config.data.image_size)


    return train_dataset, test_dataset



