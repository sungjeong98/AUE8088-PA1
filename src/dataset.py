# Python packages
from termcolor import colored
from tqdm import tqdm
import os
import tarfile
import wget

# PyTorch & Pytorch Lightning
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Custom packages


class TinyImageNetDatasetModule(LightningDataModule):
    __DATASET_NAME__ = 'tiny-imagenet-200'

    def __init__(self, batch_size, cfg):
        super().__init__()
        self.batch_size = batch_size
        self.cfg = cfg

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        if not os.path.exists(os.path.join(self.cfg.dataset_root_path, self.__DATASET_NAME__)):
            # download data
            print(colored("\nDownloading dataset...", color='green', attrs=('bold',)))
            filename = self.__DATASET_NAME__ + '.tar'
            wget.download(f'https://hyu-aue8088.s3.ap-northeast-2.amazonaws.com/{filename}')

            # extract data
            print(colored("\nExtract dataset...", color='green', attrs=('bold',)))
            with tarfile.open(name=filename) as tar:
                # Go over each member
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    # Extract member
                    tar.extract(path=self.cfg.dataset_root_path, member=member)
            os.remove(filename)

    def train_dataloader(self):
        tf_train = transforms.Compose([
            transforms.RandomRotation(self.cfg.image_rotation),
            transforms.RandomHorizontalFlip(self.cfg.image_flip_prob),
            transforms.RandomCrop(self.cfg.image_num_crops, padding=self.cfg.image_pad_crops),
            transforms.ToTensor(),
            transforms.Normalize(self.cfg.image_mean, self.cfg.image_std),
        ])
        dataset = ImageFolder(os.path.join(self.cfg.dataset_root_path, self.__DATASET_NAME__, 'train'), tf_train)
        msg = f"[Train]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            num_workers=self.cfg.num_workers,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        tf_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.cfg.image_mean, self.cfg.image_std),
        ])
        dataset = ImageFolder(os.path.join(self.cfg.dataset_root_path, self.__DATASET_NAME__, 'val'), tf_val)
        msg = f"[Val]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            pin_memory=True,
            num_workers=self.cfg.num_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        tf_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.cfg.image_mean, self.cfg.image_std),
        ])
        dataset = ImageFolder(os.path.join(self.cfg.dataset_root_path, self.__DATASET_NAME__, 'test'), tf_test)
        msg = f"[Test]\t root dir: {dataset.root}\t | # of samples: {len(dataset):,}"
        print(colored(msg, color='blue', attrs=('bold',)))

        return DataLoader(
            dataset,
            num_workers=self.cfg.num_workers,
            batch_size=self.batch_size,
        )
