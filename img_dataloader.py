# Pytroch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

# Other libs
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ImageDataset(Dataset):

    def __init__(self, transform=transforms.ToTensor(), color='L'):

        self.transform = transform
        self.color = color
        self.image_dir = "/home/zeppelin/PycharmProjects/math179final/101_ObjectCategories/"
        self.image_info = pd.read_csv(
            "/home/zeppelin/PycharmProjects/math179final/image_labels.csv")
        self.image_filenames = self.image_info["name"]
        self.labels = self.image_info["labels"]
        self.classes = self.labels.tolist()

    def __len__(self):

        return len(self.image_filenames)

    def __getitem__(self, ind):

        # Compose the path from the absolute directory, the subdirectory (class name), and the image name
        image_path = os.path.join(self.image_dir, os.path.join(self.labels.ix[ind], self.image_filenames.ix[ind]))

        # Load the image
        image = Image.open(image_path).convert(mode=str(self.color))

        # Resize the image
        image = image.resize((300, 300))

        # If a transform is specified, apply it
        if self.transform is not None:
            image = self.transform(image)

        # Verify that image is in  Tensor format
        if type(image) is not torch.Tensor:
            image = transforms.ToTensor()(image)

        # Convert multi-class label into binary encoding
        label = self.convert_label(self.labels[ind], self.classes)

        # Retur the image and its label
        return (image, label)

    def convert_label(self, label, classes):

        return classes.index(label)


def create_train_test_loader(batch_size, seed, transform=transforms.ToTensor(), p_test=0.2,
                             shuffle=True, show_sample=False, extras={}):
    # Get a ImageDataset object
    dataset = ImageDataset(transform)

    # Dimensions and indices of trainnig set
    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)

    # Split the test dataset from the train dataset
    test_split = int(np.floor(p_test * len(all_indices)))
    train_ind, test_ind = all_indices[test_split:], all_indices[:test_split]

    # Make a sampler for the test dataset
    sample_test = SubsetRandomSampler(test_ind)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    # Define the test dataloader
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_test,
                             num_workers=num_workers, pin_memory=pin_memory)

    return (test_loader, train_ind)


def create_train_dataloader(train_ind, batch_size, transform=transforms.ToTensor,
                            show_sample=False, extras={}):
    dataset = ImageDataset(transform)

    sample_train = SubsetRandomSampler(train_ind)

    num_workers = 0
    pin_memory = False
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_train,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader


# Function takes in the indices of the training data and the number of folds
# and generates a list of splits (val_indices, train_indices) for each possible split
# of the training data
def get_validation_indices(train_indices, k):
    # Get the validation subset size
    train_set_size = len(train_indices)
    val_subset_size = int(np.floor(train_set_size / k))

    # Get a list of (val_indices, train_ind) for each validation subset
    val_indices = []
    for i in range(0, k):
        subset_start = int(i * val_subset_size)
        subset_end = int((i + 1) * val_subset_size)

        val_subset_ind = train_indices[subset_start:subset_end]
        train_subset_ind = train_indices[:subset_start] + train_indices[subset_end:]

        val_indices.append((val_subset_ind, train_subset_ind))

    return val_indices


def create_k_split_dataloaders(val_indices, batch_size, kth_index, transform=transforms.ToTensor(),
                               show_sample=False, extras={}):
    # Get an ImageDataset object
    dataset = ImageDataset(transform)

    # Get the validatoin set indices and the train set indices
    val_train_set = val_indices[kth_index]
    val_ind = val_train_set[0]
    train_ind = val_train_set[1]

    # Use the subset random sampler as the sampler for each dataset
    sample_val = SubsetRandomSampler(val_ind)
    sample_train = SubsetRandomSampler(train_ind)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    # Define the dataloaders for the validation and train sets
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_val,
                            num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sample_train,
                              num_workers=num_workers, pin_memory=pin_memory)

    return (val_loader, train_loader)
