import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io

import glob 
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.is_available()


# loading ImageNet base 
from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

weights = VGG16_Weights.DEFAULT
vgg_model = vgg16(weights=weights)

# freeze the model
vgg_model.requires_grad_(False)
next(iter(vgg_model.parameters())).requires_grad

# general layers of the model
vgg_model.classifier[0:3]

# changing the last layer
classes = 2

grub_model = nn.Sequential(
    vgg_model.features,
    vgg_model.avgpool,
    nn.Flatten(),
    vgg_model.classifier[0:3],
    nn.Linear(25088, 500),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(500, classes),
    nn.Softmax(dim=1)

)
print(grub_model)

# compile model with loss and metric options
loss_function = nn.BCELoss()
optimizer = Adam(grub_model.parameters())

grub_model = torch.compile(grub_model.to(device), mode='reduce-overhead')

#import data augmentation
import augmentation as aug

# loading and training the dataset
data_labels = [('Free food detected', 0), ('No free food detected', 1)]

class GrubDataset(Dataset):
    def __init__(self, data_dir):
        self.imgs = []
        self.labels = []

        for l_index, label in enumerate(data_labels):
            data_paths = glob.glob(data_dir + '/' + label[0] + '/*.png', recursive=True)
            for path in data_paths:
                augmented_img, _ = aug.apply_augmentation(path)
                self.imgs.append(augmented_img)
                self.labels.append(label[1])
    
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.imgs)
    
# Initializing batch size
n = 32

# Train dataset and initialize dataloader
train_path = ''
train_dataset = GrubDataset(train_path)
train_loader = DataLoader(train_dataset, batch_size=n, shuffle=True)
train_N = len(train_loader.dataset)

# Validation dataset and initialize dataloader
valid_path = ''
valid_dataset = GrubDataset(valid_path)
valid_loader = DataLoader(valid_dataset, batch_size=n, shuffle=False)
valid_N = len(valid_loader.dataset)








