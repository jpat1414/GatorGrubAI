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

# change the last layer
classes = 2

model = nn.Sequential(
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
print(model)


