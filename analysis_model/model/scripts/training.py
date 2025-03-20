import main_model
from main_model import GrubDataset
from main_model import DataLoader

import glob

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

