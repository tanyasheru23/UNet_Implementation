import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim
from u_net_model import UNet
from torch.utils.data import DataLoader
from dataset import CarvanaDataset

from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

LEARNING_RATE = 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 12
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val/"
VAL_MASK_DIR = "data/val_masks"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device = DEVICE)
        
        # forward
        model.train()
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss = loss.item())        

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )
    
    val_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )
    
    model = UNet(in_channels=3, out_channels=1).to(device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # with logits because in our model we didn't perform sigmoid after finalConv
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(train_dir=TRAIN_IMG_DIR,
                                            train_maskdir=TRAIN_MASK_DIR,
                                            val_dir=VAL_IMG_DIR,
                                            val_maskdir=VAL_MASK_DIR,
                                            train_transform=train_transform,
                                            val_transform=val_transform,
                                            batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS,
                                            pin_memory=PIN_MEMORY)
    
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(EPOCHS):
        print(f"------Epoch: {epoch}------")
        
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # save the model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)
        
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
        
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader,
            model,
            folder_dir="saved_images/",
            device=DEVICE
        )

def test_fn():
    pass

if __name__ == "__main__":
    main()