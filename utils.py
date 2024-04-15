import torch
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
import torchvision

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir, 
    val_maskdir,
    train_transform,
    val_transform,
    batch_size,
    num_workers,
    pin_memory = True
):
    train_data = CarvanaDataset(image_dir=train_dir,
                                mask_dir=train_maskdir, 
                                transform=train_transform)
    val_data = CarvanaDataset(image_dir=val_dir, 
                                mask_dir=val_maskdir, 
                                transform=val_transform)
    
    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory)
    
    return train_dataloader, val_dataloader

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("==> Saving CheckPoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("==> Loading CheckPoint")
    model.load_state_dict(checkpoint["state_dict"])
    
def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    
    with torch.inference_mode():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(X))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum())/((preds + y).sum()+ 1e-8)
    print(
        f"Got {num_correct}/{num_pixels} with accuracy {(num_correct/num_pixels)*100: .3f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()
    
def save_predictions_as_imgs(loader, model,device, folder_dir = "saved_images/"):
    model.eval()
    
    for idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        with torch.inference_mode():
            preds = torch.sigmoid(model(X))
            preds = (preds > 0.5).float()
            
        torchvision.utils.save_image(preds, f"{folder_dir}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder_dir}{idx}.png")
    
    model.train()        