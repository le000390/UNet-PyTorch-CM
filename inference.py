import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from cell_dataset import CellDataset
from unet import UNet

def pred_show_image_grid(data_path, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device), weights_only=True))
    image_dataset = CellDataset(data_path, test=True)
    images = []
    orig_masks = []
    pred_masks = []
    pred_masks_binary = []

    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)

        pred_mask = model(img)

        img = img.squeeze(0).cpu().detach()
        img = img.permute(1, 2, 0)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask_binary = (pred_mask > 0.5).float()

        orig_mask = orig_mask.cpu().detach()
        orig_mask = orig_mask.permute(1, 2, 0)

        images.append(img)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)
        pred_masks_binary.append(pred_mask_binary)

    images.extend(orig_masks)
    images.extend(pred_masks)
    images.extend(pred_masks_binary)
    fig = plt.figure()
    for i in range(1, 4*len(image_dataset)+1):
        fig.add_subplot(4, len(image_dataset), i)
        plt.imshow(images[i-1], cmap="gray")
        plt.xticks([])
        plt.yticks([])
    plt.show()


def single_image_inference(image_pth, model_pth, device):
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device), weights_only=True))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()])

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)
   
    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask_binary = (pred_mask > 0.5).float()

    fig = plt.figure()
    for i in range(1, 3): 
        fig.add_subplot(1, 2, i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(pred_mask, cmap="gray")
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == "__main__":
    # SINGLE_IMG_PATH = "C:/Users/Preston Le (School)/Documents/UMN/Fall 2024/BMEN 5910 - Special Topics in Biomedical Engineering - Biomedical Data Science/Final Project/NN_PyTorch/data/manual_test/p92.png"
    DATA_PATH = "C:/Users/Preston Le (School)/Documents/UMN/Fall 2024/BMEN 5910 - Special Topics in Biomedical Engineering - Biomedical Data Science/Final Project/NN_PyTorch/data"
    MODEL_PATH = "C:/Users/Preston Le (School)/Documents/UMN/Fall 2024/BMEN 5910 - Special Topics in Biomedical Engineering - Biomedical Data Science/Final Project/NN_PyTorch/models/unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    # single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)
