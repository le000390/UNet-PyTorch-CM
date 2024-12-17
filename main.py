import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from cell_dataset import CellDataset

def jaccard_loss(y_pred, y_true, smooth=1e-6):
    """
    Calculate the Jaccard loss (IoU-based loss) between predicted and true masks.
    This is a differentiable approximation of IoU.
    
    Args:
        y_pred: The predicted masks from the network (after applying sigmoid activation).
        y_true: The ground truth binary masks.
        smooth: A small constant to prevent division by zero.

    Returns:
        The computed Jaccard loss value.
    """

    # Flatten the tensors to treat them as vectors
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection

    # Compute Jaccard loss
    jaccard = (intersection + smooth) / (union + smooth)
    penalty = 1 - jaccard
    return penalty  # 1 - Jaccard to make it a loss (i.e., higher IoU -> lower loss)

def fringe_penalty_loss(y_pred, img_size, sigma=50):
    """
    Penalize masks (1) further from the center of the image & emptiness (0) closer towards the center using a Gaussian distribution.
    
    Args:
        y_pred: The predicted masks from the network (after applying sigmoid activation).
        img_size: A tuple (H, W) indicating the height and width of the image.
        sigma: Controls the width of the Gaussian penalty (smaller sigma -> stricter penalty).

    Returns:
        The computed fringe penalty loss.
    """
    # Compute the distance matrix from the center of the image
    H, W = img_size
    center_y, center_x = H // 2, W // 2

    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=y_pred.device), torch.arange(W, device=y_pred.device), indexing='ij'
    )
    distances = ((y_coords - center_y)**2 + (x_coords - center_x)**2).float()
    
    # Compute Gaussian weights
    gaussian_weights = torch.exp(-distances / (2 * sigma**2))

    # Compute the penalty as a weighted sum of the predictions
    mask = torch.sum((1 - gaussian_weights) * y_pred)
    penalty_mask = mask / (H * W)
    empt = torch.sum((gaussian_weights) * (1-y_pred))
    penalty_empt = empt / (H * W)

    penalty = 0.5 * penalty_mask + 0.5 * penalty_empt
    return penalty

def CM_loss(y_pred, y_true, img_size, alpha=0.5, sigma=50, smooth=1e-6):
    """
    Combined Jaccard loss and fringe penalty loss.

    Args:
        y_pred: The predicted masks from the network (logits).
        y_true: The ground truth binary masks.
        img_size: A tuple (H, W) indicating the height and width of the image.
        alpha: Weight for combining the Jaccard loss and the fringe penalty loss.
        sigma: Controls the width of the Gaussian penalty (smaller sigma -> stricter penalty).
        smooth: A small constant to prevent division by zero for Jaccard loss.

    Returns:
        The combined loss value.
    """
    jaccard_penalty = jaccard_loss(y_pred, y_true, smooth)
    fringe_penalty = fringe_penalty_loss(y_pred, img_size, sigma)

    # Combine the two losses
    combined_loss = alpha * jaccard_penalty + (1 - alpha) * fringe_penalty
    return combined_loss

if __name__ == "__main__":
    LEARNING_RATE = 0.001
    BATCH_SIZE = 1
    EPOCHS = 10
    DATA_PATH = "C:/Users/Preston Le (School)/Documents/UMN/Fall 2024/BMEN 5910 - Special Topics in Biomedical Engineering - Biomedical Data Science/Final Project/NN_PyTorch/data"
    MODEL_SAVE_PATH = "C:/Users/Preston Le (School)/Documents/UMN/Fall 2024/BMEN 5910 - Special Topics in Biomedical Engineering - Biomedical Data Science/Final Project/NN_PyTorch/models/unet.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = CellDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = CM_loss

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = model(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask, [256, 256], 0.5, 1e-6)
            train_running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask, [256, 256], 0.5, 1e-6)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
