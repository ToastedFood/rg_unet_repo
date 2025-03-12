from dataset import CustomDataset, TRAINING_DATA_DIR, TESTING_DATA_DIR, VALIDATION_DATA_DIR
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import cv2
from ultralytics import SAM

class PlateDetectorModel(pl.LightningModule):
     def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=3,                      # model output channels (number of classes in your dataset)
        )

def show_image_with_labels(image, labels):
        fig, ax = plt.subplots(1)
        # Transpose the image tensor and convert to numpy array
        ax.imshow(np.array(image))

        # Assuming labels are bounding boxes [x_min, y_min, x_max, y_max]
        for label in labels:
            rect = patches.Rectangle((label[0], label[1]), label[2] - label[0], label[3] - label[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()


def main():
    train_dataset = CustomDataset(TRAINING_DATA_DIR)
    test_dataset = CustomDataset(TESTING_DATA_DIR)
    val_dataset = CustomDataset(VALIDATION_DATA_DIR)

    print(f'Training set size: {len(train_dataset)}')
    print(f'Test set size: {len(test_dataset)}')
    print(f'Validation set size: {len(val_dataset)}')

    n_cpu = os.cpu_count()
    train_dataloader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=n_cpu
    )
    valid_dataloader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=n_cpu
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=n_cpu
    )

    sample = train_dataset[0]

    """
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_predictor = SamPredictor(sam)

    image_bgr = cv2.imread(sample[2])
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)

    masks, scores, logits = mask_predictor.predict(
        box=sample[0],
        multimask_output=True
    )"
    """

    model = SAM("sam_b.pt")

    # Display model information (optional)
    model.info()

    print(sample)
    results = model(sample[2], bboxes=sample[1]["boxes"])
    show_image_with_labels(sample[0], sample[1]["boxes"])


if __name__ == "__main__":
     main()
