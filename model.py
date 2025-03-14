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
from torch.optim import lr_scheduler


class PlateDetectorModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=3,                      # model output channels (number of classes in your dataset)
        )

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]

        h, w = image.shape[2:]

        mask = batch["mask"]

        logits_mask = self.forward(image)

        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    
    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])


        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


    

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
    EPOCHS = 100

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

    sam_model = SAM("sam_b.pt")

    # Display model information (optional)
    sam_model.info()

    #testing
    print(sample)
    results = sam_model(sample[2], bboxes=sample[1]["boxes"])
    show_image_with_labels(sample[0], sample[1]["boxes"])

    #add mask to the sample information
    for cursample in train_dataloader:
        cursample[1]["mask"] = sam_model(cursample[2], bboxes=cursample[1]["boxes"])

    for cursample in valid_dataloader:
        cursample[1]["mask"] = sam_model(cursample[2], bboxes=cursample[1]["boxes"])

    model = PlateDetectorModel()

    trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )



if __name__ == "__main__":
     main()
