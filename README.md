# CFRP Defect Segmentation Using U-Net

This repository contains the deep-learning pipeline developed for semantic
segmentation and classification of manufacturing defects in carbon fiber
reinforced polymer (CFRP) microscopy images.

The project investigates whether a U-Net-based convolutional neural network
can identify and classify microscopic CFRP defects using a relatively small
manually annotated dataset.

## Dataset

The dataset contains 94 optical microscopy images of CFRP specimens.

Images were manually annotated using CVAT and divided into:

- 70 training images
- 12 validation images
- 12 test images

The multiclass segmentation problem contains four classes:

0. Background
1. Void
2. Resin-rich region
3. Crack

The dataset used for this project is not included in this repository.

## Model Architecture

The model is implemented using `segmentation_models_pytorch`.

Architecture:

- U-Net segmentation architecture
- SE-ResNet50 encoder (`se_resnet50`)
- ImageNet-pretrained encoder weights
- 3-channel image input
- 4-class pixel-wise output

Model initialization:

    model = seg.Unet(
        encoder_name="se_resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4
    )

The pretrained encoder provides transfer learning from ImageNet while the
U-Net decoder reconstructs spatial information for pixel-level defect
segmentation.

## Training

Training uses the Adam optimizer with an initial learning rate of:

    3e-4

A scheduler learning-rate monitors validation mean IoU:

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.7,
        patience=13,
        threshold=0.001,
        min_lr=1e-6
    )

The learning rate is therefore reduced when validation IoU stops improving.

Training is performed using CUDA acceleration when an NVIDIA GPU is
available.

## Loss Function

The training objective combines weighted cross-entropy with Tversky loss.

Class weights:

    [1.0, 4.0, 4.0, 15.0]

corresponding to:

    [background, void, resin-rich, crack]

The increased defect-class weights compensate for substantial pixel-level
class imbalance, particularly for the thin and relatively uncommon crack
class.

The final training loss is:

    Loss = Weighted Cross Entropy + Tversky Loss

## Data Augmentation

Training images undergo randomized augmentation to increase variation in the
limited dataset.

Augmentation includes transformations such as:

- Horizontal/vertical orientation changes
- Brightness variation
- Contrast variation

Augmentation is applied only to the training data.

## Validation and Model Selection

Model performance is evaluated during training using Intersection over Union
(IoU).

Validation IoU is tracked for:

- Mean segmentation performance
- Void
- Resin-rich regions
- Cracks

Separate checkpoints are retained for the highest observed:

    bestaverage_iou_model
    bestvoid_iou_model
    bestresin_iou_model
    bestcrack_iou_model

## Specialist Ensemble and Inference

Final multiclass inference can combine the average-IoU model with
class-specialist checkpoints.

The specialist models provide additional evidence for their corresponding
defect classes. Their predictions are combined with the general model before
the final pixel-wise class decision.

The final segmentation is obtained using the class with the highest combined
prediction at each pixel.

Test-time augmentation (TTA) is also used during inference to improve
prediction robustness.

## Evaluation

Primary evaluation metrics include:

- Intersection over Union (IoU)
- Dice coefficient
- Per-class IoU
- Confusion matrices

Cracks are especially challenging because they occupy relatively few pixels
and have thin geometries.
