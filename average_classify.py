import segmentation_models_pytorch as seg
import torch
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

images = Path("Final Test Images")
annotations = Path("Final Annotation")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pair = []

for image in images.iterdir():
    mask = annotations / (image.stem+ ".png") #AI suggested this as the way to parse through files

    if mask.exists():
        img = cv2.imread(str(image))
        annmask = cv2.imread(str(mask))
        #img = cv2.resize(
        #img,
        #(1280,1024),
        #interpolation=cv2.INTER_CUBIC
        #)
        #annmask = cv2.resize(
        #annmask,
        #(1280,1024),
        #interpolation=cv2.INTER_NEAREST
        #)

        ann = np.zeros((512, 640), dtype = np.int64)
        ann[np.all(annmask == [0,0,255], axis = 2)] = 1 #void
        ann[np.all(annmask == [255,0,0], axis = 2)] = 2 #resin-rich
        ann[np.all(annmask == [0,255,0], axis = 2)] = 3 #crack


        #img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)/255.0 #change format after converting numpy to tensor
        #ann = torch.tensor(ann, dtype = torch.long).unsqueeze(0) #putting channel infront, new format
        pair.append((img, ann))

channel_sum = 0.0
pixelcount = 0


for img, ann in pair:
    img = img/255.0
    channel_sum += img.sum(axis=(0,1))
    pixelcount += img.shape[0]*img.shape[1]
mean = channel_sum/pixelcount


std_sum = 0.0

for img, ann in pair:
    img = img/255.0
    std_sum = ((img - mean)**2).sum(axis=(0,1))
std = np.sqrt(std_sum / pixelcount)



modelavg = seg.Unet(
    encoder_name = "se_resnet50",
    encoder_weights = "imagenet",
    classes = 4,
    in_channels = 3,
)
modelvoid = seg.Unet(
    encoder_name = "se_resnet50",
    encoder_weights = "imagenet",
    classes = 4,
    in_channels = 3,
)
modelresin = seg.Unet(
    encoder_name = "se_resnet50",
    encoder_weights = "imagenet",
    classes = 4,
    in_channels = 3,
)
modelcrack = seg.Unet(
    encoder_name = "se_resnet50",
    encoder_weights = "imagenet",
    classes = 4,
    in_channels = 3,
)
modelavg = modelavg.to(device)
modelvoid = modelvoid.to(device)
modelresin = modelresin.to(device)
modelcrack = modelcrack.to(device)

modelavg.load_state_dict(torch.load("bestaverage_iou_model"))
modelvoid.load_state_dict(torch.load("bestvoid_iou_model"))
modelresin.load_state_dict(torch.load("bestresin_iou_model"))
modelcrack.load_state_dict(torch.load("bestcrack_iou_model"))

modelavg.eval()
modelvoid.eval()
modelresin.eval()
modelcrack.eval()

truth_confusion = []
prediction_confusion = []
run = 0
averageiou = 0
totalintersections = np.zeros(4, dtype = float)
totalunion = np.zeros(4, dtype = float)

for img, ann in pair:
    ious = []
    run += 1

    img = img/255.0
    #img = (img-mean)/std

    img = torch.tensor(img, dtype=torch.float32).permute(2,0,1).to(device) #change format after converting numpy to tensor
    ann = torch.tensor(ann, dtype = torch.long).to(device) #new format

    img_flipped_horizontal = torch.flip(img, dims=[2])
    img_flipped_vertical = torch.flip(img, dims=[1])
    img_180 = torch.rot90(img, k = 2, dims=[1, 2])

    with torch.no_grad():
        predavg_noflip = modelavg(img.unsqueeze(0))
        predavg_flipped_horizontal = modelavg(img_flipped_horizontal.unsqueeze(0))
        predavg_flipped_horizontal = torch.flip(predavg_flipped_horizontal, dims=[3])
        predavg_flipped_vertical = modelavg(img_flipped_vertical.unsqueeze(0))
        predavg_flipped_vertical = torch.flip(predavg_flipped_vertical, dims=[2])
        predavg_180 = modelavg(img_180.unsqueeze(0))
        predavg_180 = torch.rot90(predavg_180, k=2, dims=[2,3])
        predavg = (predavg_flipped_horizontal + predavg_noflip + predavg_flipped_vertical + predavg_180)/4

        predvoid_noflip = modelvoid(img.unsqueeze(0))
        predvoid_flipped_horizontal = modelvoid(img_flipped_horizontal.unsqueeze(0))
        predvoid_flipped_horizontal = torch.flip(predvoid_flipped_horizontal, dims=[3])
        predvoid_flipped_vertical = modelvoid(img_flipped_vertical.unsqueeze(0))
        predvoid_flipped_vertical = torch.flip(predvoid_flipped_vertical, dims=[2])
        predvoid_180 = modelvoid(img_180.unsqueeze(0))
        predvoid_180 = torch.rot90(predvoid_180, k=2, dims=[2,3])
        predvoid = (predvoid_flipped_horizontal + predvoid_noflip + predvoid_flipped_vertical + predvoid_180)/4

        predresin_noflip = modelresin(img.unsqueeze(0))
        predresin_flipped_horizontal = modelresin(img_flipped_horizontal.unsqueeze(0))
        predresin_flipped_horizontal = torch.flip(predresin_flipped_horizontal, dims=[3])
        predresin_flipped_vertical = modelresin(img_flipped_vertical.unsqueeze(0))
        predresin_flipped_vertical = torch.flip(predresin_flipped_vertical, dims=[2])
        predresin_180 = modelresin(img_180.unsqueeze(0))
        predresin_180 = torch.rot90(predresin_180, k=2, dims=[2,3])
        predresin = (predresin_flipped_horizontal + predresin_noflip + predresin_flipped_vertical + predresin_180)/4

        predcrack_noflip = modelcrack(img.unsqueeze(0))
        predcrack_flipped_horizontal = modelcrack(img_flipped_horizontal.unsqueeze(0))
        predcrack_flipped_horizontal = torch.flip(predcrack_flipped_horizontal, dims=[3])
        predcrack_flipped_vertical = modelcrack(img_flipped_vertical.unsqueeze(0))
        predcrack_flipped_vertical = torch.flip(predcrack_flipped_vertical, dims=[2])
        predcrack_180 = modelcrack(img_180.unsqueeze(0))
        predcrack_180 = torch.rot90(predcrack_180, k=2, dims=[2,3])
        predcrack = (predcrack_flipped_horizontal + predcrack_noflip + predcrack_flipped_vertical + predcrack_180)/4

    combined = predavg.clone()
    combined[:, 1] += 1.45 * torch.relu(predvoid[:, 1] - predavg[:, 1])#1.45
    combined[:, 2] += 1.25 * torch.relu(predresin[:, 2] - predavg[:, 2])#1.25
    combined[:, 3] += 1.65 * torch.relu(predcrack[:, 3] - predavg[:, 3])#1.65

    pred = torch.argmax(combined, dim=1) #chooses largest prob
    pred = pred.squeeze(0)

    truth_confusion.extend(ann.flatten().tolist())
    prediction_confusion.extend(pred.flatten().tolist())

    for class_id in range(4): #iou calc

        pred_mask = pred == class_id
        ann_mask = ann == class_id

        intersection = torch.logical_and(pred_mask, ann_mask).sum()
        union = torch.logical_or(pred_mask, ann_mask).sum()

        if union.item() == 0:
            continue

        iou = intersection/union
        ious.append(iou)
        totalintersections[class_id] += intersection
        totalunion[class_id] += union
        print(f"Class {class_id}: {iou:.3f} - image{run}")
    averageiou += torch.stack(ious).mean().item()
    print(f"Mean IoU for image: {torch.mean(torch.stack(ious)):.3f} - image{run}")

dataiou = totalintersections/totalunion
datadice = (2 * dataiou) / (1 + dataiou)

#print(f"Mean per image IoU across all images: {averageiou/len(pair):.3f}")
print(f"Dataset IoU across all images: {(np.mean(dataiou)):.3f}")
for i in range(4):
    print(f"Dataset per class IoU: class {i} - {dataiou[i]:.3f}")
print(f"Dataset DICE across all images: {(np.mean(datadice)):.3f}")
for i in range(4):
    print(f"Dataset per class DICE: class {i} - {datadice[i]:.3f}")


confusionmatrix = confusion_matrix(
    truth_confusion,
    prediction_confusion,
    labels = [0, 1, 2, 3],
    normalize="true"
)


display = ConfusionMatrixDisplay(
    confusion_matrix = confusionmatrix,
    display_labels = ["Background", "Void", "Resin", "Crack"]
)

display.plot()
plt.show()