from pathlib import Path
import cv2
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images2 = Path("Validation Images")
annotations2 = Path("Annotation Validation")


pair2 = []

for image2 in images2.iterdir():
    mask2 = annotations2 / (image2.stem+ ".png") #AI suggested this as the way to parse through files

    if mask2.exists():
        img2 = cv2.imread(str(image2))
        annmask2 = cv2.imread(str(mask2))

        ann2 = np.zeros((512, 640), dtype = np.int64)
        ann2[np.all(annmask2 == [0,0,255], axis = 2)] = 1 #void
        ann2[np.all(annmask2 == [255,0,0], axis = 2)] = 2 #resin-rich
        ann2[np.all(annmask2 == [0,255,0], axis = 2)] = 3 #crack

        pair2.append((img2, ann2))

def get_average_iou(model, pair):
    model.eval()
    averageiou = 0
    voidiou = 0
    resiniou = 0
    crackiou = 0
    voidcount = 0
    resincount = 0
    crackcount = 0

    for img, ann in pair:
        ious = []
        img = img/255.0

        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1).to(device) #change format after converting numpy to tensor
        ann = torch.tensor(ann, dtype = torch.long).to(device) #new format

        with torch.no_grad():
            pred = model(img.unsqueeze(0))

        pred = torch.argmax(pred, dim=1) #chooses largest prob
        pred = pred.squeeze(0)

        for class_id in range(4): #iou calc

            pred_mask = pred == class_id
            ann_mask = ann == class_id

            intersection = torch.logical_and(pred_mask, ann_mask).sum()
            union = torch.logical_or(pred_mask, ann_mask).sum()

            if union.item() == 0:
                continue

            iou = intersection/union
            ious.append(iou)

            if class_id == 1:
                voidiou += iou
                voidcount += 1

            if class_id == 2:
                resiniou += iou
                resincount += 1

            if class_id == 3:
                crackiou += iou
                crackcount += 1
        averageiou += torch.stack(ious).mean().item()#add iou to average

    model.train() 
    return averageiou/len(pair), voidiou/voidcount if voidcount else 0, resiniou/resincount if resincount else 0, crackiou/crackcount if crackcount else 0


def get_average_iou_single(model, pair):
    sumiou = 0

    model.eval()
    for i in range(len(pair)):
        img, ann = pair[i]

        with torch.no_grad():
            pred = model(img.unsqueeze(0))

        pred = torch.sigmoid(pred) #makes each pixel probability
        pred = pred.squeeze()
        pred_mask = pred > 0.4 #getting rid of some of the fuzz

        #iou
        intersection = (pred_mask * ann).sum()
        union = pred_mask.sum() + ann.sum() - intersection

        iou = intersection/union
        sumiou += iou
    
    sumiou = sumiou/len(pair)
    model.train()
    return sumiou