#imports
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as seg
from pathlib import Path
import random
from torch.utils.data import DataLoader, Dataset
from Calculations import get_average_iou

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

#setting images
images = Path("images")
annotations = Path("annotations")

#calc images
images2 = Path("Validation Images")
annotations2 = Path("Annotation Validation")

pair2 = []

for image2 in images2.iterdir():
    mask2 = annotations2 / (image2.stem+ ".png")

    if mask2.exists():
        img2 = cv2.imread(str(image2))
        annmask2 = cv2.imread(str(mask2))

        #img2 = cv2.resize(
        #    img2,
        #    (1280,1024),
        #    interpolation=cv2.INTER_CUBIC
        #)
        #annmask2 = cv2.resize(
        #    annmask2,
        #    (1280,1024),
        #    interpolation=cv2.INTER_NEAREST
        #)

        ann2 = np.zeros((512, 640), dtype = np.int64)
        ann2[np.all(annmask2 == [0,0,255], axis = 2)] = 1 #void
        ann2[np.all(annmask2 == [255,0,0], axis = 2)] = 2 #resin-rich
        ann2[np.all(annmask2 == [0,255,0], axis = 2)] = 3 #crack

        pair2.append((img2, ann2))


def augmentation(img, ann):
    if random.random() < 0.5: #50% horiz
        img = cv2.flip(img, 1)
        ann = cv2.flip(ann, 1)

    if random.random() < 0.5: #50% vert
        img = cv2.flip(img, 0)
        ann = cv2.flip(ann, 0)

    if random.random() < 0.5: #brightness
        multfactor = random.uniform(0.8, 1.2)
        img = np.clip(img*multfactor, 0, 255).astype(np.uint8)

    if random.random() < 0.5: #contrast
        factor = random.uniform(0.8, 1.2)
        mean = np.mean(img)
        img = np.clip((img - mean)*factor + mean, 0, 255).astype(np.uint8)

    #if random.random() < 0.5: #tint
    #    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + random.randint(-15, 15)) % 180
    #    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    #if random.random() < 0.1:
        #img = cv2.GaussianBlur(img, (3,3), 0)

    #if random.random() < 0.1:
        #noise = np.random.normal(0, 4, img.shape)
        #img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img, ann


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

        pair.append((img, ann))#im using more memory, images stay loaded but since its 80ish doesn't matter much

#channel_sum = 0.0
#pixelcount = 0

#for img, ann in pair:
#    img = img/255.0
#    channel_sum += img.sum(axis=(0,1))
#    pixelcount += img.shape[0]*img.shape[1]
#mean = channel_sum/pixelcount


#std_sum = 0.0

#for img, ann in pair:
#    img = img/255.0
#    std_sum = ((img - mean)**2).sum(axis=(0,1))
#std = np.sqrt(std_sum / pixelcount)

bestaverageiou = -1
bestvoidiou = -1
bestresiniou = -1
bestcrackiou = -1
cycle = 0

class CarbonDataset(Dataset):
    def __init__(self, pair):
        self.pair = pair

    def __len__(self):
        return(len(self.pair))
    
    def __getitem__(self, idx):

        img, ann = self.pair[idx]
        img = img.copy()
        ann = ann.copy()

        img, ann = augmentation(img, ann) #augment image

        img = img/255.0
        #img = (img-mean)/std

        img = torch.tensor(img, dtype=torch.float32).permute(2,0,1) #change format after converting numpy to tensor
        ann = torch.tensor(ann, dtype = torch.long) #new format


        return img, ann

while cycle<20: #how many training cycles to do

    train_dataset = CarbonDataset(pair)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True
    )

    cycle += 1
    print(f"creating model {cycle}")

    model = seg.Unet( #this is se_res50 imagenet no plus plus
        encoder_name="se_resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4,#change later for more outputs
    )

    model = model.to(device)

    print(f"created model {cycle}")

    weights = torch.tensor([1.0, 4.0, 4.0, 15.0]).to(device)
    CROSSloss = torch.nn.CrossEntropyLoss(weight=weights)#pytorch, calculates error between pred and truth
    FOCALloss = seg.losses.TverskyLoss(mode = "multiclass", alpha = 0.3, beta = 0.7, gamma = 0.75, from_logits = True)
    func_opt = torch.optim.Adam(model.parameters(), lr = 0.0003)#pytorch, ADAMS model, calc weight change
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        func_opt,
        mode="max",
        factor = 0.7,
        patience = 13,
        threshold=0.001,
        min_lr=1e-6
    )

    run = 0

    model.train()
    for i in range(170):#epoch number
        run += 1

        #if i == 85:
        #    for param in func_opt.param_groups:
        #        param["lr"] = 0.0001
        #        print("learning step 1")


        for img,ann in train_loader:
            img = img.to(device)
            ann = ann.to(device)
            pred = model(img) #take my image and make a prediction on it
            lossCROSS = CROSSloss(pred,ann)
            lossFOCAL = FOCALloss(pred, ann)
            loss = lossCROSS + lossFOCAL

            func_opt.zero_grad()#resets image gradient
            loss.backward()#figures out what caused the error
            func_opt.step()#alters weight based on error  
        print(f"It is epoch {run}")

        runaverageiou, runvoidiou, runresiniou, runcrackiou = get_average_iou(model, pair2)

        scheduler.step(runaverageiou)


        if runaverageiou>bestaverageiou:
            torch.save(model.state_dict(), "bestaverage_iou_model")
            bestaverageiou = runaverageiou
            print(f"Best Average: {run}, IoU: {runaverageiou:.3f}")

        if runvoidiou>bestvoidiou:
            torch.save(model.state_dict(), "bestvoid_iou_model")
            bestvoidiou = runvoidiou
            print(f"Best Void: {run}, IoU: {runvoidiou:.3f}")

        if runresiniou>bestresiniou:
            torch.save(model.state_dict(), "bestresin_iou_model")
            bestresiniou = runresiniou
            print(f"Best Resin: {run}, IoU: {runresiniou:.3f}")

        if runcrackiou>bestcrackiou:
            torch.save(model.state_dict(), "bestcrack_iou_model")
            bestcrackiou = runcrackiou
            print(f"Best Crack: {run}, IoU: {runcrackiou:.3f}")

    print(f"Cycle {cycle} is done")

print("Done With Cycles")#checking to make sure if went through smoothly, temp