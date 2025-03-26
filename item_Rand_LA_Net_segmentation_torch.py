
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchmetrics.classification import MulticlassMatthewsCorrCoef
from torch.utils.data import DataLoader

import numpy as np
from matplotlib import pyplot as plt
import time

import numpy as np
import time

from SuperMarketDataset import SegmentationDataset

from item_Rand_LA_Net_torch import RandLANet
from item_Rand_LA_Net_torch import RandLANet_SegLoss
from item_pointnet_torch import compute_iou


print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # 0 corresponds to the first GPU

import getpass



# https://github.com/aRI0U/RandLA-Net-pytorch/blob/master/utils/tools.py














NUM_POINTS = 2048
NUM_CLASSES = 2
#BATCH_SIZE = 16
BATCH_SIZE = 2
NUM_EPOCHS = 10


SEGMENTAION_DATASET_PATH = "/home/"+getpass.getuser()+"/MiniMarket_dataset_segmentation/object_segmentation_dataset/"
SEGMENTAION_MODEL_PATH = "/home/"+getpass.getuser()+"/MiniMarket_dataset_segmentation/object_segmentation_models/"

NUM_POINTS_PER_SEG_SAMPLE = 20480

TARGET_OBJECT_DATASET_NAME = "shampoo_head_and_shoulders_citrus_400ml_1200_2048_segmentation_4800"





LR = 0.0001
REG_WEIGHT = 0.001

dataset = SegmentationDataset(SEGMENTAION_DATASET_PATH, TARGET_OBJECT_DATASET_NAME, NUM_POINTS_PER_SEG_SAMPLE)
train_dataset, valid_dataset = torch.utils.data.random_split(dataset,[4000,800])
#train_dataset, valid_dataset = torch.utils.data.random_split(dataset,[10000,2000])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Visualising segmentation sample
labels_map = {
    1: "Target",
    0: "Alien",
}
fig = plt.figure(figsize=(20, 10))
for i in range(BATCH_SIZE):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    points, label = train_dataset[sample_idx]
    ax = fig.add_subplot(2, int(BATCH_SIZE/2), i + 1, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:,3:])
    ax.set_axis_off()
    #plt.title(labels_map[label])
    plt.axis("off")
#plt.show()

# Plot sample
#fig = plt.figure(figsize=(5, 5))
#ax = Axes3D(fig)
#ax = fig.add_subplot(1, 2, 1, projection="3d")
#ax.set_xlim3d(-0.4, 0.4)
#ax.set_ylim3d(-0.4, 0.4)
#ax.set_zlim3d(-0.4, 0.4)
#sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
#points, label = train_dataset[sample_idx]
#print(points.shape)
#ax.scatter(points[:,0], points[:,1], points[:,2], c=points[:,3:])
#ax.set_axis_off()
#plt.show()




DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE


#gamma = 1
#alpha = np.ones(NUM_CLASSES)

#points, targets = next(iter(train_dataloader))

GLOBAL_FEATS = 1024
#classifier = PointNetClassHead(k=NUM_CLASSES, num_global_feats=GLOBAL_FEATS)



#optimizer = optim.Adam(classifier.parameters(), lr=LR)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=2000, cycle_momentum=False)
#criterion = PointNetLoss(alpha=alpha, gamma=gamma, reg_weight=REG_WEIGHT).to(DEVICE)

#classifier = classifier.to(DEVICE)

#mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES).to(DEVICE)




CATEGORIES = { 'alien':0, 'target':1, }
COLOR_MAP = { 0:(255, 0, 0),   1:(0, 0, 255), }
v_map_colors = np.vectorize(lambda x : COLOR_MAP[x])
NUM_CLASSES = len(CATEGORIES)
LR = 0.0001

points, targets = next(iter(train_dataloader))
d_in = next(iter(train_dataloader))[0].size(-1)
seg_model = RandLANet(d_in, num_classes=NUM_CLASSES,device=DEVICE)

#points = points.transpose(2, 1).to(DEVICE)
points = points.to(DEVICE)
seg_model = seg_model.to(DEVICE)
#out, _= seg_model(points.float())
out = seg_model(points.float())

alpha = np.ones(len(CATEGORIES))
gamma = 1
optimizer = optim.Adam(seg_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, step_size_up=2000, cycle_momentum=False)
criterion = RandLANet_SegLoss(alpha=alpha, gamma=gamma, dice=True).to(DEVICE)
seg_model = seg_model.to(DEVICE)

mcc_metric = MulticlassMatthewsCorrCoef(num_classes=NUM_CLASSES).to(DEVICE)


# store best validation iou
best_iou = 0.6
best_mcc = 0.6

# lists to store metrics
train_loss = []
train_accuracy = []
train_mcc = []
train_iou = []
valid_loss = []
valid_accuracy = []
valid_mcc = []
valid_iou = []













# stuff for training
num_train_batch = int(np.ceil(len(train_dataset)/BATCH_SIZE))
num_valid_batch = int(np.ceil(len(valid_dataset)/BATCH_SIZE))

for epoch in range(1, NUM_EPOCHS + 1):
    # place model in training mode
    seg_model = seg_model.train()
    _train_loss = []
    _train_accuracy = []
    _train_mcc = []
    _train_iou = []

    
    for i, (points, targets) in enumerate(train_dataloader, 0):

        #points = points.transpose(2, 1).to(DEVICE)
        points = points.to(DEVICE)
        #print("points.shape",points.shape)
        targets = targets.type(torch.LongTensor).squeeze().to(DEVICE)
        # zero gradients
        optimizer.zero_grad()
        
        # get predicted class logits
        #preds, _= seg_model(points.float())
        preds = seg_model(points.float())

        # get class predictions
        pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
        
        # get loss and perform backprop
        # Convert targets from one hot so torch can use it
        targets = torch.argmax(targets, dim=2)
        loss = criterion(preds, targets, pred_choice) 
        
        loss.backward()
        optimizer.step()
        scheduler.step() # update learning rate
        
        # get metrics
        correct = pred_choice.eq(targets.data).cpu().sum()
        accuracy = correct/float(BATCH_SIZE*NUM_POINTS_PER_SEG_SAMPLE)
        mcc = mcc_metric(preds.transpose(2, 1), targets)
        iou = compute_iou(targets, pred_choice)

        # update epoch loss and accuracy
        _train_loss.append(loss.item())
        _train_accuracy.append(accuracy)
        _train_mcc.append(mcc.item())
        _train_iou.append(iou.item())

        if (i+1) % 100 == 0:
            print(f'\t [{epoch}: {i+1}/{num_train_batch}] ' \
                  + f'train loss: {loss.item():.4f} ' \
                  + f'accuracy: {accuracy:.4f} ' \
                  + f'mcc: {np.mean(_train_mcc):.4f} ' \
                  + f'iou: {np.mean(_train_iou):.4f}')
        
    train_loss.append(np.mean(_train_loss))
    train_accuracy.append(np.mean(_train_accuracy))
    train_mcc.append(np.mean(_train_mcc))
    train_iou.append(np.mean(_train_iou))

    print(f'Epoch: {epoch} - Train Loss: {train_loss[-1]:.4f} ' \
          + f'- Train Accuracy: {train_accuracy[-1]:.4f} ' \
          + f'- Train MCC: {train_mcc[-1]:.4f} ' \
          + f'- Train IOU: {train_iou[-1]:.4f}')

    # pause to cool down
    time.sleep(1)

    # get test results after each epoch
    with torch.no_grad():

        # place model in evaluation mode
        seg_model = seg_model.eval()

        _valid_loss = []
        _valid_accuracy = []
        _valid_mcc = []
        _valid_iou = []
        for i, (points, targets) in enumerate(valid_dataloader, 0):

            #points = points.transpose(2, 1).to(DEVICE)
            points = points.to(DEVICE)
            targets = targets.type(torch.LongTensor).squeeze().to(DEVICE)
            #preds, _= seg_model(points.float())
            preds = seg_model(points.float())
            
            pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
            
            # Convert targets from one hot so torch can use it
            targets = torch.argmax(targets, dim=2)
            loss = criterion(preds, targets, pred_choice) 

            # get metrics
            correct = pred_choice.eq(targets.data).cpu().sum()
            accuracy = correct/float(BATCH_SIZE*NUM_POINTS_PER_SEG_SAMPLE)
            mcc = mcc_metric(preds.transpose(2, 1), targets)
            iou = compute_iou(targets, pred_choice)

            # update epoch loss and accuracy
            _valid_loss.append(loss.item())
            _valid_accuracy.append(accuracy)
            _valid_mcc.append(mcc.item())
            _valid_iou.append(iou.item())

            if (i+1) % 100 == 0:
            #if (i/100)>0 and (i/100).is_integer():
                print(f'\t [{epoch}: {i+1}/{num_valid_batch}] ' \
                  + f'valid loss: {loss.item():.4f} ' \
                  + f'accuracy: {accuracy:.4f} '
                  + f'mcc: {np.mean(_valid_mcc):.4f} ' \
                  + f'iou: {np.mean(_valid_iou):.4f}')
        
        valid_loss.append(np.mean(_valid_loss))
        valid_accuracy.append(np.mean(_valid_accuracy))
        valid_mcc.append(np.mean(_valid_mcc))
        valid_iou.append(np.mean(_valid_iou))
        print(f'Epoch: {epoch} - Valid Loss: {valid_loss[-1]:.4f} ' \
              + f'- Valid Accuracy: {valid_accuracy[-1]:.4f} ' \
              + f'- Valid MCC: {valid_mcc[-1]:.4f} ' \
              + f'- Valid IOU: {valid_iou[-1]:.4f}')


        # pause to cool down
        time.sleep(3)

    # save best models
    if valid_iou[-1] >= best_iou:
        best_iou = valid_iou[-1]
        torch.save(seg_model.state_dict(), SEGMENTAION_MODEL_PATH + "RandLANet_" + TARGET_OBJECT_DATASET_NAME + f'_seg_model_{epoch}.pth')


fig, ax = plt.subplots(4, 1, figsize=(8, 5))
ax[0].plot(np.arange(1, NUM_EPOCHS + 1), train_loss, label='train')
ax[0].plot(np.arange(1, NUM_EPOCHS + 1), valid_loss, label='valid')
ax[0].set_title('loss')

ax[1].plot(np.arange(1, NUM_EPOCHS + 1), train_accuracy)
ax[1].plot(np.arange(1, NUM_EPOCHS + 1), valid_accuracy)
ax[1].set_title('accuracy')

ax[2].plot(np.arange(1, NUM_EPOCHS + 1), train_mcc)
ax[2].plot(np.arange(1, NUM_EPOCHS + 1), valid_mcc)
ax[2].set_title('mcc')

ax[3].plot(np.arange(1, NUM_EPOCHS + 1), train_iou)
ax[3].plot(np.arange(1, NUM_EPOCHS + 1), valid_iou)
ax[3].set_title('iou')

fig.legend(loc='upper right')
plt.subplots_adjust(wspace=0., hspace=0.85)
plt.show()






