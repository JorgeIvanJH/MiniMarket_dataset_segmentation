import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchmetrics.classification import MulticlassMatthewsCorrCoef
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
import h5py

from SuperMarketDataset import SegmentationDataset

from item_pointnet_torch import compute_iou
from item_Rand_LA_Net_torch import RandLANet
from item_Rand_LA_Net_torch import RandLANet_SegLoss


# CHANGEABLES
PC_SEGMENTATION_DIR = r"D:\Datasets\MinimarketPointCloud\MiniMarket_point_clouds\2048\segmentation_dataset\ketchup_heinz_400ml_segmentation_20250526_121710_numPoints_2048_maxObjects_10_orientations_1.h5"
# PC_SEGMENTATION_DIR = r"D:\Datasets\MinimarketPointCloud\MiniMarket_point_clouds\64\segmentation_dataset\ketchup_heinz_400ml_segmentation_date_20250526_time_163143_numPoints_64_maxObjects_10_numOrientations_10.h5"
dataset_name = os.path.basename(PC_SEGMENTATION_DIR)
SAVE_DIR = os.path.join(os.getcwd(), "experiments", dataset_name)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
BATCH_SIZE = 8
NUM_EPOCHS = 10
LR = 0.0001
REG_WEIGHT = 0.001

with h5py.File(PC_SEGMENTATION_DIR, 'r') as f:
    # Read datasets
    seg_points = f["seg_points"][:]
    seg_colors = f["seg_colors"][:]
    seg_labels = f["seg_labels"][:]
    NUM_CLOUDS,NUM_POINTS_PER_SEG_SAMPLE,NUM_CLASSES = seg_labels.shape


dataset = SegmentationDataset(PC_SEGMENTATION_DIR )

dataset_size = NUM_CLOUDS
train_size = int(0.8 * dataset_size) # 80% training set
val_size = dataset_size - train_size # 20% validation set

train_dataset, valid_dataset = torch.utils.data.random_split(dataset,[train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LR = 0.0001

points, targets = next(iter(train_dataloader))

seg_model = RandLANet(num_points=NUM_POINTS_PER_SEG_SAMPLE, m=NUM_CLASSES)

# out, _ = seg_model(points.float().transpose(2, 1))
# print(f'Seg shape: {out.shape}')

alpha = np.ones(NUM_CLASSES)
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
    print(f'Epoch {epoch}/{NUM_EPOCHS}')
    # place model in training mode
    seg_model = seg_model.train()
    _train_loss = []
    _train_accuracy = []
    _train_mcc = []
    _train_iou = []
    for i, (points, targets) in enumerate(train_dataloader, 0):
        print(f'\t [{epoch}: {i+1}/{num_train_batch}] ' )
        points = points.transpose(2, 1).to(DEVICE)
        targets = targets.type(torch.LongTensor).squeeze().to(DEVICE)
        # zero gradients
        optimizer.zero_grad()
        
        # get predicted class logits
        preds = seg_model(points.float())
        # get class predictions
        
        pred_choice = torch.argmax(preds, dim=2)
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
        mcc = mcc_metric(preds.reshape(-1, 2), targets.reshape(-1))
        iou = compute_iou(targets, pred_choice)
        print("iou: ", iou.item())
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

            points = points.transpose(2, 1).to(DEVICE)
            targets = targets.type(torch.LongTensor).squeeze().to(DEVICE)
            preds = seg_model(points.float())
            
            pred_choice = torch.argmax(preds, dim=2)
            
            # Convert targets from one hot so torch can use it
            targets = torch.argmax(targets, dim=2)
            loss = criterion(preds, targets, pred_choice) 

            # get metrics
            correct = pred_choice.eq(targets.data).cpu().sum()
            accuracy = correct/float(BATCH_SIZE*NUM_POINTS_PER_SEG_SAMPLE)
            mcc = mcc_metric(preds.reshape(-1, 2), targets.reshape(-1))
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
        path_w = os.path.join(SAVE_DIR, f'RandLaNet_{dataset_name}_best_iou_{best_iou}.pth')
        torch.save(seg_model.state_dict(),path_w)

    # Update Graph
    fig, ax = plt.subplots(4, 1, figsize=(8, 5))
    ax[0].plot(np.arange(1, epoch + 1), train_loss, label='train')
    ax[0].plot(np.arange(1, epoch + 1), valid_loss, label='valid')
    ax[0].set_title('loss')

    ax[1].plot(np.arange(1, epoch + 1), train_accuracy)
    ax[1].plot(np.arange(1, epoch + 1), valid_accuracy)
    ax[1].set_title('accuracy')

    ax[2].plot(np.arange(1, epoch + 1), train_mcc)
    ax[2].plot(np.arange(1, epoch + 1), valid_mcc)
    ax[2].set_title('mcc')

    ax[3].plot(np.arange(1, epoch + 1), train_iou)
    ax[3].plot(np.arange(1, epoch + 1), valid_iou)
    ax[3].set_title('iou')

    fig.legend(loc='upper right')
    plt.subplots_adjust(wspace=0., hspace=0.85)
    path_i = os.path.join(SAVE_DIR, f'training_metrics_plot.png')
    fig.savefig(path_i)