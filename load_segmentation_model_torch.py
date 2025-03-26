import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchmetrics.classification import MulticlassMatthewsCorrCoef
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
from matplotlib import pyplot as plt

import open3d as o3d #  
import random

import getpass

from SuperMarketDataset import SegmentationDataset

from item_pointnet_torch import PointNetSegHead
from item_pointnet2_torch import PointNet2_SegHead
from item_Rand_LA_Net_torch import RandLANet

SEGMENTAION_DATASET_PATH = "/home/"+getpass.getuser()+"/MiniMarket_dataset_segmentation/object_segmentation_dataset/"
SEGMENTAION_MODEL_PATH = "/home/"+getpass.getuser()+"/MiniMarket_dataset_segmentation/object_segmentation_models/"

REAL_WORLD_OBJECT_SCENES = "/home/"+getpass.getuser()+"/MiniMarket_dataset_segmentation/real_world_object_scenes/scene2_cam2.pcd"


TARGET_OBJECT_DATASET_NAME = "shampoo_head_and_shoulders_citrus_400ml_1200_2048_segmentation_4800"


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#DEVICE =  'cpu'




#Network = "PointNet"
#MODEL = "pointnet_shampoo_head_and_shoulders_citrus_400ml_1200_2048_segmentation_4800seg_model_9.pth"
#Network = "PointNet2"
#MODEL = "pointnet2_shampoo_head_and_shoulders_citrus_400ml_1200_2048_segmentation_4800seg_model_10.pth"
Network = "RandLANet"
MODEL = "RandLANet_shampoo_head_and_shoulders_citrus_400ml_1200_2048_segmentation_4800seg_model_6.pth"
NUM_POINTS_PER_SEG_SAMPLE = 20480
BATCH_SIZE = 2

dataset = SegmentationDataset(SEGMENTAION_DATASET_PATH, TARGET_OBJECT_DATASET_NAME, NUM_POINTS_PER_SEG_SAMPLE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
points, targets = next(iter(dataloader))

if(Network=="PointNet"):
    # PointNet
    DEVICE =  'cpu'
    seg_model = PointNetSegHead(num_points=NUM_POINTS_PER_SEG_SAMPLE, m=2)

    seg_model.load_state_dict(torch.load(SEGMENTAION_MODEL_PATH+MODEL))
    seg_model.eval()
    
    points = points.transpose(2, 1).to(DEVICE)
    targets = targets.type(torch.LongTensor).squeeze().to(DEVICE)

    # get predicted class logits
    preds, _, _ = seg_model(points.float())

    # get class predictions
    pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
    targets = torch.argmax(targets, dim=2)

    # Invert the points back to normal to be able to plot
    points = points.transpose(2, 1).to(DEVICE)

elif(Network=="PointNet2"):
    # PointNet2
    DEVICE =  'cpu'
    seg_model = PointNet2_SegHead(2)
    seg_model.load_state_dict(torch.load(SEGMENTAION_MODEL_PATH+MODEL))
    seg_model.eval()

    points = points.transpose(2, 1).to(DEVICE)
    targets = targets.type(torch.LongTensor).squeeze().to(DEVICE)

    # get predicted class logits
    preds, _= seg_model(points.float())

    # get class predictions
    pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
    targets = torch.argmax(targets, dim=2)

    # Invert the points back to normal to be able to plot
    points = points.transpose(2, 1).to(DEVICE)

elif(Network=="RandLANet"):
    # RandLANet
    seg_model = RandLANet(6, num_classes=2,device=DEVICE)
    seg_model.load_state_dict(torch.load(SEGMENTAION_MODEL_PATH+MODEL))
    seg_model.eval()

    points = points.to(DEVICE)
    targets = targets.type(torch.LongTensor).squeeze().to(DEVICE)

    preds = seg_model(points.float()).to(DEVICE)

    # get class predictions
    pred_choice = torch.softmax(preds, dim=2).argmax(dim=2).to(DEVICE)

    targets = torch.argmax(targets, dim=2)
    DEVICE =  'cpu'
    points = points.to(DEVICE)
    targets = targets.to(DEVICE)
    pred_choice = pred_choice.to(DEVICE)







# Visualising segmentation sample
labels_map = {
    1: "Target",
    0: "Alien",
}
fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.scatter(points[0,:, 0], points[0,:, 1], points[0,:, 2], c=points[0,:,3:])
ax.set_axis_off()
#plt.title(labels_map[label])
plt.axis("off")

ax = fig.add_subplot(1, 3, 2, projection="3d")
r = np.zeros((pred_choice[0,:].shape[0],1))
g = np.zeros((pred_choice[0,:].shape[0],1))
b = pred_choice[0,:].reshape((pred_choice[0,:].shape[0],1))
ax.scatter(points[0,:, 0], points[0,:, 1], points[0,:, 2], c=np.concatenate( (r, g, b),axis=1))

ax = fig.add_subplot(1, 3, 3, projection="3d")
r = np.zeros((targets[0,:].shape[0],1))
g = np.zeros((targets[0,:].shape[0],1))
b = targets[0,:].reshape((targets[0,:].shape[0],1))
ax.scatter(points[0,:, 0], points[0,:, 1], points[0,:, 2], c=np.concatenate( (r, g, b),axis=1))

#plt.title(labels_map[label])
plt.axis("off")

plt.show()




######################################
## Real world object pointcloud scene
######################################
def parse_real_world_pcd(path, desired_number_of_sample_points):
    # Load the point cloud
    cloud = o3d.io.read_point_cloud(path)
    points= np.asarray(cloud.points)
    colors= np.asarray(cloud.colors)
    print("number of points = ", points.shape[0])
    
    if points.shape[0] > desired_number_of_sample_points:
        random_index = random.sample(range(0, points.shape[0]), points.shape[0]-desired_number_of_sample_points)
        downsampled_points = np.delete(points, random_index, axis=0)
        downsampled_colors = np.delete(colors, random_index, axis=0)
        point_cloud = downsampled_points
        color_cloud = downsampled_colors
        print("point cloud downsampled.")

    if points.shape[0] < desired_number_of_sample_points:
        print("Padding ...")

        # Pad the point clouds with 0s
        pad_amount = desired_number_of_sample_points - points.shape[0]
        points_padded = np.pad(points, ((0, pad_amount),(0, 0)), 'constant', constant_values=(0, 0))
        colors_padded = np.pad(colors, ((0, pad_amount),(0, 0)), 'constant', constant_values=(0, 0))
        point_cloud = points_padded
        color_cloud = colors_padded

    if points.shape[0] == desired_number_of_sample_points:
        point_cloud = points
        color_cloud = colors
    
    return np.concatenate( (point_cloud, color_cloud),axis=1)

real_data = parse_real_world_pcd(REAL_WORLD_OBJECT_SCENES, NUM_POINTS_PER_SEG_SAMPLE)
print(real_data.shape)

real_data = np.stack((real_data,) * BATCH_SIZE, axis=0)
print(real_data.shape)

real_data = torch.tensor(real_data)
print(real_data.shape)



if(Network=="PointNet"):
    #transpose points to infer
    real_data = real_data.transpose(2, 1).to(DEVICE)
    # get predicted class logits
    preds, _, _ = seg_model(real_data.float())
    # get class predictions
    pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
    # transpose again back to normal to be able to plot
    real_data = real_data.transpose(2, 1).to(DEVICE)

elif(Network=="PointNet2"):
    #transpose points to infer
    real_data = real_data.transpose(2, 1).to(DEVICE)
    preds, _= seg_model(real_data.float())
    # get class predictions
    pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
    # transpose again back to normal to be able to plot
    real_data = real_data.transpose(2, 1).to(DEVICE)

elif(Network=="RandLANet"):
    #RandLANet
    DEVICE =  'cuda'
    real_data = real_data.to(DEVICE)
    preds = seg_model(real_data.float()).to(DEVICE)
    # get class predictions
    pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
    real_data = real_data.to('cpu')
    pred_choice = pred_choice.to('cpu')




# Plotting Results
fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(1, 3, 1, projection="3d")
ax.scatter(real_data[0,:, 0], real_data[0,:, 1], real_data[0,:, 2], c=real_data[0,:,3:])
ax.set_axis_off()
#plt.title(labels_map[label])
plt.axis("off")

ax = fig.add_subplot(1, 3, 2, projection="3d")
r = np.zeros((pred_choice[0,:].shape[0],1))
g = np.zeros((pred_choice[0,:].shape[0],1))
b = pred_choice[0,:].reshape((pred_choice[0,:].shape[0],1))
ax.scatter(real_data[0,:, 0], real_data[0,:, 1], real_data[0,:, 2], c=np.concatenate( (r, g, b),axis=1))

plt.axis("off")

plt.show()