import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple
import numpy as np

import time

#from annoy import AnnoyIndex
import random
from torch_points_kernels import knn

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):

        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        self.device = device

    def forward(self, coords, features, idx, dist):

        # finding neighboring points
        #idx, dist = knn_output
        B, N, K = idx.shape
        
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx).to("cpu")
        # relative point position encoding
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)
        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):

        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, num_points, N, 1)

        return self.mlp(features)



class LocalFeatureAggregation(nn.Module):
    def __init__(self, num_points, d_out, num_neighbors, device='cpu'):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(num_points, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(num_points, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()
        

    def forward(self, coords, features):

        coords = coords.transpose(1, 2).cpu().contiguous().float()  # [B, 3, N] → [B, N, 3]
        knn_output = knn(coords,coords, self.num_neighbors)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        coords = coords.to(device)
        
        x = self.mlp1(features)

        #x = self.lse1(coords, x, knn_output)
        x = self.lse1(coords, x, knn_output[0].to(device), knn_output[1].to(device))
        x = self.pool1(x)

        #x = self.lse2(coords, x, knn_output)
        x = self.lse2(coords, x, knn_output[0].to(device), knn_output[1].to(device))
        x = self.pool2(x)

        x = x.to(device)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))

def compute_loss(end_points, dataset, criterion):

    logits = end_points['logits']
    labels = end_points['labels']

    logits = logits.transpose(1, 2).reshape(-1, dataset.m)
    labels = labels.reshape(-1)

    # Boolean mask of points that should be ignored
    ignored_bool = (labels == 0)

    for ign_label in dataset.ignored_labels:
        ignored_bool = ignored_bool | (labels == ign_label)

    # Collect logits and labels that are not ignored
    valid_idx = ignored_bool == 0
    valid_logits = logits[valid_idx, :]
    valid_labels_init = labels[valid_idx]
    # Reduce label values in the range of logit shape
    reducing_list = torch.arange(0, dataset.m).long().to(logits.device)
    inserted_value = torch.zeros((1,)).long().to(logits.device)
    for ign_label in dataset.ignored_labels:
        reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
    valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
    loss = criterion(valid_logits, valid_labels).mean()
    end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
    end_points['loss'] = loss
    return loss, end_points


class RandLANet(nn.Module):
    def __init__(self, num_points, m, num_neighbors=2, decimation=4, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(RandLANet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(6, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 32, **decoder_kwargs),
            SharedMLP(64, 8, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, m)
        )
        self.device = device

        self = self.to(device)

    def forward(self, input):
        N = input.shape[2]
        d = self.decimation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        coords = input[:, :3, :].clone().to(device)       # [B, 3, N]
        x = self.fc_start(input.transpose(1, 2)).transpose(1, 2).unsqueeze(-1)  # [B, 8, N, 1]
        x = self.bn_start(x)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []
        coord_stack = []

        permutation = torch.randperm(N)
        coords = coords[:, :, permutation]
        x = x[:, :, permutation]

        for lfa in self.encoder:
            num_points = N // decimation_ratio
            coords_i = coords[:, :, :num_points]
            x_i = x[:, :, :num_points]
            
            x_i = lfa(coords_i, x_i)

            # Save matching coords/features for skip connections
            x_stack.append(x_i.clone())
            coord_stack.append(coords_i.clone())

            # Update coords and x for next layer
            coords = coords_i
            x = x_i
            decimation_ratio *= d

        # >>>>>>>>>> BOTTLENECK
        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:
            decimation_ratio //= d

            # Retrieve matching skip connection tensors
            x_skip = x_stack.pop()
            coords_skip = coord_stack.pop()  # [B, 3, N_high]

            # Use KNN to upsample x to match resolution of x_skip
            query_coords = coords_skip.transpose(1, 2).contiguous().float().cpu()  # high-res
            support_coords = coords.transpose(1, 2).contiguous().float().cpu()     # low-res

            neighbors, _ = knn(support_coords,query_coords, 1)  # [B, N_high, 1]
            neighbors = neighbors.to(device)

            # Expand neighbor indices to gather features
            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, -1)
            x_neighbors = torch.gather(x, dim=2, index=extended_neighbors)  # [B, C, N_high, 1]

            # Fuse with skip connection
            x = torch.cat((x_neighbors, x_skip), dim=1)  # concatenate on channel dim
            x = mlp(x)

            # Update coords for next decoder step
            coords = coords_skip

        # >>>>>>>>>> POST-DECODER

        # Invert permutation
        x = x[:, :, torch.argsort(permutation)]
        scores = self.fc_end(x)  # [B, C, N, 1]
        scores = scores.permute(0, 2, 1, 3)  # [B, N, C, 1]

        return scores.squeeze(-1)  # [B, N, C]



class RandLANet_SegLoss(nn.Module):
    
    def __init__(self, alpha=None, gamma=0, size_average=True, dice=False):
        super(RandLANet_SegLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.dice = dice

        # sanitize inputs
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,(list, np.ndarray)): self.alpha = torch.Tensor(alpha)

        # get Balanced Cross Entropy Loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=self.alpha)
        

    def forward(self, predictions, targets, pred_choice=None):

        # get Balanced Cross Entropy Loss
        ce_loss = self.cross_entropy_loss(predictions.transpose(2, 1), targets)
        #ce_loss = self.cross_entropy_loss(predictions, targets.float())

        # reformat predictions (b, n, c) -> (b*n, c)
        predictions = predictions.contiguous().view(-1, predictions.size(2)) 
        # get predicted class probabilities for the true class
        pn = F.softmax(predictions)
        pn = pn.gather(1, targets.view(-1, 1)).view(-1)

        # compute loss (negative sign is included in ce_loss)
        loss = ((1 - pn)**self.gamma * ce_loss)
        if self.size_average: loss = loss.mean() 
        else: loss = loss.sum()

        # add dice coefficient if necessary
        if self.dice: return loss + self.dice_loss(targets, pred_choice, eps=1)
        else: return loss
    
    @staticmethod
    def dice_loss(predictions, targets, eps=1):
        ''' Compute Dice loss, directly compare predictions with truth '''

        targets = targets.reshape(-1)
        predictions = predictions.reshape(-1)

        cats = torch.unique(targets)

        top = 0
        bot = 0
        for c in cats:
            locs = targets == c

            # get truth and predictions for each class
            y_tru = targets[locs]
            y_hat = predictions[locs]

            top += torch.sum(y_hat == y_tru)
            bot += len(y_tru) + len(y_hat)


        return 1 - 2*((top + eps)/(bot + eps)) 