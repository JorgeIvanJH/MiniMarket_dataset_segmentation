import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Tnet(nn.Module):
    '''Learns a Transformation matrix for the specified dimension'''
    def __init__(self, num_channels):
        super(Tnet, self).__init__()

        self.num_channels = num_channels 

        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_channels**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)        

    def forward(self, x):

        bs = x.shape[0]

        # shared MLP(64, 128, 1024)
        x = F.relu(self.bn1(self.conv1(x))) # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x))) # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x))) # (B, 1024, N)

        # max pooling across points
        x = torch.max(x, dim=2, keepdim=False)[0] # (B, 1024)

        # fully connected layers with output sizes 512, 256
        x = F.relu(self.bn4(self.linear1(x))) # (B, 512)
        x = F.relu(self.bn5(self.linear2(x))) # (B, 256)

        # Resize to build transformation matrix, hence **2
        x = self.linear3(x) # (B, num_channels^2)

        # initialize identity matrix
        iden = torch.eye(self.num_channels) # identity matrix for 1 channel
        iden = iden.repeat(bs, 1, 1).to(x.device) # repeat for each batch
        
        # reshape to get transformation matrix
        x = x.view(-1, self.num_channels, self.num_channels) 
        # add identity matrix to the transformation matrix for regularization
        x = x + iden

        return x



class PointNetBackbone(nn.Module):
    '''The entire backbone before the classification or segmentation heads''' 
    def __init__(self,num_channels, append_local_feat=True, return_critical_indexes=False):
        super(PointNetBackbone, self).__init__()

        self.append_local_feat = append_local_feat
        self.num_channels = num_channels
        self.return_critical_indexes = return_critical_indexes

        # Spatial Transformer Networks (T-nets)
        self.tnet1 = Tnet(3) # For the xyz coordinates
        self.tnet2 = Tnet(64) # For the 64 dimensional features

        # shared MLP 1
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        # shared MLP 2
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1)
        
        # batch norms for both shared MLPs
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    
    def forward(self, x):
        
        # get batch shape
        B, C, N = x.shape # (B, num_channels, num_points)
        
        if C < 3:
            raise ValueError("Input must have at least 3 channels for xyz coordinates.")
        
        # Get and Apply transform to position channels only
        xyz = x[:, :3, :]
        Transf1 = self.tnet1(xyz) # (B, 3, 3)
        transformed_xyz = torch.bmm(xyz.transpose(2, 1), Transf1).transpose(2, 1)
        if C > 3:
            features = x[:, 3:, :]
            x = torch.cat((transformed_xyz, features), dim=1)
        else:
            x = transformed_xyz

        # shared MLP(64, 64)
        x = F.relu(self.bn1(self.conv1(x))) # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x))) # (B, 64, N)
        
        # Get and apply Second Transform matrix
        Transf2 = self.tnet2(x) # (B, 64, 64)
        x = torch.bmm(x.transpose(2, 1), Transf2).transpose(2, 1) # (B, 64, N)
        local_features = x.clone() # (B, 64, N)

        # shared MLP(64, 128, 1024)
        x = F.relu(self.bn3(self.conv3(x))) # (B, 64, N)
        x = F.relu(self.bn4(self.conv4(x))) # (B, 128, N)
        x = F.relu(self.bn5(self.conv5(x))) # (B, 1024, N)

        # Max pooling to get global features
        global_features, critical_indexes = torch.max(x, dim=2, keepdim=True)  # (B, 1024, 1), (B, 1024)
        global_features = global_features.view(B, -1) # (B, 1024)

        # Output
        if self.append_local_feat: # for Segmentation
            global_expanded = global_features.unsqueeze(-1) # (B, 1024, 1) # extra dim for broadcasting
            global_expanded = global_expanded.repeat(1, 1, N) # (B, 1024, N)
            locnglob_features = torch.cat((local_features, global_expanded), dim=1) # (B, 1088, N)

            if self.return_critical_indexes:
                return locnglob_features, critical_indexes # (B, 1088, N), (B, 1024)
            else:
                return locnglob_features # (B, 1088, N)
            
        else: # for Classification
            if self.return_critical_indexes:
                return global_features, critical_indexes # (B, 1024), (B, 1024)
            else:
                return global_features # (B, 1024)



class PointNetClassHead(nn.Module):
    '''' Classification Head '''
    def __init__(self, num_channels=3, return_critical_indexes=False, num_output_classes=2):
        super(PointNetClassHead, self).__init__()

        self.return_critical_indexes = return_critical_indexes

        # get the backbone (only need global features for classification)
        self.backbone = PointNetBackbone(num_channels,False,return_critical_indexes)

        # MLP (512, 256, k)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_output_classes)

        # batchnorm for the first 2 linear layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        # dropout for the last linear layer
        self.dropout = nn.Dropout(p=0.3) # "keep ratio 0.7 on the last fully connected layer"
        

    def forward(self, x):

        if self.return_critical_indexes:
            # get global features and critical indexes
            x, crit_idxs = self.backbone(x) # (B, 1024) , (B, 1024)
        else:
            # get global features
            x = self.backbone(x) # (B, 1024)

        x = F.relu(self.bn1(self.linear1(x))) # (B, 512)
        x = F.relu(self.bn2(self.linear2(x))) # (B, 256)
        x = self.dropout(x) # (B, 256)
        x = self.linear3(x) # (B, num_output_classes)

        if self.return_critical_indexes:
            # return logits and critical indexes
            return x, crit_idxs # (B, num_output_classes), (B, 1024)
        else:
            # return logits only
            return x # (B, num_output_classes)


class PointNetSegHead(nn.Module):
    ''' Segmentation Head '''
    def __init__(self, num_channels=6, return_critical_indexes=False, num_output_classes=2):
        super(PointNetSegHead, self).__init__()

        self.return_critical_indexes = return_critical_indexes

        # get the backbone 
        self.backbone = PointNetBackbone(num_channels,True,return_critical_indexes)

        # shared MLP(512, 256, 128, num_output_classes)
        self.conv1 = nn.Conv1d(1088, 512, kernel_size=1)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=1)
        self.conv4 = nn.Conv1d(128, num_output_classes, kernel_size=1)

        # batch norms for shared MLP
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)


    def forward(self, x):
    

        if self.return_critical_indexes:
            # get global features and critical indexes
            x, crit_idxs = self.backbone(x) # (B, 1088, N), (B, 1024)
        else:
            # get global features
            x = self.backbone(x) # (B, 1088, N)

        # pass through shared MLP
        x = F.relu(self.bn1(self.conv1(x))) # (B, 512, N)
        x = F.relu(self.bn2(self.conv2(x))) # (B, 256, N)
        x = F.relu(self.bn3(self.conv3(x))) # (B, 128, N)
        x = self.conv4(x) # (B, num_output_classes, N)

        x = x.transpose(2, 1) # (B, N, num_output_classes)
        
        if self.return_critical_indexes:
            # return logits and critical indexes
            return x, crit_idxs # (B, N, num_output_classes), (B, 1024)
        else:
            # return logits only
            return x # (B, N, num_output_classes)


class PointNetSegLoss(nn.Module):
    def __init__(self, alpha=None, reg_weight=0.001):
        super(PointNetSegLoss, self).__init__()
        self.reg_weight = reg_weight

        # Prepare class weights if provided
        if isinstance(alpha, (float, int)):
            alpha = torch.Tensor([alpha, 1 - alpha])
        elif isinstance(alpha, (list, np.ndarray)):
            alpha = torch.Tensor(alpha)
        
        self.class_weights = alpha
        self.cross_entropy = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, logits, targets, transform_matrix=None):
        """
        logits: (B, N, num_classes)
        targets: (B, N)
        transform_matrix: (B, D, D), from T-Net
        """
        # Cross-entropy loss over per-point logits
        ce_loss = self.cross_entropy(logits.transpose(2, 1), targets)

        # Orthogonality regularization loss
        if transform_matrix is not None:
            I = torch.eye(transform_matrix.size(1), device=transform_matrix.device).unsqueeze(0)
            AAT = torch.bmm(transform_matrix, transform_matrix.transpose(2, 1))
            reg_loss = ((AAT - I) ** 2).sum(dim=(1, 2)).mean()
            total_loss = ce_loss + self.reg_weight * reg_loss
        else:
            total_loss = ce_loss

        return total_loss


def compute_iou(targets, predictions):

    targets = targets.reshape(-1)
    predictions = predictions.reshape(-1)

    intersection = torch.sum(predictions == targets) # true positives
    union = len(predictions) + len(targets) - intersection

    return intersection / union
