You can generate the segmentation dataset (mask) for a particular object using the repository: 
https://github.com/msorour/MiniMarket_dataset_processing

and the raw object dataset on kaggle link below:
https://kaggle.com/datasets/83896356f3cefb84a1256545154992a94d8ed5495c49b901bff8471c30daaacc

Once generated (can be several gigabytes in size depending on point cloud resolution used) you should place it in the folder object_segmentation_dataset

Three point cloud based deep networks are adapted for our MiniMarket dataset:
1. PointNet
2. PointNet2 (or PointNet++0)
3. RandLA-Net 

Use the conda environment py39.yml to run the segmentation algorithms.
