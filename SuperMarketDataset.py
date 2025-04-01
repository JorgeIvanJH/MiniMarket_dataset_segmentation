from torch.utils.data import Dataset
import h5py
import random
import glob
import numpy as np
import os



class ClassificationDataset(Dataset):
    # number_of_alien_objects_to_train_against must be even number
    # TODO: enforce this constraint in the future
    def __init__(self, 
                 object_pointcloud_dataset_directory, 
                 target_object_name, 
                 number_of_alien_objects_to_train_against):
        self.dataset_dir = object_pointcloud_dataset_directory
        self.target_obj = target_object_name
        self.num_alien_obj = number_of_alien_objects_to_train_against
        
        class_map = {target_object_name,"alien_item"}
        
        # Get the target object file and all alien object files 
        # A file is a point cloud file with 1200 samples of a particular object, each sample is 2048 colored points
        target_hdf5_file = object_pointcloud_dataset_directory + target_object_name
        alien_hdf5_files = sorted(glob.glob(object_pointcloud_dataset_directory+"*"))
        alien_hdf5_files.remove(target_hdf5_file)
        
        # Remove files randomly from the set of alien files to keep only the desired number of alien objects
        while len(alien_hdf5_files) > number_of_alien_objects_to_train_against:
            alien_hdf5_files.pop(random.randrange(len(alien_hdf5_files)))

        # Target class training data collection
        with h5py.File(target_hdf5_file, "r") as f:
            point_data = f["point_clouds"][()]  # returns as a numpy array
            color_data = f["color_clouds"][()]  # returns as a numpy array
        self.dataset_samples = np.concatenate( (point_data, color_data),axis=2)
        self.dataset_labels = np.ones(self.dataset_samples.shape[0])
        
        # Alien class training data collection
        number_of_alien_object_samples = int(1200/number_of_alien_objects_to_train_against) # to satisfy 50% alien objects 50% target object dataset
        for i, alien_class_files in enumerate(alien_hdf5_files):
            with h5py.File(alien_hdf5_files[i], "r") as f:
                point_data = f["point_clouds"][()]  # returns as a numpy array
                color_data = f["color_clouds"][()]  # returns as a numpy array
            this_alien_object_samples = np.concatenate( (point_data, color_data),axis=2)
            np.random.shuffle(this_alien_object_samples)
            self.dataset_samples = np.append(self.dataset_samples, this_alien_object_samples[:number_of_alien_object_samples] , axis=0)
            self.dataset_labels = np.append(self.dataset_labels, np.zeros(number_of_alien_object_samples) , axis=0)
        print(self.dataset_samples.shape)
        print(self.dataset_labels.shape)
        
    def __len__(self):
        return len(self.dataset_labels)

    def __getitem__(self, index):
        point_cloud_sample = self.dataset_samples[index]
        label = self.dataset_labels[index]
        return point_cloud_sample, label


class SegmentationDataset(Dataset):
    # number_of_alien_objects_to_train_against must be even number
    def __init__(self, 
                 object_segmentation_dataset_directory, 
                 target_object_dataset_name, 
                 number_of_points_per_segmentation_sample):
        self.dataset_dir = object_segmentation_dataset_directory
        self.target_obj = target_object_dataset_name
        self.num_points_per_seg_sample = number_of_points_per_segmentation_sample
        
        # Get the target object segmentation file
        #target_hdf5_file = self.dataset_dir + target_object_name + "_segmentation_" + str(self.num_points_per_seg_sample)
        #target_hdf5_file = self.dataset_dir + "coffee_nescafe_3in1_original_6cups_1200_2048_segmentation_20480_12000"
        #target_hdf5_file = self.dataset_dir + "coffee_nescafe_3in1_original_6cups_1200_2048_segmentation_20480_4800"
        #target_hdf5_file = self.dataset_dir + "shampoo_head_and_shoulders_citrus_400ml_1200_2048_segmentation_4800"
        # target_hdf5_file = self.dataset_dir + target_object_dataset_name
        target_hdf5_file = os.path.join(self.dataset_dir, target_object_dataset_name)
        # Target class training data collection
        with h5py.File(target_hdf5_file, "r") as f:
            point_data = f["seg_points"][()]  # returns as a numpy array
            color_data = f["seg_colors"][()]  # returns as a numpy array
            label_data = f["seg_labels"][()]  # returns as a numpy array
        self.dataset_samples = np.concatenate( (point_data, color_data),axis=2)
        self.dataset_labels = label_data
        #print(self.dataset_samples.shape)
        #print(self.dataset_labels.shape)
        
    def __len__(self):
        return len(self.dataset_labels)

    def __getitem__(self, index):
        point_cloud_sample = self.dataset_samples[index]
        label = self.dataset_labels[index]
        return point_cloud_sample, label