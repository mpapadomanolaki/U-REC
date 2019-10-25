from torch.utils.data.dataset import Dataset
from skimage import io
import numpy as np
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, csv_path, image_ids, image_folder, label_folder, patch_size):
        # Read the csv file and shuffle it
        self.data_info = pd.read_csv(csv_path, header=None)
        self.data_info = self.data_info.iloc[np.random.permutation(len(self.data_info))]
        # First column contains the image paths
        self.X = self.data_info.iloc[:,0]
        # Second column is the labels
        self.Y = self.data_info.iloc[:,1]
        # Third column is the image ID
        self.IM_id = self.data_info.iloc[:,2]

        ###create image/label list
        self.images = []
        self.labels = []
        for id in image_ids:
            self.images.append(io.imread(image_folder + 'top_mosaic_09cm_area{}.tif'.format(id))/255.0)  #divide by 255 for normalization          
            self.labels.append(io.imread(label_folder + 'top_mosaic_09cm_area{}.tif'.format(id)))

        self.patch_size = patch_size
        # Calculate len
        self.data_len = self.data_info.shape[0]-1

    def __getitem__(self, index):
        # Get image patch
        x = int(self.X[index])
        y = int(self.Y[index])
        image_id = int(self.IM_id[index]) - 7000

        image_patch = self.images[image_id] [x:x + self.patch_size, y:y + self.patch_size, :]
        image_patch = np.transpose(image_patch, (2,0,1))
        label_patch = self.labels[image_id] [x:x + self.patch_size, y:y + self.patch_size]

        return (image_patch, label_patch)

    def __len__(self):
        return self.data_len
