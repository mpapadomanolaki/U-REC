import numpy as np
from skimage import io
from skimage.transform import rotate, resize
import os
import cv2
import pandas as pd

train_ids = [11, 13, 1, 21, 23, 26, 28, 30, 32, 34, 37, 3, 5, 7]
infer_ids = [15,17]

FOLDER='/home/mariapap/DATA/GROUNDTRUTH/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE_1D/'

step=64
patch_s=256


def sliding_window(i_city, ss, labeled_areas, label, window_size, step):
    city=[]
    fpatches_labels=[]

    x=0
    while (x!=label.shape[0]):
     y=0
     while(y!=label.shape[1]):

               if (not y+window_size > label.shape[1]) and (not x+window_size > label.shape[0]):
                line=np.array([x,y, labeled_areas.index(ss)])
                city.append(line)
                stride=step

               if y + window_size == label.shape[1]:
                  break

               if y + window_size > label.shape[1]:
                y = label.shape[1] - window_size
               else:
                y = y+stride

     if x + window_size == label.shape[0]:
        break

     if x + window_size > label.shape[0]:
       x = label.shape[0] - window_size
     else:
      x = x+stride

    return np.asarray(city)


cities=[]
print('train areas:')
for i in train_ids:
 path=FOLDER+'top_mosaic_09cm_area{}.tif'.format(i)
 im_name = os.path.basename(path)
 ss = im_name[im_name.find('_area')+5:im_name.find('.tif')]
 print(ss)
 train_gt = io.imread(path)
 xy_city =  sliding_window(i, int(ss), train_ids, train_gt, patch_s, step)
 cities.append(xy_city)

final_cities = np.concatenate(cities, axis=0)
df = pd.DataFrame({'X': list(final_cities[:,0]),
                   'Y': list(final_cities[:,1]),
                   'image_ID': list(final_cities[:,2]),
                   })
df.to_csv('./xys/myxys_train.csv', index=False, columns=["X", "Y", "image_ID"])
#np.save('./xys/xys_train.npy', final_cities)

#####################################################################

cities=[]
print('validation areas:')
for i in infer_ids:   ##THIS
 path=FOLDER+'top_mosaic_09cm_area{}.tif'.format(i)
 im_name = os.path.basename(path)
 ss = im_name[im_name.find('_area')+5:im_name.find('.tif')]
 print(ss)
 train_gt = io.imread(path)
 xy_city =  sliding_window(i, int(ss), infer_ids, train_gt, patch_s, patch_s)
 cities.append(xy_city)

final_cities = np.concatenate(cities, axis=0)
df = pd.DataFrame({'X': list(final_cities[:,0]),
                   'Y': list(final_cities[:,1]),
                   'image_ID': list(final_cities[:,2]),
                   })
df.to_csv('./xys/myxys_val.csv', index=False, columns=["X", "Y", "image_ID"])

#np.save('./xys/xys_val.npy', final_cities)
