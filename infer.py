import numpy as np
from skimage import io
import os
from tqdm import tqdm
import torch
from torch.autograd import Variable
import u_rec
import tools

def to_color(label):
    colors = [[255, 255, 255],
             [0, 0, 255],
             [0, 255, 255],
             [0, 255, 0],
             [255, 255, 0],
             [255, 0, 0],
             [0, 0 , 0]]
    return np.asarray(colors[int(label)])

def sliding_window(IMAGE, patch_size, step):
    prediction = np.zeros((IMAGE.shape[0], IMAGE.shape[1], 6))
    
    x=0
    while (x!=IMAGE.shape[0]):
     y=0
     while(y!=IMAGE.shape[1]):

               if (not y+patch_size > IMAGE.shape[1]) and (not x+patch_size > IMAGE.shape[0]):
                patch = IMAGE[x:x + patch_size, y:y + patch_size, :]
                patch = np.transpose(patch, (2,0,1))/255.0 #normalization
                patch = np.reshape(patch, (1, patch.shape[0], patch.shape[1], patch.shape[2]))
                patch = tools.to_cuda(torch.from_numpy(patch).float())
                output, rec = model(patch)
                output = output.cpu().data.numpy().squeeze()
                output = np.transpose(output, (1,2,0))
                patch_pred = np.zeros((patch_size, patch_size))
                for i in range(0, patch_size):
                    for j in range(0, patch_size):
                        prediction[x+i, y+j] += (output[i,j,:])

                stride=step

               if y + patch_size == IMAGE.shape[1]:
                  break

               if y + patch_size > IMAGE.shape[1]:
                y = IMAGE.shape[1] - patch_size
               else:
                y = y+stride

     if x + patch_size == IMAGE.shape[0]:
        break

     if x + patch_size > IMAGE.shape[0]:
       x = IMAGE.shape[0] - patch_size
     else:
      x = x+stride

    final_pred = np.zeros((IMAGE.shape[0], IMAGE.shape[1], 3))
    print('ok')
    for i in range(0, final_pred.shape[0]):
        for j in range(0, final_pred.shape[1]):
            final_pred[i,j] = to_color(np.argmax(prediction[i,j]))

    return final_pred

model=u_rec.UNet(3,6)
model.load_state_dict(torch.load('./models/model_2.pt'))
model = tools.to_cuda(model)
model = model.eval()
infer_ids=[10,12,14,16,20,22,24,27,29,2,31,33,35,38,4,6,8]
patch_size = 256
step = 128
tifs_folder = '../top/' #folder with the testing Vaihingen images
save_folder = 'PREDICTIONS'
os.mkdir(save_folder)

for i in infer_ids:
  print('Processing area ', i)
  img = io.imread(tifs_folder + 'top_mosaic_09cm_area{}.tif'.format(i))
  pred = sliding_window(img, patch_size, step)
  io.imsave('./' + save_folder + '/top_mosaic_09cm_area{}.tif'.format(i), pred)
