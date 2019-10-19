from skimage import io
import os
import glob
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import u_rec
import torchnet as tnt
import tools
import custom

csv_file = '../myxys.csv'
csv_file_val = '../myxys_val.csv'
image_ids = [11, 13, 1, 21, 23, 26, 28, 30, 32, 34, 37, 3, 5, 7] #train areas
infer_ids = [15,17] #validation areas
img_folder = '../top/' #folder with tif images of Vaihingen city
lbl_folder = '../groundtruth/' #folder with 2D groundtruth tif images
             #see https://github.com/nshaud/DeepNetsForEO/blob/master/legacy/notebooks/convert_gt.py for
             #the conversion
patch_size = 256
change_dataset_val = custom.MyDataset(csv_file_val, infer_ids, img_folder, lbl_folder, patch_size)

batchsize=14
model=tools.to_cuda(u_rec.UNet(3,6))
base_lr = 0.0001
optimizer = optim.Adam(model.parameters(), lr=base_lr)
criterion1=tools.to_cuda(torch.nn.CrossEntropyLoss())
criterion2=tools.to_cuda(torch.nn.L1Loss())

iter_ = 0
epochs=100
confusion_matrix = tnt.meter.ConfusionMeter(6)
save_folder = 'models'
os.mkdir(save_folder)
ff=open('./' + save_folder + '/progress.txt','w')

for epoch in range(1,epochs+1):
 change_dataset =  custom.MyDataset(csv_file, image_ids, img_folder, lbl_folder, patch_size)
 model.train()
 train_losses = []
 confusion_matrix.reset()

 for t in tqdm(range(1, len(change_dataset)-batchsize, batchsize)):

  img_batch, lbl_batch = [], []

  for i in range(t, min(t+batchsize, len(change_dataset))):
        img, lbl = change_dataset[i]
        img_batch.append(img)
        lbl_batch.append(lbl)
  img_batch = np.asarray(img_batch)
  lbl_batch = np.asarray(lbl_batch)

  batch_th = torch.from_numpy(img_batch).cuda(1)
  target_th = torch.from_numpy(lbl_batch).cuda(1)
  
  optimizer.zero_grad()
  output, rec=model(batch_th.float())  ##(11,6,256,256)

  output_conf, target_conf = tools.conf_m(output, target_th)
  confusion_matrix.add(output_conf, target_conf)
  loss1=criterion1(output, target_th.long())
  loss2=criterion2(rec, batch_th.float())
  loss= tools.to_cuda(0.9*loss1 + 0.1*loss2)
  loss.backward()
  train_losses.append(loss.item())
  optimizer.step()

  if iter_ % 100 == 0:
   rgb = np.asarray(255 * np.transpose(batch_th.data.cpu().numpy()[0],(1,2,0)), dtype='uint8')
   pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
   gt = target_th.data.cpu().numpy()[0]
   print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss_SEGM: {:.6f}\tLoss_REC: {:.6f}\tAccuracy: {}'.format(
                    epoch, epochs, t, len(change_dataset),100.*t/len(change_dataset), loss1.item(), loss2.item(), tools.accuracy(pred, gt)))
   train_acc=tools.accuracy(pred,gt)

  iter_ += 1
  del(batch_th, target_th, loss)

 print(confusion_matrix.conf)
 train_acc=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf))) *100
 confusion_matrix.reset()

 with torch.no_grad():
  model.eval()
  val_losses = []
  for t in tqdm(range(1, len(change_dataset_val)-batchsize, batchsize)):
   img_batch, lbl_batch = [], []
   for i in range(t, min(t+batchsize, len(change_dataset_val))):
        img, lbl = change_dataset_val[i]
        img_batch.append(img)
        lbl_batch.append(lbl)
   img_batch = np.asarray(img_batch)
   lbl_batch = np.asarray(lbl_batch)

   batch_th = torch.from_numpy(img_batch).cuda(1)
   target_th = torch.from_numpy(lbl_batch).cuda(1)

   output, rec=model(batch_th.float())
   loss1=criterion1(output, target_th.long())
   loss2=criterion2(rec, batch_th.float())
   loss= tools.to_cuda(0.9*loss1 + 0.1*loss2)
   val_losses.append(loss.item())
   output_conf, target_conf = tools.conf_m(output, target_th)
   confusion_matrix.add(output_conf, target_conf)

  print(confusion_matrix.conf)
  testAccuracy=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf))) *100
  print('Train_Accuracy: ', train_acc)
  print('Test_Accuracy:  ', testAccuracy)

  tools.write_results(ff, save_folder, epoch, train_acc, testAccuracy, np.mean(train_losses), np.mean(val_losses))
  if epoch%5==0:
       torch.save(model.state_dict(), './' + save_folder + '/model_{}.pt'.format(epoch))
