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
from torch.utils.data import DataLoader

csv_file = './xys/myxys_train.csv'
csv_file_val = './xys/myxys_val.csv'
image_ids = [11, 13, 1, 21, 23, 26, 28, 30, 32, 34, 37, 3, 5, 7] #train areas
infer_ids = [15,17] #validation areas
img_folder = '/home/mariapap/DATA/top/' #folder with tif images of Vaihingen city
lbl_folder = '/home/mariapap/DATA/GROUNDTRUTH/ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE_1D/' #folder with 2D groundtruth images, category indices:0,1,2,3,4,5
patch_size = 256
train_dataset =  custom.MyDataset(csv_file, image_ids, img_folder, lbl_folder, patch_size)
val_dataset = custom.MyDataset(csv_file_val, infer_ids, img_folder, lbl_folder, patch_size)

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
    mydataset = DataLoader(train_dataset, batch_size=14, shuffle=True)
    model.train()
    train_losses = []
    confusion_matrix.reset()

    for i, batch, in enumerate(tqdm(mydataset)):
        img_batch, lbl_batch = batch
        img_batch, lbl_batch = tools.to_cuda(img_batch), tools.to_cuda(lbl_batch)

        optimizer.zero_grad()
        output, rec=model(img_batch.float())  ##(11,6,256,256)

        output_conf, target_conf = tools.conf_m(output, lbl_batch)
        confusion_matrix.add(output_conf, target_conf)
        loss1=criterion1(output, lbl_batch.long())
        loss2=criterion2(rec, img_batch.float())
        loss= tools.to_cuda(0.9*loss1 + 0.1*loss2)
        loss.backward()
        train_losses.append(loss.item())
        optimizer.step()

        if iter_ % 100 == 0:
            pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
            gt = lbl_batch.data.cpu().numpy()[0]
            print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss_SEGM: {:.6f}\tLoss_REC: {:.6f}\tAccuracy: {}'.format(
                    epoch, epochs, i, len(mydataset),100.*i/len(mydataset), loss1.item(), loss2.item(), tools.accuracy(pred, gt)))
            train_acc=tools.accuracy(pred,gt)

        iter_ += 1
        del(img_batch, lbl_batch, loss)

    train_acc=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf))) *100
    print('TRAIN_LOSS: ', '%.3f' % np.mean(train_losses), 'TRAIN_ACC: ', '%.3f' % train_acc)
    confusion_matrix.reset()


    with torch.no_grad():
        model.eval()
        mydataset_val = DataLoader(val_dataset, batch_size=14, shuffle=True)
        val_losses = []

        for i, batch, in enumerate(tqdm(mydataset_val)):
            img_batch, lbl_batch = batch
            img_batch, lbl_batch = tools.to_cuda(img_batch), tools.to_cuda(lbl_batch)

            output, rec=model(img_batch.float())
            loss1=criterion1(output, lbl_batch.long())
            loss2=criterion2(rec, img_batch.float())
            loss= tools.to_cuda(0.9*loss1 + 0.1*loss2)
            val_losses.append(loss.item())
            output_conf, target_conf = tools.conf_m(output, lbl_batch)
            confusion_matrix.add(output_conf, target_conf)

        print(confusion_matrix.conf)
        testAccuracy=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf))) *100
        print('VAL_LOSS: ', '%.3f' % np.mean(val_losses), 'VAL_ACC: ', '%.3f' % testAccuracy)
        tools.write_results(ff, save_folder, epoch, train_acc, testAccuracy, np.mean(train_losses), np.mean(val_losses))
        if epoch%5==0:
            torch.save(model.state_dict(), './' + save_folder + '/model_{}.pt'.format(epoch))
