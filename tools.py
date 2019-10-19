import os
import torch
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = 1
def to_cuda(v):
    if USE_CUDA:
        return v.cuda(DEVICE)
    return v

def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

def conf_m(output, target_th):
  output_conf=(output.data.squeeze()).transpose(1,3)
  output_conf=output_conf.transpose(1,2)
  output_conf=(output_conf.contiguous()).view(output_conf.size(0)*output_conf.size(1)*output_conf.size(2), output_conf.size(3))
  target_conf=target_th.data.squeeze()
  target_conf=(target_conf.contiguous()).view(target_conf.size(0)*target_conf.size(1)*target_conf.size(2))

  return output_conf, target_conf

def write_results(ff, save_folder, epoch, train_acc, test_acc, train_losses, val_losses):
    ff=open('./' + save_folder + '/progress.txt','a')
    ff.write('train_OA: ')
    ff.write(str('%.3f' % train_acc))
    ff.write(' ')
    ff.write(' val_OA: ')
    ff.write(str('%.3f' % test_acc))
    ff.write('  ')
    ff.write(' E: ')
    ff.write(str(epoch))
    ff.write('  ')
    ff.write(' TRAIN_LOSS: ')
    ff.write(str('%.3f' % train_losses))
    ff.write(' VAL_LOSS: ')
    ff.write(str('%.3f' % val_losses))
    ff.write('\n')
