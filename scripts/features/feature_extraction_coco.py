import argparse
import time
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import progressbar

from PIL import Image
import numpy as np

from resnet import *

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default=None, required=True,
                    help='data directory (default: None)')
parser.add_argument('--coco-img-dir', type=str, default=None, required=True,
                    help='directory storing the coco images (default: None)')
args = parser.parse_args()

dataset = 'coco'
folder_name = "resnet152"

model = resnet152(pretrained=True)
model = torch.nn.DataParallel(model).cuda()
cudnn.benchmark = True

# Data loading code

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([transforms.ToTensor(), normalize])

model.eval()

# feature extraction
def feature_extraction(data):
    pbar = progressbar.ProgressBar()
    file_names = data['file_names']

    # feature_gen = np.zeros((len(file_names), 14*14, 2048), dtype=np.float16)
    # feature_gen = np.zeros((len(file_names), 7*7, 2048), dtype=np.float16)
    feature_dis = {}

    for i in pbar(xrange(len(file_names))):
        # print((i, len(file_names)))
        file_name = file_names[i]
        if '_train2014_' in file_name:
            path = os.path.join(args.coco_img_dir, 'train2014', file_name)
        elif '_val2014_' in file_name:
            path = os.path.join(args.coco_img_dir, 'val2014', file_name)
        elif '_test2014_' in file_name:
            path = os.path.join(args.coco_img_dir, 'test2014', file_name)
        else:
            raise Exception("Invalid filename : %s"%file_name)

        # print(path)
        img = torch.unsqueeze(transform(Image.open(path).convert('RGB')), 0)
        input_var = torch.autograd.Variable(img, volatile=True)
        # feature = model(input_var).data.cpu().numpy().reshape([14*14, 2048])
        feature = model(input_var).data.cpu().numpy().reshape([7*7, 2048])

        # feature_gen[i, :, :] = feature
        feature_dis[file_name] = np.mean(feature, axis=0)

    # return feature_gen, feature_dis
    return feature_dis


save_path = os.path.join(args.data_dir, folder_name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# train
print('train')
data_train = np.load(os.path.join(args.data_dir, 'data_train_full.npy')).item()
feature_dis_train = feature_extraction(data_train)
# feature_gen_train, feature_dis_train = feature_extraction(data_train)
# print(feature_gen_train.shape)
# np.save(os.path.join(save_path, 'feature_gen_train_full.npy'),
#         feature_gen_train)
np.save(os.path.join(save_path, 'feature_dis_train_full.npy'),
        feature_dis_train)

# val
print('val')
data_val = np.load(os.path.join(args.data_dir, 'data_val_full.npy')).item()
feature_dis_val = feature_extraction(data_val)
# feature_gen_val, feature_dis_val = feature_extraction(data_val)
# print(feature_gen_val.shape)
# np.save(os.path.join(save_path, 'feature_gen_val_full.npy'),
#         feature_gen_val)
np.save(os.path.join(save_path, 'feature_dis_val_full.npy'),
        feature_dis_val)

# test
print('test')
data_test = np.load(os.path.join(args.data_dir, 'data_test_full.npy')).item()
feature_dis_test = feature_extraction(data_test)
# feature_gen_test, feature_dis_test = feature_extraction(data_test)
# print(feature_gen_test.shape)
# np.save(os.path.join(save_path, 'feature_gen_test_full.npy'),
#         feature_gen_test)
np.save(os.path.join(save_path, 'feature_dis_test_full.npy'),
        feature_dis_test)
