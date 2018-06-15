import json
import argparse
import time
import os
import numpy as np
import progressbar
import nltk
import string

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
        '--submission', type=str,
        help='Path to the json file that contains all the submissions.')
parser.add_argument(
        '--word-to-idx', type=str, default="./data/word_to_idx.npy",
        help='Path to the npy file that contains mapping from word to index.')
parser.add_argument(
        '--split', type=str, default="./data/karpathysplit/dataset_coco.json",
        help='Path to the JSON file that contains how to split the dataset')
parser.add_argument(
        '--output-path', type=str, default="./data/",
        help='Path to the JSON file that contains how to split the dataset')
parser.add_argument(
        '--num-steps', type=int, default=15,
        help='Length of all captions (default 15).')
parser.add_argument(
        '--name', type=str, default='mysubmission',
        help='Name of the method.')

args = parser.parse_args()
assert args.name != 'human' # Prevent naming conflits

output_path = os.path.join(args.output_path, args.name)
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load splits, default to be Kaparthy's split
split = json.loads(open(args.split).read())
word2idx = np.load(args.word_to_idx).item()

mysubmission = json.load(open(args.submission))

train_data = {}
val_data = {}
test_data = {}
train_filename = []
val_filename = []
test_filename = []

def prep_caption_feat(filenames, data):
    feat = {}
    pbar = progressbar.ProgressBar()
    for i in pbar(range(len(filenames))):
        filename = filenames[i]
        feat[filename] = {}
        gt = data[filename]['human']
        nt = data[filename][args.name]
        feat[filename]['human'] = np.zeros((len(gt), args.num_steps), dtype=np.int32)
        feat[filename][args.name] = np.zeros((args.num_steps), dtype=np.int32)
        for j in range(len(gt)):
            ann = gt[j]
            for k in range(min(len(ann), args.num_steps)):
                if ann[k] in word2idx:
                    feat[filename]['human'][j,k] = word2idx[ann[k]]
                else:
                    feat[filename]['human'][j,k] = word2idx['<unk>']
            if len(ann) < args.num_steps:
                feat[filename]['human'][j,len(ann)] = word2idx['<eos>']

        for j in range(min(args.num_steps, len(nt))):
            if nt[j] in word2idx:
                feat[filename][args.name][j] = word2idx[nt[j]]
            else:
                feat[filename][args.name][j] = word2idx['<unk>']
            if len(nt) < args.num_steps:
                feat[filename][args.name][len(nt)] = word2idx['<eos>']
    return feat

def gen_new_data(filename, data):
    image_idxs = []
    for i in range(len(filename)):
        image_idxs.extend([i] * len(data[filename[i]]['human']))
    captions = np.zeros((len(image_idxs), args.num_steps), dtype=np.int32)
    c = 0
    for i in range(len(filename)):
        caps = data[filename[i]]['human']
        for j in range(caps.shape[0]):
            captions[c, :] = caps[j, :]
            c += 1

    ret = {}
    ret['file_names'] = filename
    ret['image_idxs'] = image_idxs
    ret['word_to_idx'] = word2idx
    ret['captions'] = {}
    ret['captions']['gen'] = captions
    ret['captions']['dis'] = data
    ret['features'] = {}

    return ret

def tokenize(sent):
    tokens = nltk.word_tokenize(sent.lower())
    tokens = [w for w in tokens if w not in string.punctuation]
    return tokens

print("Assign images to split.")
pbar = progressbar.ProgressBar()
for i in pbar(range(len(split['images']))):
    anns = split['images'][i]['sentences']
    filename = split['images'][i]['filename']
    tokens = []
    for j in range(len(anns)):
        human_sent = anns[j]['raw']
        # tokens.append(anns[j]['tokens'])
        tokens.append(tokenize(human_sent))

    sent = mysubmission[filename]
    cap_tokens = tokenize(sent)
    if split['images'][i]['split'] == 'train' or split['images'][i]['split'] == 'restval':
        train_filename.append(filename)
        train_data[filename] = {}
        train_data[filename]['human'] = tokens
        # train_data[filename][args.name] = mysubmission[filename].split()
        train_data[filename][args.name] = cap_tokens
    elif split['images'][i]['split'] == 'val':
        val_filename.append(filename)
        val_data[filename] = {}
        val_data[filename]['human'] = tokens
        # val_data[filename][args.name] = mysubmission[filename].split()
        val_data[filename][args.name] = cap_tokens
    elif split['images'][i]['split'] == 'test':
        test_filename.append(filename)
        test_data[filename] = {}
        test_data[filename]['human'] = tokens
        # test_data[filename][args.name] = mysubmission[filename].split()
        test_data[filename][args.name] = cap_tokens

print("Prepare captions from training set.")
train_data = prep_caption_feat(train_filename, train_data)
data_train = gen_new_data(train_filename, train_data)

print("Prepare captions from validation set.")
val_data   = prep_caption_feat(val_filename,   val_data)
data_val = gen_new_data(val_filename, val_data)

print("Prepare captions from test set.")
test_data  = prep_caption_feat(test_filename,  test_data)
data_test = gen_new_data(test_filename, test_data)

# np.save(os.path.join(output_path, 'train_data.npy'), data_train)
# np.save(os.path.join(output_path, 'val_data.npy'),   data_val)
# np.save(os.path.join(output_path, 'test_data.npy'),  data_test)

print("Saving data...")
np.save(os.path.join(output_path, 'data_train_full.npy'), data_train)
np.save(os.path.join(output_path, 'data_val_full.npy'),   data_val)
np.save(os.path.join(output_path, 'data_test_full.npy'),  data_test)


print("Done.")
