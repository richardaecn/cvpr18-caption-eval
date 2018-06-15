import json
import argparse
import time
import os
import numpy as np
import progressbar

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
        '--split', type=str, default="./data/karpathysplit/dataset_coco.json",
        help='Path to the JSON file that contains how to split the dataset')
parser.add_argument(
        '--output-path', type=str, default="./data",
        help='Path to the JSON file that contains how to split the dataset')

args = parser.parse_args()

# Load splits, default to be Kaparthy's split
split = json.loads(open(args.split).read())
coco_dict = {}
token_len = []
pbar = progressbar.ProgressBar()
for i in pbar(range(len(split['images']))):
    if split['images'][i]['split'] == 'train' or split['images'][i]['split'] == 'restval':
        anns = split['images'][i]['sentences']
        for j in range(len(anns)):
            tokens = anns[j]['tokens']
            token_len.append(len(tokens))
            for k in range(len(tokens)):
                token = tokens[k].lower()
                if token in coco_dict:
                    coco_dict[token] += 1
                else:
                    coco_dict[token] = 1

import operator
sorted_dict = sorted(coco_dict.items(), key=operator.itemgetter(1))
sorted_dict = sorted_dict[::-1]

coco_word_10k = []
coco_word_10k.append('<pad>')
coco_word_10k.append('<unk>')
coco_word_10k.append('<sos>')
coco_word_10k.append('<eos>')
for i in range(10000):
    coco_word_10k.append(sorted_dict[i][0])
print("Vocab length:%d"%len(coco_word_10k))

word_to_idx = {}
for i in range(len(coco_word_10k)):
    word_to_idx[coco_word_10k[i]] = i

np.save(os.path.join(args.output_path, 'word_to_idx.npy'),  word_to_idx)

# Prepare Glove Word Embedding
idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
def load_glove(filename):
    ret = {}
    with open(filename) as f:
        for l in f:
            lst = l.split()
            key = lst[0]
            value = np.array(lst[1:]).astype(np.float)
            ret[key] = value
    return ret

D50  = load_glove('./glove/glove.6B.50d.txt')
D100 = load_glove('./glove/glove.6B.100d.txt')
D200 = load_glove('./glove/glove.6B.200d.txt')
D300 = load_glove('./glove/glove.6B.300d.txt')


def D_to_W(D, dim):
    W = np.zeros((len(idx_to_word), dim), dtype=np.float32)
    for i in xrange(len(idx_to_word)):
        word = idx_to_word[i]
        if word in D:
            W[i, :] = D[word]
        else:
            W[i, :] = D['unk']
    return W

W50 = D_to_W(D50, 50)
W100 = D_to_W(D100, 100)
W200 = D_to_W(D200, 200)
W300 = D_to_W(D300, 300)

np.save(os.path.join(args.output_path, 'word_embedding_50.npy' ), W50)
np.save(os.path.join(args.output_path, 'word_embedding_100.npy'), W100)
np.save(os.path.join(args.output_path, 'word_embedding_200.npy'), W200)
np.save(os.path.join(args.output_path, 'word_embedding_300.npy'), W300)

