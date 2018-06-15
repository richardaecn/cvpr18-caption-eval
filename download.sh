#! /bin/bash

#   1. Glove 6B (from offical website, move to ./glove )
wget http://nlp.stanford.edu/data/glove.6B.zip
mkdir glove
mv glove.6B.zip glove
cd glove
unzip glove.6B.zip
rm glove.6B.zip
cd ..

#   1. Download the kaparthy split (dataset_coco.json, move to ./data/karpathysplit/dataset_coco.json)
#   2. MC Sampled Captions (dumped_train.npy, dumped_val.npy, dumped_test.npy, move to ./data)
#   3. preprocessed data for the experiments (data_train_full.npy, data_val_full.npy, data_test_full.npy)
wget https://s3.amazonaws.com/yincui/cvpr18-caption-eval.zip
unzip cvpr18-caption-eval.zip
mkdir -p data
mv cvpr18-caption-eval/* data/
rm -r cvpr18-caption-eval
rm -r cvpr18-caption-eval.zip
rm -r __MACOSX

