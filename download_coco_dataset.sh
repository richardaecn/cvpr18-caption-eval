#! /bin/bash

# Download COCO Dataset
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
mv train2014.zip data
mv val2014.zip data
cd data
unzip train2014.zip
unzip val2014.zip
rm train2014.zip
rm val2014.zip

