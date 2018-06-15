#! /bin/bash

mkdir -p checkpoints/imagenet

# wget https://download.pytorch.org/models/resnet18-5c106cde.pth;
# mv resnet18-5c106cde.pth checkpoints/imagenet/resnet18.pth

# wget https://download.pytorch.org/models/resnet34-333f7ec4.pth;
# mv resnet34-333f7ec4.pth checkpoints/imagenet/resnet34.pth;

# wget https://download.pytorch.org/models/resnet50-19c8e357.pth;
# mv resnet50-19c8e357.pth checkpoints/imagenet/resnet50.pth;

# wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth;
# mv resnet101-5d3b4d8f.pth checkpoints/imagenet/resnet101.pth;

wget https://download.pytorch.org/models/resnet152-b121ed2d.pth;
mv resnet152-b121ed2d.pth checkpoints/imagenet/resnet152.pth
