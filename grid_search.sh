#!/bin/bash

eval "$(conda shell.bash hook)"

"""
grid search over 
--batch_size: Batch size during training
--nf: Size of feature maps in encoder & decoder
"""

max_epochs=10
gpus=2

for batch_size in 32 64 128 256
do  
    for nf in 64 128 256
    do  
        echo "Training Autoencoder with --batch_size=$batch_size --nfe=$nf --nfd=$nf --max_epochs=$max_epochs --gpus=$gpus"
        CUDA_VISIBLE_DEVICES=0,1 python autoencoder.py --batch_size=$batch_size --nfe=$nf --nfd=$nf --max_epochs=$max_epochs --gpus=$gpus
    done
done