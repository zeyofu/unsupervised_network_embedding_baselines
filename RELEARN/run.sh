#!/bin/bash

source activate env3.6

dataset=dblp-sub
a=0.25
b=0.25
c=0.25
d=0.25
t=0.4       # for supervised
rel_num=2   # for unsupervised

for i in 1
do
    # run full RELEARN
    python3 src/train.py --dataset $dataset --prefix full_model --a $a --b $b --c $c --d $d --t $t --use_superv 1 --mode train
    # run RELEARN-SUP
#python3 src/train.py --dataset $dataset --save_every 100 --prefix no_superv --a $a --b $b --c $c --d $d --relation $rel_num --use_superv 0 --mode train_new
    # run RELEARN-VAE
#python3 src/train.py --dataset $dataset --save_every 100 --prefix no_vae --a $a --b $b --c $c --d $d --mode no_vi
    # run RELEARN-DIFF
#    python3 src/train.py --dataset $dataset --save_every 100 --prefix no_diff --a $a --b $b --c 0 --d 0 --mode no_vi
done
