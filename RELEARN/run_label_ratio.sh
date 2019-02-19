#!/bin/bash

source activate env3.6

dataset=dblp
prefix=different_label_ratio
mode=train_new # or train (original model)
a=0.25
b=0.25
c=0.25
d=0.25
t=0.4

for ratio in `seq 0.1 0.1 1.0`
do
    python3 src/train.py --dataset $dataset --prefix $prefix --superv_ratio $ratio --mode $mode --use_superv 1 --a $a --b $b --c $c --d $d --t $t
done
