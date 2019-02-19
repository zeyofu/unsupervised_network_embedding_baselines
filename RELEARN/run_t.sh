#!/bin/bash

source activate env3.6

dataset=dblp-sub
prefix=different_t
mode=train_new # or train (original model)
a=0.25
b=0.25
c=0.25
d=0.25

for ratio in 0.6 #`seq 0.0 0.2 0.8`
do
    python3 src/train.py --dataset $dataset --prefix $prefix --mode $mode --use_superv 1 --a $a --b $b --c $c --d $d --t $ratio
done
