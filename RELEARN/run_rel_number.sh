#!/bin/bash

#!/bin/bash

source activate env3.6

dataset=dblp
prefix=different_relation_number
mode=train_new # or train (original model)
a=0.25
b=0.25
c=0.25
d=0.25

for num in `seq 1 1 10`
do
    python3 src/train.py --gpu 5 --dataset $dataset --prefix $prefix --relation $num --mode $mode --use_superv 0 --a $a --b $b --c $c --d $d
done
