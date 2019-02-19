# RELEARN
---------------

Code for the paper "...." KDD 2019

### Required Inputs
---------------


### Dependencies
---------------

This project is based on ```python>=3.6``` and ```pytorch>=0.4```. The dependent package for this project is listed as below:
```
gensim
skopt
```

### Command
---------------

To train an RELEARN model with default setting, please run
```
./python3 src/train.py --mode train
```

To search a optimal hyper-parameter (weight for different loss) for RELEARN model, please run
```
./python3 src/train.py --mode search --budget 50
```

You can specify the parameters in the bash files. The variables names are self-explained.


### Key Parameters
---------------

**budget**: how many calls in greedy search for hyper-parameter.

**sample_mode**: there are 4 type of loss, each one correspond to one component of sample mode. **n** is node feature reconstruct loss,  **l** is link prediction loss, **dc** is diffusion content reconstruction loss, **ds** is diffusion structure(link) prediction loss.

**diffusion_threshold**: filter out diffusion which contain nodes less than this threshold.

**neighbor_sample_size**: how many neighbor to aggregate in GCN layer.

**sample_size**: how many data to be used in one epoch for each sample mode. Note that for the two link prediction loss, sample size is the sum of positive sample size and negative sample size.

**negative_sample_size**: it is negative sample / positive sample.

**sample_embed**: the dimension of hidden state, also the dimension of learned embedding.

**use_vi**: whether to use variational inference.

**relation**: number of relations to be used in variational inference.

**use_superv**: whether to add supervision in trainig.

**superv_ratio**: how many supervision to add, used in label efficiency experiments.

**a, b, c, d**: weights for different loss, main hyper-parameter to tune in practice.

## Citation
---------------

Please cite the following two papers if you are using our tool. Thanks!

- Jingbo Shang*, Liyuan Liu*, Xiaotao Gu, Xiang Ren, Teng Ren and Jiawei Han, "**[Learning Named Entity Tagger using Domain-Specific Dictionary](https://arxiv.org/abs/1809.03599)**", in Proc. of 2018 Conf. on Empirical Methods in Natural Language Processing (EMNLP'18), Brussels, Belgium, Oct. 2018. (* Equal Contribution)

```

@article{shang2018automated,
  title = {Automated phrase mining from massive text corpora},
  author = {Shang, Jingbo and Liu, Jialu and Jiang, Meng and Ren, Xiang and Voss, Clare R and Han, Jiawei},
  journal = {IEEE Transactions on Knowledge and Data Engineering},
  year = {2018},
  publisher = {IEEE}
}
```
