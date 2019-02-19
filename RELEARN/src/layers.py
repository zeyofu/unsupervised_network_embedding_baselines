"""
Added Fully Connected Layer.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


def MLPLayer(input, output, hidden_layer=1):
    if hidden_layer==1:
        return torch.nn.Sequential(torch.nn.Linear(input, input//2),
                                   torch.nn.ReLU(inplace=True),
                                   torch.nn.Linear(input//2, output))
    elif hidden_layer==2:
        return torch.nn.Sequential(torch.nn.Linear(input, input // 2),
                                   torch.nn.ReLU(inplace=True),
                                   torch.nn.Linear(input // 2, input // 2),
                                   torch.nn.ReLU(inplace=True),
                                   torch.nn.Linear(input // 2, output))


class DecodeLink(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, embed_dim):
        super(DecodeLink, self).__init__()
        self.embed_dim = embed_dim
        self.transform = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=False), nn.Tanh())

    def forward(self, input):
        l, r = input.split(self.embed_dim, dim=1)
        output = (self.transform(l)*self.transform(r)).sum(dim=1, keepdim=True)
        return output


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # print(adj.double(), support.double())
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class FullyConnectedLayer(Module):
    """
    Simple FC layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(FullyConnectedLayer, self).__init__()
        self.in_features = in_features
        self.out_features = int(out_features/2)
        self.weight = Parameter(torch.FloatTensor(in_features, int(out_features/2)))
        if bias:
            self.bias = Parameter(torch.FloatTensor(int(out_features/2)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.mm(input, self.weight.double())
        if self.bias is not None:
            support += self.bias.double()
        splitted_outputs = torch.split(support,int(support.shape[0]/2))
        # print(splitted_outputs[0].shape, splitted_outputs[1].shape, support.shape)
        output = torch.cat((splitted_outputs[0],splitted_outputs[1]), dim = 1)
        # print("final layer output shape",output.shape)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def _sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    dims = logits.dim()
    # gumbel_noise = Variable(_sample_gumbel(logits.size(), eps=eps, out=logits.data.new()))
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.data.new())
    y = logits + gumbel_noise
    return F.softmax(y / tau, dims - 1)


def gumbel_softmax(logits, tau=0.8, hard=False, eps=1e-10):
    """
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: `[batch_size, n_class]` unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if ``True``, take `argmax`, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = _gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        _, k = y_soft.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros_like(logits).scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y
