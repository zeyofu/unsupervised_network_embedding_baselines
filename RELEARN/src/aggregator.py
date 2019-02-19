from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np
import gensim

import torch
from torch.utils.data import DataLoader

sys.path.append('./')

from src.utils import construct_feature, load_data
from src.models import MLP
from src.dataset import EvaDataset


class Dataset:
    def __init__(self, args, dataset):
        self.adj, self.features, self.graph, self.old_adj, self.tuples, self.nonzero_nodes, self.diffusion \
            = load_data(f'data/{dataset}/', dataset, args.diffusion_threshold)

        self.num_node, self.feature_len = self.features.shape
        self.neighbor_sample_size = args.neighbor_sample_size

    def sample_subgraph(self, selection):
        """
            1. Selecting edges according to the selection indexs.
            2. For each node, sampling their neighbor nodes.
            3. Build a partial adjacency matrix and neigbor feature vector.
        """

        if isinstance(selection, list):
            if isinstance(selection[0], list):
                final_l = [i for l in selection for i in l]
            else:
                final_l = selection
        elif isinstance(selection, np.ndarray):
            final_l = selection.flatten()
        else:
            exit('unknown type for selection')

        sampled_neighbors = []
        col_dim = 0
        # dim_check = 0
        final_input_features = torch.tensor([])
        for idx in final_l:
            if idx not in self.graph:
                sampled_neighbors.append([idx])
                if final_input_features.shape[0] == 0 :
                    final_input_features = self.features[idx].view(1,-1)
                else: final_input_features = torch.cat((final_input_features, self.features[idx].view(1,-1)))
                col_dim+=1
                # dim_check+=1
            else:
                if len(self.graph[idx]) <= self.neighbor_sample_size:
                    sampled_neighbors.append([idx] + self.graph[idx])
                    if final_input_features.shape[0] == 0 :
                        final_input_features = self.features[idx].view(1,-1)
                    else:
                        final_input_features = torch.cat((final_input_features, self.features[idx].view(1,-1)))
                    final_input_features = torch.cat((final_input_features, self.features[self.graph[idx]]))
                    col_dim+= (1+self.features[self.graph[idx]].shape[0])
                    # dim_check+=(1+self.features[self.graph[idx]].shape[0])
                else:
                    idx_for_sample = np.random.randint(0,len(self.graph[idx]), self.neighbor_sample_size)
                    sample_set = [self.graph[idx][x_j] for x_j in range(len(self.graph[idx])) if x_j in set(idx_for_sample)]
                    sampled_neighbors.append([idx] + sample_set)
                    if final_input_features.shape[0] == 0 :
                        final_input_features = self.features[idx].view(1,-1)
                    else: final_input_features = torch.cat((final_input_features, self.features[idx].view(1,-1)))
                    final_input_features = torch.cat((final_input_features, self.features[sample_set]))
                    col_dim+= (1+self.features[sample_set].shape[0])
                    # dim_check+=(1+self.features[sample_set].shape[0])
        sampled_adj = np.zeros((len(final_l), col_dim))
        col_idx = 0
        for row_idx in range(len(final_l)):
            for real_col_idx in sampled_neighbors[row_idx]:
                sampled_adj[row_idx, col_idx] = self.old_adj[final_l[row_idx], real_col_idx]
                col_idx += 1

        # sampled_labels = torch.cat((final_features_1, final_features_2), dim = 1)
        # return final_input_features, sampled_adj, sampled_labels, torch.nonzero(torch.sum(sampled_labels, 1)).shape[0]

        return final_input_features, torch.from_numpy(sampled_adj).float()


def parse_args():
    # general settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dblp',
                        help='dataset name.')
    parser.add_argument('--eval_file', type=str, default='data/dblp/eval/rel1.txt',
                        help='evaluation file path.')
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use")
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')


    # sample settings
    parser.add_argument('--neighbor_sample_size', default=30, type=int,
                        help='sample size for neighbor to be used in gcn')
    parser.add_argument('--sample_embed', default=100, type=int,
                        help='sample size for embedding generation')
    parser.add_argument('--repeat', default=5, type=int,
                        help='repeat times')


    # evluating settings
    parser.add_argument('--epochs_eval', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--lr_eval', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--batch_size_eval', type=float, default=10,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_eval', type=int, default=100,
                        help='Number of hidden units.')

    return parser.parse_args()


def evaluate(args, embedding, repeat_times=3):
    best_train_accs, best_test_accs = [], []
    best_train_acc_epochs, best_test_acc_epochs = [], []

    train = construct_feature(args.train_data, embedding)
    test = construct_feature(args.test_data, embedding)

    for i in range(repeat_times):
        np.random.shuffle(train)

        X_train, y_train = torch.FloatTensor(train[:, :-1]), torch.LongTensor(train[:, -1])
        X_test, y_test = torch.FloatTensor(test[:, :-1]), torch.LongTensor(test[:, -1])
        dataloader = DataLoader(EvaDataset(X_train, y_train), batch_size=args.batch_size_eval, shuffle=True)
        X_train = X_train.to(args.device)
        X_test = X_test.to(args.device)
        y_train = y_train.to(args.device)
        y_test = y_test.to(args.device)

        kwargs = {
            'input_dim': X_train.size(1),
            'hidden_dim': args.hidden_eval,
            'output_dim': args.output_dim
        }
        model = MLP(**kwargs).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_eval)

        best_test_acc, best_train_acc = 0, 0
        best_test_acc_epoch, best_train_acc_epoch = 0, 0
        for epoch in range(args.epochs_eval):
            for i, (batch, label) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = model(batch.to(args.device), label.to(args.device))
                loss.backward()
                optimizer.step()

            preds, test_acc = model.predict(X_test, y_test)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_epoch = epoch + 1
                best_pred = preds

            _, train_acc = model.predict(X_train, y_train)
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_acc_epoch = epoch + 1

            print('\repoch {}/{} train acc={}, test acc={}, best train acc={} @epoch:{}, best test acc={} @epoch:{}'.
                  format(epoch + 1, args.epochs_eval, train_acc, test_acc, best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch), end='')
            sys.stdout.flush()

        print('')
        best_train_accs.append(best_train_acc)
        best_test_accs.append(best_test_acc)
        best_train_acc_epochs.append(best_train_acc_epoch)
        best_test_acc_epochs.append(best_test_acc_epoch)

    best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch = \
        np.mean(best_train_accs), np.mean(best_train_acc_epochs), np.mean(best_test_accs), np.mean(
            best_test_acc_epochs)
    std = np.std(best_test_accs)
    print('{}: best train acc={} @epoch:{}, best test acc={} += {} @epoch:{}'.
          format(args.eval_file, best_train_acc, best_train_acc_epoch, best_test_acc, std, best_test_acc_epoch))

    return best_train_acc, best_test_acc, std


def aggregate(features, adj):
    features = features.cuda()
    adj = adj.cuda()
    output = torch.spmm(adj, features).data.cpu().numpy()
    return output


def main(args, Data):
    test_accs = []
    for _ in range(args.repeat):
        learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(Data.feature_len)
        for i in range(0, len(args.nodes), args.sample_embed):
            nodes = args.nodes[i:i+args.sample_embed]
            features, adj = Data.sample_subgraph(nodes)
            embedding = aggregate(features, adj)
            learned_embed.add([str(node) for node in nodes], embedding)
        train_acc, test_acc, std = evaluate(args, learned_embed)
        test_accs.append(test_acc)

    test_accs = np.array(test_accs)
    print(f'final results: test acc={test_accs.mean()}, std={test_accs.std()}')


if __name__ == '__main__':
    # Initialize args and seed
    args = parse_args()
    print('Number CUDA Devices:', torch.cuda.device_count())
    # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    torch.cuda.device(args.gpu)
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu > -1 else "cpu")
    print('Active CUDA Device: GPU', torch.cuda.current_device())

    # Load data
    args.diffusion_threshold = 10000
    Data = Dataset(args, args.dataset)

    labels, labeled_data = set(), []
    nodes = set()
    with open(args.eval_file, 'r') as lf:
        for line in lf:
            if line.rstrip() == 'test':
                # finish load training data
                train_data = labeled_data
                labeled_data = []
                continue
            line = line.rstrip().split('\t')
            data1, data2, label = line[0], line[1], int(line[2])
            labeled_data.append((data1, data2, label))
            labels.add(label)
            nodes.update([int(data1), int(data2)])
    args.nodes = list(nodes)
    args.train_data = train_data
    args.test_data = labeled_data
    args.output_dim = len(labels)

    main(args, Data)
