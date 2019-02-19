from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np
from datetime import datetime
from random import shuffle
from tqdm import trange
import time
import argparse
import pickle, json
import gensim

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('./')

from src.utils import print_config, save_checkpoint, save_embedding, construct_feature
from src.models import GCNDecoder, GCNDecoder2, GCNDecoder3, MLP
from src.dataset import Dataset, EvaDataset
from src.logger import myLogger


def parse_args():
    # general settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dblp-sub',
                        help='dataset name. [dblp, dblp-sub]')
    parser.add_argument('--eval_file', type=str, default='',
                        help='evaluation file path.')
    parser.add_argument("--load_model", type=str, default=False,
                        help="whether to load model")
    parser.add_argument("--mode", type=str, default='train',
                        help="train or evaluate or search or parallel_search [search, train, part, search_new, train_new, no_vi]")
    parser.add_argument("--budget", type=int, default=50,
                        help="budget for greedy search")
    parser.add_argument("--gpu", type=int, default=0,
                        help="which GPU to use")
    parser.add_argument('--log_level', default=20,
                        help='logger level.')
    parser.add_argument("--prefix", type=str, default='',
                        help="prefix use as addition directory")
    parser.add_argument('--suffix', default='', type=str,
                        help='suffix append to log dir')
    parser.add_argument('--log_every', type=int, default=200,
                        help='log results every epoch.')
    parser.add_argument('--save_every', type=int, default=200,
                        help='save learned embedding every epoch.')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')


    # sample settings
    parser.add_argument("--sample_mode", type=str, default='n|l|dc|ds',
                        help="n:node, l:link, dc:diffusion content, ds:diffusion structure")
    parser.add_argument('--diffusion_threshold', default=10, type=int,
                        help='threshold for diffusion')
    parser.add_argument('--neighbor_sample_size', default=30, type=int,
                        help='sample size for neighbor to be used in gcn')
    parser.add_argument('--sample_size', default=200, type=int,
                        help='sample size for training data')
    parser.add_argument('--negative_sample_size', default=1, type=int,
                        help='negative sample / positive sample')
    parser.add_argument('--sample_embed', default=100, type=int,
                        help='sample size for embedding generation')


    # training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=100,
                        help='Number of hidden units, also the dimension of node representation after GCN.')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--relation', type=int, default=2,
                        help='Number of relations.')
    parser.add_argument('--rel_dim', type=int, default=300,
                        help='dimension of relation embedding.')
    parser.add_argument('--hard_gumbel', type=int, default=0,
                        help='whether to make gumbel one hot.')
    parser.add_argument('--use_superv', type=int, default=1,
                        help='whether to add supervision(prior) to variational inference')
    parser.add_argument('--superv_ratio', type=float, default=0.8,
                        help='how many data to use in training(<=0.8), default is 80%, note that 20% are used as test')
    parser.add_argument('--t', type=float, default=0.4,
                        help='ratio of supervision in sampled data')
    parser.add_argument('--a', type=float, default=0.25,
                        help='weight for node feature loss')
    parser.add_argument('--b', type=float, default=0.25,
                        help='weight for link loss')
    parser.add_argument('--c', type=float, default=0.25,
                        help='weight for diffusion loss')
    parser.add_argument('--d', type=float, default=0.25,
                        help='weight for diffusion content loss')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='temperature for gumbel softmax')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='whether to use early stop')
    parser.add_argument('--patience', type=int, default=500,
                        help='used for early stop')

    # evluating settings
    parser.add_argument('--epochs_eval', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr_eval', type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--batch_size_eval', type=float, default=10,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_eval', type=int, default=100,
                        help='Number of hidden units.')
    parser.add_argument('-patience_eval', type=int, default=500,
                        help='used for early stop in evaluation')

    return parser.parse_args()


def evaluate(args, embedding, logger, repeat_times=5):
    best_train_accs, best_test_accs = [], []
    best_train_acc_epochs, best_test_acc_epochs = [], []
    if args.use_superv:
        train = construct_feature(args.train, embedding)
        test = construct_feature(args.test, embedding)
    else:
        data = construct_feature(args.label_data, embedding)
        split = int(len(args.label_data) / repeat_times)

    for i in range(repeat_times):
        if not args.use_superv:
            p1, p2 = i*split, (i+1)*split
            test = data[p1:p2, :]
            train1, train2 = data[:p1, :], data[p2:, :]
            train = np.concatenate([train1, train2])

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
        count = 0
        for epoch in range(args.epochs_eval):
            for i, (batch, label) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = model(batch.to(args.device), label.to(args.device))
                loss.backward()
                optimizer.step()

            preds, test_acc = model.predict(X_test, y_test)
            test_acc *= 100
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_epoch = epoch + 1
                best_pred = preds
                count = 0
            else:
                count += 1
                if count >= args.patience_eval:
                    break

            _, train_acc = model.predict(X_train, y_train)
            train_acc *= 100
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_acc_epoch = epoch + 1

            if args.mode != 'parallel_search':
                print('\repoch {}/{} train acc={:.4f}, test acc={:.4f}, best train acc={:.4f} @epoch:{:d}, best test acc={:.4f} @epoch:{:d}'.
                      format(epoch + 1, args.epochs_eval, train_acc, test_acc, best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch), end='')
                sys.stdout.flush()

        if args.mode != 'parallel_search': print('')
        best_train_accs.append(best_train_acc)
        best_test_accs.append(best_test_acc)
        best_train_acc_epochs.append(best_train_acc_epoch)
        best_test_acc_epochs.append(best_test_acc_epoch)

    best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch = \
        np.mean(best_train_accs), np.mean(best_train_acc_epochs), np.mean(best_test_accs), np.mean(best_test_acc_epochs)
    std = np.std(best_test_accs)
    logger.info('{}: best train acc={:.2f} @epoch:{:d}, best test acc={:.2f} += {:.2f} @epoch:{:d}'.
                format(args.eval_file, best_train_acc, int(best_train_acc_epoch), best_test_acc, std, int(best_test_acc_epoch)))

    return best_train_acc, best_test_acc, std



def evaluate_rel(args, data, logger, repeat_times=5):
    best_train_accs, best_test_accs = [], []
    best_train_acc_epochs, best_test_acc_epochs = [], []

    split = int(len(args.label_data) / repeat_times)


    for i in range(repeat_times):
        p1, p2 = i*split, (i+1)*split
        test = data[p1:p2, :]
        train1, train2 = data[:p1, :], data[p2:, :]
        train = np.concatenate([train1, train2])

        X_train, y_train = torch.FloatTensor(train[:, :-1]), torch.LongTensor(train[:, -1])
        X_test, y_test = torch.FloatTensor(test[:, :-1]), torch.LongTensor(test[:, -1])
        dataloader = DataLoader(EvaDataset(X_train, y_train), batch_size=args.batch_size_eval, shuffle=True)
        X_train = X_train.to(args.device)
        X_test = X_test.to(args.device)
        y_train = y_train.to(args.device)
        y_test = y_test.to(args.device)

        kwargs = {
            'input_dim': X_train.size(1),
            'hidden_dim': 1000,
            'output_dim': args.output_dim,
            'layer':2
        }
        model = MLP(**kwargs).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_eval)
        best_test_acc, best_train_acc = 0, 0
        best_test_acc_epoch, best_train_acc_epoch = 0, 0
        count = 0
        for epoch in range(args.epochs_eval):
            for i, (batch, label) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = model(batch.to(args.device), label.to(args.device))
                loss.backward()
                optimizer.step()

            preds, test_acc = model.predict(X_test, y_test)
            test_acc *= 100
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_epoch = epoch + 1
                best_pred = preds
                count = 0
            else:
                count += 1
                if count >= args.patience_eval:
                    break

            _, train_acc = model.predict(X_train, y_train)
            train_acc *= 100
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_acc_epoch = epoch + 1

            if args.mode != 'parallel_search':
                print('\repoch {}/{} train acc={:.4f}, test acc={:.4f}, best train acc={:.4f} @epoch:{:d}, best test acc={:.4f} @epoch:{:d}'.
                      format(epoch + 1, args.epochs_eval, train_acc, test_acc, best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch), end='')
                sys.stdout.flush()

        if args.mode != 'parallel_search': print('')
        best_train_accs.append(best_train_acc)
        best_test_accs.append(best_test_acc)
        best_train_acc_epochs.append(best_train_acc_epoch)
        best_test_acc_epochs.append(best_test_acc_epoch)

    best_train_acc, best_train_acc_epoch, best_test_acc, best_test_acc_epoch = \
        np.mean(best_train_accs), np.mean(best_train_acc_epochs), np.mean(best_test_accs), np.mean(best_test_acc_epochs)
    std = np.std(best_test_accs)
    logger.info('{}: best train acc={:.2f} @epoch:{:d}, best test acc={:.2f} += {:.2f} @epoch:{:d}'.
                format(args.eval_file, best_train_acc, int(best_train_acc_epoch), best_test_acc, std, int(best_test_acc_epoch)))

    return best_train_acc, best_test_acc, std



def train(args, model, Data, log_dir, logger, writer=None, optimizer=None):
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.modes = []
    if 'n' in args.sample_mode and model.a > 0:
        args.modes.append('node')
    if 'l' in args.sample_mode and model.b > 0:
        args.modes.append('link')
    if 'dc' in args.sample_mode and model.d > 0:
        args.modes.append('diffusion_content')
    if 'ds' in args.sample_mode and model.c > 0:
        args.modes.append('diffusion_structure')

    t = time.time()
    best_acc, best_epoch, best_std = 0, 0, 0
    count = 0
    model.train()

    for epoch in range(1, args.epochs+1):
        losses = []
        for mode in args.modes:
            optimizer.zero_grad()
            (sampled_features, sampled_adj, prior), sampled_labels = Data.sample(mode)
            loss = model(sampled_features, sampled_adj, sampled_labels, mode, prior)
            loss.backward()
            optimizer.step()

            if epoch % args.log_every == 0:
                losses.append(loss.item())

        if epoch % args.log_every == 0:
            duration = time.time() - t
            msg = 'Epoch: {:04d} '.format(epoch)
            for mode, loss in zip(args.modes, losses):
                msg += 'loss_{}: {:.4f}\t'.format(mode, loss)
                if writer is not None:
                    writer.add_scalar('data/{}_loss'.format(mode), loss, epoch)
            logger.info(msg+' time: {:d}s '.format(int(duration)))

        if epoch % args.save_every == 0:
            model.eval()
            shuffle(args.label_data)
            node_pair = np.array([[int(n1), int(n2)] for (n1, n2, _) in args.label_data])
            labels = np.array([label for (_, _, label) in args.label_data])
            sampled_features, sampled_adj, _ = Data.sample_subgraph(node_pair, generate_prior=False, return_nodes=False)
            g_ij = model.forward_encoder(sampled_features.to(args.device), sampled_adj.to(args.device))
            _, h_ij = model.forward_decoder(g_ij)
            embedding = h_ij.data.cpu().numpy()
            data = np.concatenate((embedding, labels.reshape(-1,1)), axis=1)
            model.train()

            train_acc, test_acc, std = evaluate_rel(args, data, logger)


            # learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(model.nembed)
            # for i in range(0, len(args.nodes), args.sample_embed):
            #     nodes = args.nodes[i:i+args.sample_embed]
            #     features, adj, _ = Data.sample_subgraph(nodes, False)
            #     embedding = model.generate_embedding(features, adj)
            #     learned_embed.add([str(node) for node in nodes], embedding)
            # train_acc, test_acc, std = evaluate(args, learned_embed, logger)
            duration = time.time() - t
            logger.info('Epoch: {:04d} '.format(epoch)+
                        'train_acc: {:.2f} '.format(train_acc)+
                        'test_acc: {:.2f} '.format(test_acc)+
                        'std: {:.2f} '.format(std)+
                        'time: {:d}s'.format(int(duration)))
            if writer is not  None:
                writer.add_scalar('data/test_acc', test_acc, epoch)
            if test_acc > best_acc:
                best_acc = test_acc
                best_std = std
                best_epoch = epoch
                # save_embedding(learned_embed, os.path.join(log_dir, 'embedding.bin'))
                save_checkpoint({
                    'args': args,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, log_dir,
                    f'epoch{epoch}_time{int(duration):d}_trainacc{train_acc:.2f}_testacc{test_acc:.2f}_std{std:.2f}.pth.tar', logger, True)
                count = 0
            else:
                if args.early_stop:
                    count += args.save_every
                if count >= args.patience:
                    logger.info('early stopped!')
                    break

    logger.info(f'best test acc={best_acc:.2f} +- {best_std:.2f} @ epoch:{int(best_epoch):d}')
    return best_acc


def train_new(args, embedding, Data, log_dir, logger, writer=None):
    # Model and optimizer
    model = GCNDecoder2(device=args.device,
                        embedding=embedding,
                       nfeat=args.feature_len,
                       nhid=args.hidden,
                       ncont=args.content_len,
                       nrel=int(args.relation), rel_dim=args.rel_dim,
                       dropout=args.dropout,
                       a=float(args.a), b=float(args.b), c=float(args.c), d=float(args.d),
                       tau=args.tau, hard_gumbel=args.hard_gumbel)
    model.to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.modes = []
    if 'n' in args.sample_mode and model.a > 0:
        args.modes.append('node')
    if 'l' in args.sample_mode and model.b > 0:
        args.modes.append('link')
    if 'dc' in args.sample_mode and model.d > 0:
        args.modes.append('diffusion_content')
    if 'ds' in args.sample_mode and model.c > 0:
        args.modes.append('diffusion_structure')

    t = time.time()
    best_acc, best_epoch, best_std = 0, 0, -1
    count = 0
    model.train()

    for epoch in range(1, args.epochs+1):
        losses = []
        for mode in args.modes:
            optimizer.zero_grad()
            (sampled_features, sampled_adj, prior, nodes), sampled_labels = Data.sample(mode, return_nodes=True)
            loss = model(sampled_features, sampled_adj, sampled_labels, mode, nodes, prior)
            if torch.isnan(loss):
                print('nan loss!')
            loss.backward()
            optimizer.step()

            if epoch % args.log_every == 0:
                losses.append(loss.item())

        if epoch % args.log_every == 0:
            duration = time.time() - t
            msg = 'Epoch: {:04d} '.format(epoch)
            for mode, loss in zip(args.modes, losses):
                msg += 'loss_{}: {:.4f}\t'.format(mode, loss)
                if writer is not None:
                    writer.add_scalar('data/{}_loss'.format(mode), loss, epoch)
            logger.info(msg+' time: {:d}s '.format(int(duration)))

        if epoch % args.save_every == 0:
            learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(Data.feature_len)
            embedding = model.embedding(torch.LongTensor(args.nodes).to(args.device)).data.cpu().numpy()
            learned_embed.add([str(node) for node in args.nodes], embedding)
            train_acc, test_acc, std = evaluate(args, learned_embed, logger)
            duration = time.time() - t
            logger.info('Epoch: {:04d} '.format(epoch)+
                        'train_acc: {:.2f} '.format(train_acc)+
                        'test_acc: {:.2f} '.format(test_acc)+
                        'std: {:.2f} '.format(std)+
                        'time: {:d}s'.format(int(duration)))
            if writer is not  None:
                writer.add_scalar('data/test_acc', test_acc, epoch)
            if test_acc > best_acc:
                best_acc = test_acc
                best_std = std
                best_epoch = epoch
                # save_embedding(learned_embed, os.path.join(log_dir, 'embedding.bin'))
                # save_checkpoint({
                #     'args': args,
                #     'model': model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                # }, log_dir,
                #     f'epoch{epoch}_time{int(duration):d}_trainacc{train_acc:.2f}_testacc{test_acc:.2f}_std{std:.2f}.pth.tar', logger, True)
                count = 0
            else:
                if args.early_stop:
                    count += args.save_every
                if count >= args.patience:
                    logger.info('early stopped!')
                    break

    logger.info(f'best test acc={best_acc:.2f} +- {best_std:2f} @ epoch:{int(best_epoch):d}')
    return best_acc


def search_new(args, embedding, Data, logger, search_range=(0, 1, 1)):
    import skopt
    from sklearn.externals.joblib import Parallel, delayed

    def save(res):
        to_save = {'x':res.x, 'y':res.fun}
        json.dump(to_save, open(os.path.join(args.log_dir, 'res.json'), 'w'), indent=4)

    def callback(res):
        print('best so far', res.x, res.fun)
        save(res)

    def estimate(params):
        if args.use_superv:
            args.a, args.b, args.c, args.d, args.t = params
            log_dir = os.path.join(args.log_dir, f'a{args.a}_b{args.b}_c{args.c}_d{args.d}_t{args.t}')
        else:
            args.t = 0
            args.a, args.b, args.c, args.d, args.relation = params
            log_dir = os.path.join(args.log_dir, f'a{args.a}_b{args.b}_c{args.c}_d{args.d}_nrel{args.relation}')

        print('='*50, ' new trial: ', params, '='*50)

        # Initialize logger
        args.model_path = log_dir
        # writer = SummaryWriter(log_dir=log_dir)
        # writer.add_text('Parameters', str(vars(args)))
        print_config(args, logger)

        Data.set_superv_sample_ratio(args.t)

        torch.cuda.empty_cache()
        acc = train_new(args, embedding, Data, log_dir, logger)

        return -acc

    if args.use_superv:
        params = [np.linspace(0.5, 1, 3), np.linspace(0, 1, 5), np.linspace(0, 1, 5), np.linspace(0, 1, 5), np.linspace(0, 0.8, 5)]
    else:
        params = [np.linspace(0.5, 1, 3), np.linspace(0, 1, 5), np.linspace(0, 1, 5), np.linspace(0, 1, 5), np.linspace(2, 10, 5)]

    res = skopt.gp_minimize(func=estimate, dimensions=params, n_calls=args.budget, verbose=True, callback=callback)
    print(res.x, res.fun)
    save(res)


def search(args, Data, logger, search_range=(0, 1, 1)):
    import skopt

    def save(res):
        to_save = {'x':res.x, 'y':res.fun}
        json.dump(to_save, open(os.path.join(args.log_dir, 'res.json'), 'w'), indent=4)

    def callback(res):
        print('best so far', res.x, res.fun)
        save(res)

    def estimate(params):
        if args.use_superv:
            args.a, args.b, args.c, args.d, args.t = params
            log_dir = os.path.join(args.log_dir, f'a{args.a}_b{args.b}_c{args.c}_d{args.d}_t{args.t}')
        else:
            args.t = 0
            args.a, args.b, args.c, args.d, args.relation = params
            log_dir = os.path.join(args.log_dir, f'a{args.a}_b{args.b}_c{args.c}_d{args.d}_nrel{args.relation}')

        print('='*50, ' new trial: ', params, '='*50)

        # Initialize logger
        args.model_path = log_dir
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        Data.set_superv_sample_ratio(args.t)
        model = GCNDecoder(device=args.device,
                           nfeat=args.feature_len,
                           nhid=args.hidden,
                           ncont=args.content_len,
                           nrel=int(args.relation), rel_dim=args.rel_dim,
                           dropout=args.dropout,
                           a=float(args.a), b=float(args.b), c=float(args.c), d=float(args.d),
                           tau=float(args.tau), hard_gumbel=args.hard_gumbel)
        model.to(args.device)
        torch.cuda.empty_cache()
        acc = train(args, model, Data, log_dir, logger)

        return -acc

    if args.use_superv:
        params = [np.linspace(0.5, 1, 3), np.linspace(0, 1, 5), np.linspace(0, 1, 5), np.linspace(0, 1, 5), np.linspace(0, 0.8, 5)]
    else:
        params = [np.linspace(0.5, 1, 3), np.linspace(0, 1, 5), np.linspace(0, 1, 5), np.linspace(0, 1, 5), np.linspace(2, 10, 5)]

    res = skopt.gp_minimize(func=estimate, dimensions=params, n_calls=args.budget, verbose=True, callback=callback)
    print(res.x, res.fun)
    save(res)


def try_aggregate(args, Data, logger):
    t = time.time()
    learned_embed = gensim.models.keyedvectors.Word2VecKeyedVectors(Data.feature_len)
    for i in range(0, len(args.nodes), args.sample_embed):
        nodes = args.nodes[i:i+args.sample_embed]
        features, adj, _ = Data.sample_subgraph(nodes, False)
        features = features.to(args.device)
        adj = adj.to(args.device)
        embedding = torch.spmm(adj, features).data.cpu().numpy()
        learned_embed.add([str(node) for node in nodes], embedding)
    train_acc, test_acc, std = evaluate(args, learned_embed, logger)
    duration = time.time() - t
    logger.info('train_acc: {:.2f} '.format(train_acc)+
                'test_acc: {:.2f} '.format(test_acc)+
                'std: {:.2f} '.format(std)+
                'time: {:d}s'.format(int(duration)))

    logger.info(f'aggregate feature test acc={test_acc:.2f}')
    return test_acc


def run_part(args, Data, logger, repeat_times=1):
    def estimate(params):
        args.a, args.b, args.c, args.d = params
        model = GCNDecoder(device=args.device,
                           nfeat=args.feature_len,
                           nhid=args.hidden,
                           ncont=args.content_len,
                           nrel=args.relation, rel_dim=args.rel_dim,
                           dropout=args.dropout,
                           a=float(args.a), b=float(args.b), c=float(args.c), d=float(args.d),
                           tau=float(args.tau), hard_gumbel=args.hard_gumbel)
        model.to(args.device)
        torch.cuda.empty_cache()
        print_config(args, logger)
        acc = train(args, model, Data, log_dir, logger)
        return acc

    split = int(len(args.label_data) / 5)
    modes = ['node', 'link', 'diffusion_structure', 'diffusion_content']
    results = {i:[] for i in modes}
    for j in range(repeat_times):
        if args.use_superv:
            s1, s2 = split*j, split*(j+1)
            args.test = args.label_data[s1:s2]
            args.train = args.label_data[:s1]+args.label_data[s2:]
            Data = Dataset(args, args.dataset)
            args.feature_len = Data.feature_len
            args.content_len = Data.content_len
            args.num_node, args.num_link, args.num_diffusion = Data.num_node, Data.num_link, Data.num_diff

        for i, mode in enumerate(modes):
            params = [0] * len(modes)
            params[i] = 1
            args.model_path = os.path.join(args.log_dir, f'{mode}_{j}')
            results[mode].append(estimate(params))

    for mode, l in results.items():
        l = np.array(l)
        mean, std = l.mean(), l.std()
        logger.info(f'mode:{mode}\tacc={mean}+-{std}')


if __name__ == '__main__':
    # Initialize args and seed
    args = parse_args()
    # print('Number CUDA Devices:', torch.cuda.device_count())
    # call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    # torch.cuda.device(args.gpu)
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu > -1 else "cpu")
    # print('Active CUDA Device: GPU', torch.cuda.current_device())
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    args.superv_ratio = float(args.superv_ratio)

    # Load data
    if not args.eval_file:
        args.eval_file = f'data/{args.dataset}/eval/rel.txt'
    labels, labeled_data = set(), []
    nodes = set()
    with open(args.eval_file, 'r') as lf:
        for line in lf:
            if line.rstrip() == 'test':
                continue
            line = line.rstrip().split('\t')
            data1, data2, label = line[0], line[1], int(line[2])
            labeled_data.append((data1, data2, label))
            labels.add(label)
            nodes.update([int(data1), int(data2)])
    shuffle(labeled_data)
    args.nodes = list(nodes)
    args.label_data = labeled_data
    args.output_dim = len(labels)

    if args.mode == 'no_vi':
        args.use_superv = 0
        args.relation = 0

    if args.use_superv:
        args.relation = len(labels)
        test_split = int(0.8*len(args.label_data))
        args.test = args.label_data[test_split:]
        train_split = int(args.superv_ratio*len(args.label_data))
        args.train = args.label_data[:train_split]

    Data = Dataset(args, args.dataset)
    args.feature_len = Data.feature_len
    args.content_len = Data.content_len
    args.num_node, args.num_link, args.num_diffusion = Data.num_node, Data.num_link, Data.num_diff

    # initialize logger
    comment = f'_{args.dataset}_{args.mode}_{args.suffix}'
    current_time = datetime.now().strftime('%b_%d_%H-%M-%S')
    if args.prefix:
        base = os.path.join('running_log', args.prefix)
        log_dir = os.path.join(base, args.suffix)
    else:
        log_dir = os.path.join('running_log', current_time + comment)
    args.log_dir = log_dir
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    logger = myLogger(name='exp', log_path=os.path.join(log_dir, 'log.txt'))
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_text('Parameters', str(vars(args)))
    print_config(args, logger)
    logger.setLevel(args.log_level)

    # start
    if args.mode == 'search':
        search(args, Data, logger)
    elif args.mode == 'part':
        run_part(args, Data, logger)
    elif args.mode == 'search_new':
        # initial embedding by aggregating feature
        embedding = torch.FloatTensor(Data.num_node, Data.feature_len)
        all_nodes = list(range(Data.num_node))
        t = time.time()
        for i in trange(0, Data.num_node, args.sample_embed):
            nodes = all_nodes[i:i + args.sample_embed]
            features, adj, _ = Data.sample_subgraph(nodes, False)
            features = features.to(args.device)
            adj = adj.to(args.device)
            embedding[i:i + len(nodes)] = torch.spmm(adj, features)
        duration = time.time() - t

        search_new(args, embedding, Data, logger)
    elif args.mode == 'train_new':
        # initial embedding by aggregating feature
        embedding = torch.FloatTensor(Data.num_node, Data.feature_len)
        all_nodes = list(range(Data.num_node))
        t = time.time()
        for i in trange(0, Data.num_node, args.sample_embed):
            nodes = all_nodes[i:i + args.sample_embed]
            features, adj, _ = Data.sample_subgraph(nodes, False)
            features = features.to(args.device)
            adj = adj.to(args.device)
            embedding[i:i + len(nodes)] = torch.spmm(adj, features)
        duration = time.time() - t
        logger.info(f'initialize embedding time {int(duration):d}')

        # Train model
        t_total = time.time()
        train_new(args, embedding, Data, args.log_dir, logger, writer)
        logger.info("Optimization Finished!")
        logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    elif args.mode == 'no_vi':
        # Model and optimizer
        model = GCNDecoder3(device=args.device,
                           nfeat=args.feature_len,
                           nhid=args.hidden,
                           ncont=args.content_len,
                           dropout=args.dropout,
                           a=args.a, b=args.b, c=args.c, d=args.d)
        model.to(args.device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Train model
        t_total = time.time()
        train(args, model, Data, args.log_dir, logger, writer, optimizer)
        logger.info("Optimization Finished!")
        logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    else:
        # Model and optimizer
        model = GCNDecoder(device=args.device,
                           nfeat=args.feature_len,
                           nhid=args.hidden,
                           ncont=args.content_len,
                           nrel=args.relation, rel_dim=args.rel_dim,
                           dropout=args.dropout,
                           a=args.a, b=args.b, c=args.c, d=args.d,
                           tau=args.tau, hard_gumbel=args.hard_gumbel)
        model.to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.load_model:
            if os.path.isfile(args.load_model):
                checkpoint = torch.load(args.load_model)
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info("loaded checkpoint '{}' ".format(args.load_model))
            else:
                logger.error("no checkpoint found at '{}'".format(args.load_model))
                exit(1)

        # Train model
        t_total = time.time()
        train(args, model, Data, args.log_dir, logger, writer, optimizer)
        logger.info("Optimization Finished!")
        logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
