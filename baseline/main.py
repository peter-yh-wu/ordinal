'''
Script to run SVR and MLP baselines

e.g. to run MLP: python3 main.py --model MLP --data ComParE --process pca upsample
e.g. to run SVR: python3 main.py --model SVR --data ComParE BoAW-2000

Peter Wu
peterw1@andrew.cmu.edu
'''

import argparse
import numpy as np
import os
import random
import sys
import torch

from sklearn import svm
from sklearn.externals import joblib

import model

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comp', help='complexity of svr;', type=float, default=0.1)
    parser.add_argument('--batch-size', help='batch size;', type=int, default=64)
    parser.add_argument('--data', help='ComParE, BoAW-2000, etc;', nargs='+', type=str, default='ComParE')
    parser.add_argument('--process', help='none, upsample, pca, etc;', nargs='+', type=str, default='none')
    parser.add_argument('--pca-param', help='fraction of variance retained;', type=float, default=0.99)
    parser.add_argument('--lr', help='lr;', type=float, default=1e-05)
    parser.add_argument('--dropout', help='dropout;', type=float, default=0.0)
    parser.add_argument('--num-layers', help='number of layers;', type=int, default=3)
    parser.add_argument('--hidden-dim', help='hidden_dim;', type=int, default=128)
    parser.add_argument('--model', help='model;', type=str, default="MLP")
    parser.add_argument('--loss-type', help='wloss for weighted loss;', type=str, default="normal")
    parser.add_argument('--log-path', help='log path;', type=str, default="log")
    parser.add_argument('--num-epochs', help='number of epochs;', type=int, default=100)
    parser.add_argument('--patience', help='patience;', type=int, default=5)
    parser.add_argument('--seed', help='random seed;', type=int, default=0)
    return parser.parse_args()


def get_result_paths(log_path):
    model_path = 'model.ckpt'
    conf_mat_path = 'conf_mat.npy'
    slash_index = log_path.rfind('/')
    dot_index = log_path.rfind('.')
    if slash_index == -1:
        if dot_index != -1:
            model_path = log_path[:dot_index]+'.ckpt'
            conf_mat_path = log_path[:dot_index]+'-conf_mat.npy'
    else:
        if dot_index == -1 or dot_index < slash_index:
            model_path = log_path[:slash_index]+'/model.ckpt'
            conf_mat_path = log_path[:slash_index]+'/conf_mat.npy'
        else:
            model_path = log_path[:dot_index]+'.ckpt'
            conf_mat_path = log_path[:dot_index]+'-conf_mat.npy'
    return model_path, conf_mat_path


def main(verbose=True):
    args = parse_args()
    data_type = args.data
    if isinstance(data_type, str):
        data_type = [data_type]

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_path, conf_mat_path = get_result_paths(args.log_path)

    input_dim, output_dim, train_x, train_y, dev_x, dev_y = \
        load_data(args.data, args.process, args.pca_param)
    y_values = list(set(np.unique(train_y)).union(set(np.unique(dev_y))))
    y_values.sort()

    train_y_tensor = torch.LongTensor(train_y) # y-values 0-indexed
    dev_y_tensor = torch.LongTensor(dev_y) # y-values 0-indexed
    if torch.cuda.is_available():
        train_y_tensor = train_y_tensor.cuda()
        dev_y_tensor = dev_y_tensor.cuda()
    num_classes = len(y_values)

    loss_weights = None
    if args.loss_type == 'wloss':
        y_indices_list_train = [np.where(train_y == i)[0] for i in y_values]
        y_value_counts_train = [len(y) for y in y_indices_list_train]
        y_indices_list_dev = [np.where(dev_y == i)[0] for i in y_values]
        y_value_counts_dev = [len(y) for y in y_indices_list_dev]
        y_value_counts = [t+d for t, d in zip(y_value_counts_train, y_value_counts_dev)]
        min_count = np.min(y_value_counts)
        weight_list = [min_count/c for c in y_value_counts]
        loss_weights = torch.FloatTensor(weight_list)

    if args.model == 'SVR':
        net = svm.LinearSVR(C=args.comp, random_state=0)
        net.fit(train_x, train_y)
        y_pred = np.clip( np.asarray(np.round(net.predict(dev_x)), dtype=np.int32), 1, 9)
        acc, conf_mat, spearman = get_metrics(dev_y, y_pred)
        print_log('acc - %f, spearman - %f' % (acc, spearman), args.log_path)
        np.save(conf_mat_path, conf_mat)
        joblib.dump(net, model_path)
    else:
        model_class = getattr(model, args.model)
        net = model_class(input_dim, output_dim, args)
        print_log(net, args.log_path)
        if torch.cuda.is_available():
            net.cuda()

        criterion = nn.CrossEntropyLoss(weight=loss_weights)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        _, y_pred = predict(net, dev_x)
        acc, conf_mat, spearman = get_metrics(dev_y, y_pred)
        best_acc = acc
        best_met = spearman
        print_log('acc before training: %f' % best_acc, args.log_path)
        print_log('spearman before training: %f' % best_met, args.log_path)

        best_val_loss = sys.maxsize
        prev_best_epoch = 0
        for e in range(args.num_epochs):
            train(net, optimizer, criterion, train_x, train_y_tensor, batch_size=args.batch_size)
            logits, y_pred = predict(net, dev_x)
            loss = criterion(logits, dev_y_tensor)
            
            val_loss = loss.item()
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                prev_best_epoch = e
            elif e - prev_best_epoch > args.patience:
                break
            
            acc, conf_mat, spearman = get_metrics(dev_y, y_pred)
            print_log('epoch %d: acc - %f, spearman - %f' % (e+1, acc, spearman), args.log_path)

            if acc > best_acc:
                best_acc = acc
            if spearman > best_met:
                best_met = spearman
                torch.save(net.state_dict(), model_path)
                np.save(conf_mat_path, conf_mat)

        print_log('best acc - %f, best spearman - %f' % (best_acc, best_met), args.log_path)


if __name__ == "__main__":
    main()