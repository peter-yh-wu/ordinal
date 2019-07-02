'''
Script to train model with ordinal triplet loss

Peter Wu
peterw1@andrew.cmu.edu
'''

import argparse
import numpy as np
import os
import random
import sys
import torch

from model import *
from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', help='batch size;', type=int, default=64)
    parser.add_argument('--data', help='ComParE, BoAW-2000, etc;', nargs='+', type=str, default='ComParE')
    parser.add_argument('--process', help='none, upsample, pca, etc;', nargs='+', type=str, default='none')
    parser.add_argument('--pca-param', help='fraction of variance retained;', type=float, default=0.99)
    parser.add_argument('--lr', help='lr;', type=float, default=1e-05)
    parser.add_argument('--loss-param', help='otl loss param;', type=float, default=1.0)
    parser.add_argument('--s-margin', help='max dist between s and a;', type=int, default=1)
    parser.add_argument('--d-margin', help='min dist between d and a;', type=int, default=2)
    parser.add_argument('--dropout', help='dropout;', type=float, default=0.0)
    parser.add_argument('--num-layers', help='number of layers;', type=int, default=3)
    parser.add_argument('--hidden-dim', help='MLP hidden dim;', type=int, default=128)
    parser.add_argument('--emb-dim', help='MLP emb dim;', type=int, default=32)
    parser.add_argument('--loss-type', help='wloss for weighted loss;', type=str, default="normal")
    parser.add_argument('--log-path', help='log path;', type=str, default="log")
    parser.add_argument('--num-epochs', help='number of epochs;', type=int, default=100)
    parser.add_argument('--patience', help='patience;', type=int, default=5)
    parser.add_argument('--seed', help='random seed;', type=int, default=0)
    return parser.parse_args()


def get_result_paths(log_path):
    met_model_path = 'met_model.ckpt'
    loss_model_path = 'loss_model.ckpt'
    met_conf_mat_path = 'met_conf_mat.npy'
    loss_conf_mat_path = 'loss_conf_mat.npy'
    slash_index = log_path.rfind('/')
    dot_index = log_path.rfind('.')
    if slash_index == -1:
        if dot_index != -1:
            met_model_path = log_path[:dot_index]+'-met.ckpt'
            loss_model_path = log_path[:dot_index]+'-loss.ckpt'
            met_conf_mat_path = log_path[:dot_index]+'-met_conf_mat.npy'
            loss_conf_mat_path = log_path[:dot_index]+'-loss_conf_mat.npy'
    else:
        if dot_index == -1 or dot_index < slash_index:
            met_model_path = log_path[:slash_index]+'/met_model.ckpt'
            loss_model_path = log_path[:slash_index]+'/loss_model.ckpt'
            met_conf_mat_path = log_path[:slash_index]+'/met_conf_mat.npy'
            loss_conf_mat_path = log_path[:slash_index]+'/loss_conf_mat.npy'
        else:
            met_model_path = log_path[:dot_index]+'-met.ckpt'
            loss_model_path = log_path[:dot_index]+'-loss.ckpt'
            met_conf_mat_path = log_path[:dot_index]+'-met_conf_mat.npy'
            loss_conf_mat_path = log_path[:dot_index]+'-loss_conf_mat.npy'
    return met_model_path, loss_model_path, met_conf_mat_path, loss_conf_mat_path


def main(verbose=True):
    args = parse_args() # todo header labels in text files
    data_type = args.data
    if isinstance(data_type, str):
        data_type = [data_type]

    random.seed(args.seed)
    np.random.seed(args.seed)

    met_model_path, loss_model_path, met_conf_mat_path, loss_conf_mat_path = get_result_paths(args.log_path)

    input_dim, _, train_x, train_y, dev_x, dev_y = \
        load_data(args.data, args.process, args.pca_param)
    y_values = list(set(np.unique(train_y)).union(set(np.unique(dev_y))))
    y_values.sort()
    num_classes = len(y_values)
    output_dim = 2 # because soft labels

    y_indices_list_train = [np.where(train_y == i)[0] for i in y_values]
    y_value_counts_train = [len(y) for y in y_indices_list_train]
    y_indices_list_dev = [np.where(dev_y == i)[0] for i in y_values]
    y_value_counts_dev = [len(y) for y in y_indices_list_dev]
    y_value_counts = [t+d for t, d in zip(y_value_counts_train, y_value_counts_dev)]

    grouped_train_x = group_data_by_class(train_x, train_y, num_classes)
    
    sia_train_p_y_indices = get_sia_p_y_indices(train_y, num_classes)
    grouped_train_x_flat = [item for sublist in grouped_train_x for item in sublist]

    train_y_tensor = torch.LongTensor(train_y) # y-values 0-indexed
    dev_y_tensor = torch.LongTensor(dev_y)
    
    train_y_tensor = mk_y_slabs(train_y_tensor, num_classes)
    dev_y_tensor = mk_y_slabs(dev_y_tensor, num_classes)
    if torch.cuda.is_available():
        train_y_tensor = train_y_tensor.cuda()
        dev_y_tensor = dev_y_tensor.cuda()

    num_train = len(train_y_tensor)
    loss_weights = num_train/2/torch.sum(train_y_tensor, 0)
    if torch.cuda.is_available():
        loss_weights = loss_weights.cuda()

    net = MLP_TriSlab(input_dim, output_dim, args)
    print_log(net, args.log_path)
    if torch.cuda.is_available():
        net.cuda()
    
    criterion = OrdinalTripletLoss(alpha=args.loss_param, weight=loss_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    _, y_pred = slab_predict(net, dev_x, num_classes)
    acc, conf_mat, spearman = get_metrics(dev_y, y_pred)
    best_acc = acc
    best_met = spearman
    print_log('acc before training: %f' % best_acc, args.log_path)
    print_log('spearman before training: %f' % best_met, args.log_path)
    
    best_val_loss = sys.maxsize
    prev_best_epoch = 0
    for e in range(args.num_epochs):
        ss = sample_ss(grouped_train_x_flat, y_value_counts_train, sia_train_p_y_indices, args.s_margin)
        ds = sample_ds(grouped_train_x_flat, y_value_counts_train, sia_train_p_y_indices, args.d_margin)
        train_otl(net, optimizer, criterion, (train_x, ss, ds), train_y_tensor, args.batch_size)
        logits, y_pred = slab_predict(net, dev_x, num_classes)
        loss = criterion.cross_entropy(logits, dev_y_tensor)
        
        val_loss = loss.item()
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            prev_best_epoch = e
            torch.save(net.state_dict(), loss_model_path)
            np.save(loss_conf_mat_path, conf_mat)
        elif e - prev_best_epoch > args.patience:
            break
        
        acc, conf_mat, spearman = get_metrics(dev_y, y_pred)
        print_log('epoch %d: acc - %f, spearman - %f' % (e+1, acc, spearman), args.log_path)
        
        if acc > best_acc:
            best_acc = acc
        if spearman > best_met:
            best_met = spearman
            torch.save(net.state_dict(), met_model_path)
            np.save(met_conf_mat_path, conf_mat)
    
    print_log('best acc - %f, best spearman - %f' % (best_acc, best_met), args.log_path)


if __name__ == "__main__":
    main()