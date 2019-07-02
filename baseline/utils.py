'''
Utils for baseline script

Peter Wu
peterw1@andrew.cmu.edu
'''

import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable


def print_log(s, log_path):
    '''Prints the given string and writes it to the given log file

    Args:
        s: string to print & log
        log_path: path to log file
    '''
    print(s)
    with open(log_path, 'a+') as ouf:
        ouf.write("%s\n" % s)


def load_baseline_data(feature_set, data_dir):
    '''
    Args:
        feature_set: string, type of feature to use
        data_dir: string, absolute path to data directory

    Return:
        X_train: np array with shape (num_train, feature_dim)
        X_devel: np array with shape (num_dev, feature_dim)
    '''
    task_name = 'ComParE2019_ContinuousSleepiness'
    feat_conf = {'ComParE':     (6373, 1, ';', 'infer'),
                'BoAW-125':     (250, 1, ';',  None),
                'BoAW-250':     (500, 1, ';',  None),
                'BoAW-500':     (1000, 1, ';',  None),
                'BoAW-1000':    (2000, 1, ';',  None),
                'BoAW-2000':    (4000, 1, ';',  None),
                'auDeep-40':    (1024, 2, ',', 'infer'),
                'auDeep-50':    (1024, 2, ',', 'infer'),
                'auDeep-60':    (1024, 2, ',', 'infer'),
                'auDeep-70':    (1024, 2, ',', 'infer'),
                'auDeep-fused': (4096, 2, ',', 'infer')}
    num_feat = feat_conf[feature_set][0]
    ind_off  = feat_conf[feature_set][1]
    sep      = feat_conf[feature_set][2]
    header   = feat_conf[feature_set][3]
    features_path = os.path.join(data_dir, 'features') + '/'
    X_train = pd.read_csv(features_path + task_name + '.' + feature_set + '.train.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
    X_devel = pd.read_csv(features_path + task_name + '.' + feature_set + '.devel.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_devel = scaler.transform(X_devel)
    return X_train, X_devel


def load_data(data_name, process, pca_param=0.99):
    '''Loads the specified data

    y-values returned are 0-indexed

    Args:
        data_name: string or list, e.g. ['ComParE', 'BoAW-2000']
        process: string or list, e.g. ['pca', 'upsample']
        pca_param: float, fraction of variance retained
    
    Return:
        input_dim: int, feature_dim
        output_dim: int, number of y classes
        train_x: np array with shape (num_train, feature_dim)
        train_y: np array with shape (num_train,)
        dev_y: np array with shape (num_dev, feature_dim)
        dev_y: np array with shape (num_dev,)
    '''
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(parent_dir, 'data')

    if isinstance(data_name, str):
        data_name = [data_name]
    if isinstance(process, str):
        process = [process]
    
    input_dim = 0

    train_x = None
    dev_x = None
    baseline_feats = ['ComParE','BoAW-125','BoAW-250','BoAW-500','BoAW-1000','BoAW-2000','auDeep-40','auDeep-50','auDeep-60','auDeep-70','auDeep-fused']
    for feat in baseline_feats:
        if feat in data_name:
            curr_train_x, curr_dev_x = load_baseline_data(feat, data_dir)
            input_dim += curr_train_x.shape[1]
            if train_x is not None:
                train_x = np.concatenate((train_x,curr_train_x), axis=1)
            else:
                train_x = curr_train_x
            if dev_x is not None:
                dev_x = np.concatenate((dev_x,curr_dev_x), axis=1)
            else:
                dev_x = curr_dev_x

    label_file = os.path.join(data_dir, 'lab', 'labels.csv')
    if not os.path.exists(label_file):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        label_file = os.path.join(parent_dir, 'data', 'labels.csv')
    df_labels = pd.read_csv(label_file)
    train_y = pd.to_numeric(df_labels['label'][df_labels['file_name'].str.startswith('train')]).values
    dev_y = pd.to_numeric(df_labels['label'][df_labels['file_name'].str.startswith('devel')]).values

    output_dim = 9

    if 'upsample' in process:
        y_values = list(set(np.unique(train_y)).union(set(np.unique(dev_y))))
        y_values.sort()
        y_indices_list = [np.where(train_y == i)[0] for i in y_values]
        y_value_counts = [len(y) for y in y_indices_list]
        max_ys = np.max(y_value_counts)
        new_y_value_counts = [max_ys-n for n in y_value_counts]
        new_bucket_indices = [np.random.choice(num_ys_prev, size=num_ys_new) 
                            for num_ys_new, num_ys_prev 
                                in zip(new_y_value_counts, y_value_counts)]
        new_x_indices = [y_indices_list[i][js] for i, js in enumerate(new_bucket_indices)]
        new_x_indices = [item for sublist in new_x_indices for item in sublist]
        np.random.shuffle(new_x_indices)
        
        new_train_x = [train_x[i] for i in new_x_indices]
        if isinstance(train_x, list):
            train_x = train_x + new_train_x
        else:
            new_train_x = np.array(new_train_x)
            train_x = np.concatenate([train_x, new_train_x], axis=0)
        new_train_y = [train_y[i] for i in new_x_indices]
        new_train_y = np.array(new_train_y)
        train_y = np.concatenate([train_y, new_train_y])

    if 'pca' in process:
        pca = PCA(n_components=pca_param)
        pca.fit(train_x)
        input_dim = pca.n_components_
        train_x = pca.transform(train_x)
        dev_x = pca.transform(dev_x)

    if isinstance(train_x, list):
        indices = [i for i in range(len(train_x))]
        zs = list(zip(train_x, indices))
        random.shuffle(zs)
        train_x, indices = zip(*zs)
        train_x = list(train_x)
        indices = list(indices)
        train_y = train_y[indices]
    else:
        p = np.random.permutation(len(train_x))
        train_x = train_x[p]
        train_y = train_y[p]
    return input_dim, output_dim, train_x, train_y-1, dev_x, dev_y-1


def predict(model, x_eval):
    '''Evaluates model on given data

    Args:
        model: nn.Module
        x_eval: np array with shape (num_eval, feature_dim)

    Return:
        logits: tensor with shape (num_eval, num_classes)
        y_pred: np array of shape (num_eval,), elements are between
            0 and num_classes-1, inclusive
    '''
    model.eval()
    with torch.no_grad():
        inputs = torch.FloatTensor(x_eval)

        inputs = Variable(inputs)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        logits = model(inputs)
        values, indices = torch.max(logits, 1)
        return logits, indices.cpu().numpy()


def train(model, optimizer, criterion, X_train, y_train, batch_size=64):
    '''Trains model on given data

    Args:
        model: nn.Module
        optimizer: optim.Optimizer
        criterion: calculates loss
        X_train: np array with shape (num_train, feature_dim)
        y_train: LongTensor with shape (num_train,)
    '''
    model.train()
    optimizer.zero_grad()
    num_samples = len(y_train)

    for i in range(0, num_samples, batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        inputs = torch.FloatTensor(X_batch)

        inputs, targets = Variable(inputs), Variable(y_batch)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()


def get_metrics(y_true, y_pred):
    '''Calculates metrics based on given true and predicted values

    Args:
        y_true: np array with shape (num_eval,)
        y_pred: np array with shape (num_eval,)

    Return:
        acc: float, accuracy
        conf_mat: array, confusion matrix
        spearman: float, spearman coefficient
    '''
    spearman = spearmanr(y_true, y_pred)[0]
    if np.isnan(spearman):  # Might occur when the prediction is a constant
        spearman = 0.
    return accuracy_score(y_true, y_pred), \
        confusion_matrix(y_true, y_pred), \
        spearman
