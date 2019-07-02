'''
Utils for ordinal triplet loss script

Peter Wu
peterw1@andrew.cmu.edu
'''

import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from torch.nn.modules.distance import PairwiseDistance

def print_log(s, log_path):
    '''Prints the given string and writes it to the given log file

    Args:
        s: string to print & log
        log_path: path to log file
    '''
    print(s)
    with open(log_path, 'a+') as ouf:
        ouf.write("%s\n" % s)


def group_data_by_class(data, y_vals, num_classes):
    '''groups data by y-class

    Args:
        data: iterable of x values
        y_vals: 1-dim np array of ints, values are in {0, ..., num_classes-1}
        num_classes: number of y classes

    Return:
        grouped_data: size-num_classes list of lists
    '''
    grouped_data = [[] for _ in range(num_classes)]
    for i, d in enumerate(data):
        y_val = y_vals[i]
        grouped_data[y_val].append(d)
    return grouped_data


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


def load_data(data_type, process, pca_param=0.99):
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

    if isinstance(data_type, str):
        data_type = [data_type]
    if isinstance(process, str):
        process = [process]
    
    input_dim = 0
    
    train_x = None
    dev_x = None
    baseline_feats = ['ComParE','BoAW-125','BoAW-250','BoAW-500','BoAW-1000','BoAW-2000','auDeep-40','auDeep-50','auDeep-60','auDeep-70','auDeep-fused']
    for feat in baseline_feats:
        if feat in data_type:
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


def mk_y_slabs(ys, num_classes):
    '''Create soft labels based on given y-values
    
    Class y is mapped to [y/num_classes, 1-y/num_classes]

    Args:
        ys: tensor with shape (num_samples,), y-values are 0-indexed
        num_classes: number of y classes
    
    Return:
        tensor comprised of soft labels, has shape (num_samples, 2)
    '''
    col1 = ys.float()/num_classes
    col2 = 1-col1
    return torch.stack([col1, col2]).transpose(1,0)


def slab_predict(model, xs, num_classes):
    '''Evaluates model on given data

    Assumes that the model outputs soft labels

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
        inputs = torch.FloatTensor(xs)

        inputs = Variable(inputs)
        if torch.cuda.is_available():
            inputs = inputs.cuda()

        logits = F.softmax(model(inputs), 1)
        preds_tens = torch.clamp(torch.round(logits[:, 0]*num_classes), max=num_classes-1)
        return logits, preds_tens.cpu().numpy()


def sample_ss(x_groups_flat, num_ys, sia_p_y_indices, margin):
    '''Sample positive/similar samples for given anchor samples

    Args:
        x_groups_flat: list of x-values
        num_ys: used to calculate range in x_groups_flat to sample from
        sia_p_y_indices: resulting indices to put samples for each class
        margin: max allowed absolute difference between y classes
    
    Return:
        ps: 1-dim size-num_sia_xs list of x-values
    '''
    i1 = 0
    i2 = sum(num_ys[:margin])
    num_classes = len(num_ys)
    all_raw_samples = []
    num_sia_xs = 0
    for curr_class, num_y in enumerate(num_ys):
        sia_num_y = len(sia_p_y_indices[curr_class])
        num_sia_xs += sia_num_y
        if curr_class > margin:
            i1 += num_ys[curr_class-margin-1]
        if curr_class < num_classes-margin:
            i2 += num_ys[curr_class+margin]
        curr_xs = x_groups_flat[i1:i2]
        num_curr_xs = len(curr_xs)
        curr_samples = []
        num_reps = int(sia_num_y/num_curr_xs)
        num_left = sia_num_y % num_curr_xs
        curr_samples += curr_xs*num_reps
        curr_samples += random.sample(curr_xs, num_left)
        random.shuffle(curr_samples)
        all_raw_samples.append(curr_samples)
    ps = [None for _ in range(num_sia_xs)]
    for curr_class, indices in enumerate(sia_p_y_indices):
        for i, ps_i in enumerate(indices):
            ps[ps_i] = all_raw_samples[curr_class][i]
    return ps


def sample_ds(x_groups_flat, num_ys, sia_p_y_indices, margin):
    '''Sample negative/different samples for given anchor samples

    Args:
        x_groups_flat: list of x-values
        num_ys: used to calculate range in x_groups_flat to sample from
        sia_p_y_indices: resulting indices to put samples for each class
        margin: min allowed absolute difference between y classes
    
    Return:
        ns: 1-dim size-num_sia_xs list of x-values
    '''
    i1 = 0
    i2 = sum(num_ys[:margin])
    num_classes = len(num_ys)
    all_raw_samples = []
    num_sia_xs = 0
    for curr_class, num_y in enumerate(num_ys):
        sia_num_y = len(sia_p_y_indices[curr_class])
        num_sia_xs += sia_num_y
        if curr_class > margin:
            i1 += num_ys[curr_class-margin-1]
        if curr_class < num_classes-margin:
            i2 += num_ys[curr_class+margin]
        curr_xs = x_groups_flat[:i1]+x_groups_flat[i2:]
        num_curr_xs = len(curr_xs)
        curr_samples = []
        num_reps = int(sia_num_y/num_curr_xs)
        num_left = sia_num_y % num_curr_xs
        curr_samples += curr_xs*num_reps
        curr_samples += random.sample(curr_xs, num_left)
        random.shuffle(curr_samples)
        all_raw_samples.append(curr_samples)
    ns = [None for _ in range(num_sia_xs)]
    for curr_class, indices in enumerate(sia_p_y_indices):
        for i, ns_i in enumerate(indices):
            ns[ns_i] = all_raw_samples[curr_class][i]
    return ns


def get_sia_p_y_indices(ys, num_classes):
    '''Finds where each class appears in given list of labels

    Ensures that the indices returned are in the same relative
        ordering as the original data

    Args:
        ys: 1-dim np array of 0-indexed y values

    Return:
        sia_p_y_indices: list of 1-dim np arrays
            array_i contains indices in sia_xs where y_p = i
    '''
    sia_p_y_indices = []
    for c in range(num_classes):
        sia_p_y_indices.append(np.where(ys == c)[0])
    return sia_p_y_indices


def train_otl(model, optimizer, criterion, X_train, y_train, batch_size=64):
    '''Trains model on given data

    Assumes model supports ordinal triplet loss

    Args:
        model: nn.Module
        optimizer: optim.Optimizer
        criterion: calculates loss
        X_train: np array with shape (num_train, feature_dim)
        y_train: LongTensor with shape (num_train,)
    '''
    model.train()
    optimizer.zero_grad()
    a_vals, ps, ns = X_train
    num_samples = len(ns)

    for i in range(0, num_samples, batch_size):
        a_batch = a_vals[i:i+batch_size]
        p_batch = np.array(ps[i:i+batch_size])
        n_batch = np.array(ns[i:i+batch_size])

        a, p, n = torch.FloatTensor(a_batch), torch.FloatTensor(p_batch), torch.FloatTensor(n_batch)
        a, p, n = Variable(a), Variable(p), Variable(n)

        y_batch = y_train[i:i+batch_size]
        targets = Variable(y_batch)

        if torch.cuda.is_available():
            a, p, n = a.cuda(), p.cuda(), n.cuda()
            targets = targets.cuda()

        o_a = model.forward_emb(a)
        o_p = model.forward_emb(p)
        o_n = model.forward_emb(n)
        logits = model.forward_slab(o_a)
        loss = criterion(logits, targets, a, p, n)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()


def logistic(x):
    '''Logistic Function: log_2 (1+2^{-x})'''
    return torch.log2(1+torch.pow(2, x))


class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.pdist  = PairwiseDistance(2)
    
    def forward(self, anchor, positive, negative):
        pos_dist   = self.pdist.forward(anchor, positive)
        neg_dist   = self.pdist.forward(anchor, negative)
        return torch.mean(logistic(neg_dist-pos_dist))


class OrdinalTripletLoss(nn.Module):
    def __init__(self, alpha=0.1, weight=None):
        super(OrdinalTripletLoss, self).__init__()
        self.fr_loss = TripletLoss()
        self.cross_entropy = nn.BCELoss(weight=weight)
        self.alpha = alpha

    def forward(self, logits, y_true, a, p, n):
        loss1 = self.cross_entropy(logits, y_true)
        loss2 = self.fr_loss(a, p, n)
        loss  = loss1+self.alpha*loss2
        return loss