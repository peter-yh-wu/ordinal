# Ordinal Triplet Loss

PyTorch implementation of the ordinal triplet loss function detailed in the Interspeech [paper](https://pdfs.semanticscholar.org/17b6/74d628358864ae2548eaf41ff1c9cd384d59.pdf).

## Background

Ordinal triplet loss (OTL) is a classification loss function that accounts for the relative distance between ordinal classes. The two key techniques used to realize this are soft labels and a modified triplet loss function.

## Quick Start

- To use soft labels only, see the [`mk_y_slabs`](https://github.com/peter-yh-wu/ordinal/blob/master/slab/utils.py#L222) and [`slab_predict`](https://github.com/peter-yh-wu/ordinal/blob/master/slab/utils.py#L239) functions in `slab/utils.py`. For usage examples, see `slab/main.py`.

- To use the ordinal triple loss function, see the [`OrdinalTripletLoss`](https://github.com/peter-yh-wu/ordinal/blob/master/otl/utils.py#L409) class in `otl/utils.py`. For usage examples, see `otl/main.py`.

## Reproducing Results

- Download the challenge data, create a directory called ```data``` at the same level as ```baseline```, and move ```labels.csv``` and the ```features``` directory into ```data```
- The ```baseline``` directory contains instructions on how to run the baseline SVR and MLP models
- The ```slab``` directory contains instructions on how to run an MLP trained with soft labels
- The ```otl``` directory contains instructions on how to run an MLP trained using ordinal triplet loss
