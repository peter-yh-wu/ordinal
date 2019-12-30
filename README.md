# Ordinal Triplet Loss

PyTorch implementation of Ordinal Triplet Loss [paper](https://pdfs.semanticscholar.org/17b6/74d628358864ae2548eaf41ff1c9cd384d59.pdf)

Using data from the 2019 ComParE Sleepiness challenge ([website](http://www.compare.openaudio.eu/compare2019/))

# Setup Instructions

- Download the challenge data, create a directory called ```data``` at the same level as ```baseline```, and move ```labels.csv``` and the ```features``` directory into ```data```
- The ```baseline``` directory contains instructions on how to run the baseline SVR and MLP models
- The ```slab``` directory contains instructions on how to run an MLP trained with soft labels
- The ```otl``` directory contains instructions on how to run an MLP trained using ordinal triplet loss
