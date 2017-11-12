This library is the implementation of https://arxiv.org/pdf/1706.10059.pdf together with toolkit of portfolio management research.

* The deep reinforcement learning framework is the core part of the library.
The method is basically the policy gradient on immediate reward.
 One can config the topology, training method or input data in a separate json file. The training process will be record and user could visualize the training using tensorboard.
* Also, result summary and parallel training are allowed for better hyper-parameters optimization.
* The financial model based portfolio management algorithms are also embedded in this library for comparision purpose, whose implementation is based on https://github.com/OLPS/OLPS.

## Differences between article version
Note that this library is a part of our main project, which is several versions beyond the article described.

* In this version, some technical bugs has been fixed and there are also some improvement in hyper-parameters and engineering.
* The most important bug in the arxiv article version is the test time span mentioned in the article is about 30% shorter than the real one.
* With new hyper-parameters, users could training the model in a very fast pace.(less than 30 mins)
* All the updates would be put in the next version of the article.

## Platform Support
The python 3.5 in windows and python 2.7 in linux are supported.

