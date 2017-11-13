This is the orinal implementation of our paper, A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem ([arXiv:1706.10059](https://arxiv.org/abs/1706.10059)), together with a toolkit of portfolio management research develped by Li and Hoi.

* The deep reinforcement learning framework is the core part of the library.
The method is basically the policy gradient on immediate reward.
 One can config the topology, training method or input data in a separate json file. The training process will be record and user could visualize the training using tensorboard.
* Also, result summary and parallel training are allowed for better hyper-parameters optimization.
* The financial-model-based portfolio management algorithms are also embedded in this library for comparision purpose, whose implementation is based on Li and Hoi's toolkit [OLPS](https://github.com/OLPS/OLPS).

## Differences between article version
Note that this library is a part of our main project, which is several versions beyond the article described.

* In this version, some technical bugs has been fixed and there are also some improvement in hyper-parameters and engineering.
  * The most important bug in the arxiv v2 article is the test time span mentioned in the article is about 30% shorter than the real one.
* With new hyper-parameters, users can train the model in a very fast pace.(less than 30 mins)
* All updates will be incorporated into future versions of the article.
* Original versioning history,  and internal discussions, including some in-code comments, are removed in this open-sourced edition. These contains our unimplemented ideas, some of which will very likely become the foundations of our future publications

## Platform Support
Python 3.5+ in windows and Python 2.7+/3.5+ in linux are supported.

## Dependencies
Install Dependencies via `pip install -r requirements.txt`

* tensorflow (>= 1.0.0)
* tflearn
* pandas
* ...

## User Guide

Check [wiki page](https://github.com/ZhengyaoJiang/PGPortfolio/wiki/User-Guide)
