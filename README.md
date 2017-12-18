This is the original implementation of our paper, A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem ([arXiv:1706.10059](https://arxiv.org/abs/1706.10059)), together with a toolkit of portfolio management research.

* The deep reinforcement learning framework is the core part of the library.
The method is basically the policy gradient on immediate reward.
 One can configurate the topology, training method or input data in a separate json file. The training process will be recorded and user can visualize the training using tensorboard.
Result summary and parallel training are allowed for better hyper-parameters optimization.
* The financial-model-based portfolio management algorithms are also embedded in this library for comparision purpose, whose implementation is based on Li and Hoi's toolkit [OLPS](https://github.com/OLPS/OLPS).

## Differences from the article version
Note that this library is a part of our main project, and it is several versions ahead of the article.

* In this version, some technical bugs are fixed and improvements in hyper-parameter tuning and engineering are made.
  * The most important bug in the arxiv v2 article is that the test time-span mentioned is about 30% shorter than the actual experiment. Thus the volumn-observation interval (for asset selection) overlapped with the backtest data in the paper.
* With new hyper-parameters, users can train the models with smaller time durations.(less than 30 mins)
* All updates will be incorporated into future versions of the paper.
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
Please check out [User Guide](user_guide.md)

## Acknowledgement
This project would not have been finished without using the codes from the following open source projects:
* [Online Portfolio Selection toolbox](https://github.com/OLPS/OLPS)

## Community Contribution
We welcome contributions from the community, including but not limited to:
* Bug fixing
* Interfacing to other markets such as stock, futures, options
* Adding broker API (under `marketdata`)
* More backtest strategies (under `tdagent`)

## Risk Disclaimer (for Live-trading)

There is always risk of loss in trading. **All trading strategies are used at your own risk**

*The volumes of many cryptocurrency markets are still low. Market impact and slippage may badly affect the results during live trading.*

## Donation
If you have made some profits because of this project or you just love reading our codes, please consider making a small donation to our ongoing projects via the following BTC or ETH address. All donations will be used as student stipends. 

* BTC: [1PEHK1nVi8x4HQM1A67anyfYzdDFoqUo21](https://blockchain.info/address/1PEHK1nVi8x4HQM1A67anyfYzdDFoqUo21)
* ETH: [0xa8197289e16C0cCad0a4838719ce11C9A920cfB7](https://etherscan.io/address/0xa8197289e16C0cCad0a4838719ce11C9A920cfB7)
