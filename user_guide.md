# User Guide

## Quickstart

1. Edit [`pgportfolio/net_config.json`](pgportfolio/net_config.json)
2. Generate an agent:
```
python main.py --mode=generate --repeat=1
```
3. Download the data:
```
python main.py --mode=download_data
```
4. Train the agent:
```
python main.py --mode=train --processes=1
```
5. Compare the result with other algorithms:
```
python main.py --mode=plot --algos=crp,olmar,1 --labels=crp,olmar,nnagent
python main.py --mode=table --algos=1,olmar,ons --labels=nntrader,olmar,ons`
```

See below for details on each step.

## Configuration File
`pgportfolio/net_config.json` contains all the configuration parameters. The software can be configured by modifying this file and without any changes to the code.

Here is a description of the parameters in the configuration file.
### Network Topology
* `"layers"`
    * layers list of the CNN, including the output layer
    * `"type"`
        * domain is {"ConvLayer", "FullyLayer", "DropOut", "MaxPooling",
        "AveragePooling", "LocalResponseNormalization", "SingleMachineOutput",
        "LSTMSingleMachine", "RNNSingleMachine", "EIIE_Dense", "EIIE_Output_WithW"}
    * `"filter shape"`
        * shape of the filter (kernel) of the Convolutional Layer
* `"input"`
    * `"window_size"`
        * number of columns of the input matrix
    * `"coin_number"`
        * number of rows of the input matrix
    * `"feature_number"`
        * number of features (just like RGB in computer vision)
        * domain is {1, 2, 3}
        * 1 means the feature is ["close"], last price of each period
        * 2 means the feature is ["close", "volume"]
        * 3 means the features are ["close", "high", "low"]

### Market Data
* `"input "`
    * `"start_date"`
        * start date of the global data matrix
        * format is YYYY/MM/DD
    * `"end_date"`
        * start date of the global data matrix
        * format is YYYY/MM/DD
        * Performance can vary a lot in different time ranges.
    * `"volume_average_days"`
        * number of days of volume used to select the coins
    * `"test_portion"`
        * portion of backtest data, ranging from 0 to 1. Example: 0.08 means that the initial 92% of the global data matrix is used for training and the following 8% is used for testing. This version of the library does not allow for separate validation and test periods.
    * `"global_period"`
        * trading period and period of prices in input window, i.e. duration of each candlestick.
        * should be a multiple of 300 (seconds). Default value is 1800 i.e. half an hour.
    * `"coin_number"`
        * number of assets to be traded.
        * does not include cash (i.e. btc)
    * `"online"`
        * `true`: new data is retrieved from the exchange and stored in the local database
        * `false`: coin selection and input data is generated from the local database.


### Training
* `"training"`
    * training hyperparameters
    * `"steps"`
        * the total number of steps performed during training
    * `"learning_rate"`
        * learning rate for the gradient descent
    * `"batch_size"`
        * TODO
    * `"buffer_biased"`
        * TODO
    * `"snap_shot"`
        * TODO
    * `"fast_train"`
        * TODO
    * `"training_method"`
        * Optimizer used to minimize the loss
    * `"loss_function"`
        * Loss function

### Trading
* `"trading"`
    * `"trading_consumption"`
        * TODO
    * `"rolling_training_steps"`
        * TODO
    * `"learning_rate"`
        * TODO
    * `"buffer_biased"`
        * TODO

## Training the agent
In order to train the agent perform the following steps:
1. _(Optional)_ Modify the configuration in  `pgportfolio/net_config.json` according to your desired agent configuration.

2. From the main folder, run:
```
python main.py --mode=generate --repeat=n
```
where `n` is a positive integer indicating the number of replicas you would like to train.
This will create `n` subfolders in the `train_package` folder. Each subfolder contains a copy of the `net_config.json` file.
The random seed of each the subfolder runs from `0` to `n-1`. _*Please note that agents with different random seeds can have very different performances.*_

3. (Optional) Download the data with the command:
```
python main.py --mode=download_data
```

4. Train your agents with the command:
```
python main.py --mode=train --processes=1
```
    * This will start training the `n` agents one at a time. Do not start more than 1 processes if you want to download data online.
    * `--processes=m` starts `m` parallel training processes
    * `--device=gpu` can be added if your tensorflow supports GPU.
      * On _GTX1080Ti_ you should be able to run 4-5 training process simultaneously.
      * On _GTX1060_ you should be able to run 2-3 training simultaneously.

5. Each training run is composed of 2 phases: **Training** and **Backtest**.

    * During the **Training** phase, the agent is trained on the training fraction of the global data matrix. The log looks like this:

  ```
  average time for data accessing is 0.0015480489730834962
  average time for training is 0.009850282192230225
  ==============================
  step 2000
  ------------------------------
  the portfolio value on test set is 2.118205
  log_mean is 0.00027037683
  loss_value is -0.000270
  log mean without commission fee is 0.000341
  ```

  * After training is completed, the **Backtest** phase begins. This uses a rolling training window, i.e. it performs online learning in supervised learning. The log looks like this:

  ```
  the step is 536
  total assets are 4.314677 BTC
  ```

6. Once training and backtest are completed, you can check the result summary of the training in `train_package/train_summary.csv`

7. Tune the hyper-parameters based on the summary, and go to 1 again.

## Training results
Once training is completed, each subfolder in `train_package` will contain several output artifacts:

* `programlog`: a log file generated during training. This contains the same information that was visualized in output during training and backtesting.

* `tensorboard`: a folder containing the events for thensorboard. You can visualize its content by running tensorboard: e.g. `tensorboard --logdir=train_package/1`.

* ` netfile.*`: the model checkpoints. These can be used to restore a previously trained model.

* `train_summary.csv`: a file with summary information like: network configuration, portfolio value on validation set and test set etc.


## Download Data
To prefetch data to the local database without starting a training run:

```
python main.py --mode=download_data
```

The program will use the configurations in `pgportfolio/net_config.json` to select coins and download necessary data to train the network.
* Download speed could be very slow and sometimes even have errors in China.
* If you can cannot download data, please check the first release where there is a `Data.db` file. Copy the file into the database folder. Make sure the `online` in `input` in `net_config.json` to be `false` and run the example. Note that using the this file, you shouldn't make any changes to input data configuration (for example `start_date`, `end_date` or `coin_number`) otherwise the results may not be correct.


## Backtest
To execute backtest with rolling training (i.e. online learning in supervised learning) on the target model run:

```
python main.py --mode=backtest --algo=1
```

* `--algo` could be either the name of traditional method or the index of the training folder

## Traditional algorithms
The library contains the implementation of the following traditional algorithms and benchmarks:

#### Benchmarks
* `ubah`: Uniform Buy And Hold.
* `best`: Buy the best stock in hindsight.
* `crp`: Constant Rebalanced Portfolios. Rebalances to a preset portfolio at the beginning of every period.
* `bcrp`: Best Constant Rebalanced Portfolio. Sets the portfolio as the portfolio that maximizes the terminal wealth in hindsight.

#### Follow the Winner traditional algorithms
Transfer portfolio weights from the underperforming assets (experts) to the outperforming ones.
* `up`: Universal Portfolios (Cover, 1991). Uniformly buys and holds the whole set of CRP experts.
* `eg`: Exponential Gradient. Tracks the best stock and adopts regularization term to constrain the deviation from previous portfolio.
* `ons`: Online Newton Step (Agarwal et al., 2006). Tracks the best CRP to date and adopts a L2-norm regularization to constrain portfolio’s variability.

#### Follow the Loser traditional algorithms
Assume underperforming assets will revert, move portfolio weights from the outperforming assets to the underperforming assets.
* `anticor1`: Anti Correlation (Borodin et al., 2004). Transfers the wealth from the outperforming stocks to the underperforming stocks via their cross-correlation and auto-correlation
* `anticor2`: Variation of the above.
* `pamr`: Passive Aggressive Mean Reversio (Li et al., 2012). Explicitly track the worst stocks, while adopting regularization techniques to constrain the deviation from last portfolio.
* `cwmr_var`: Confidence Weighted Mean Reversion (Li et al., 2013). Models the portfolio vector with a Gaussian distribution, and explicitly updates the distribution following the mean reversion principle.
* `cwmr_std`: Variation of the above.
* `olmar`: Online Moving Average Reversion. Explicitly predicts next price relatives following the mean reversion idea. Uses simple moving average.
* `olmar2`: Online Moving Average Reversion. Explicitly predicts next price relatives following the mean reversion idea. Uses exponentially weighted moving average.

#### Pattern Matching based Approaches
Based on the assumption that market sequences with similar preceding market appearances tend to re-appear.

* `bk`: Nonparametric kernel-based sample selection (Györfi et al., 2006). Identifies the similarity set by comparing two market windows via Euclidean distance.
* `bnn`: Nonparametric nearest neighbor-based sample selection (Györfi et al., 2008). Searches the price relatives whose preceding market windows are within the _l_ nearest neighbor of latest market window in terms of Euclidean distance.
* `cornk`: Correlation-driven nonparametric sample selection (Li et al., 2011). Identifies the similarity
among two market windows via correlation coefficient.
* `cornu`: Variation of the above.

#### Other strategies
* `m0`: TODO
* `rmr`: TODO
* `sp`: TODO
* `wmamr`: TODO

For more information see the [OLPS toolbox manual](http://www.mysmu.edu.sg/faculty/chhoi/olps/OLPS_toolbox_manual.pdf).

## Plotting
To plot the results run:

```
python main.py --mode=plot --algos=crp,olmar,1 --labels=crp,olmar,nnagent
```

* `--algos`: comma separated list of traditional algorithms and agent indexes
* `--labels`: comma separated list of names that appear in the plot legend

Example result plot:
![](http://static.zybuluo.com/rooftrellen/u75egf9roy9c2sju48v6uu6o/result.png)

## Table summary
You can present a summary of the results typing:

```
python main.py --mode=table --algos=1,olmar,ons --labels=nntrader,olmar,ons
```

* `--algos` and `--labels` are the same as in plotting case. Labels indicate the row indexes. The result table looks like this:

```
           average  max drawdown  negative day  negative periods  negative week  portfolio value  positive periods  postive day  postive week  sharpe ratio
nntrader  1.001311      0.225874           781              1378            114        25.022516              1398         1995          2662      0.074854
olmar     1.000752      0.604886          1339              1451           1217         4.392879              1319         1437          1559      0.035867
ons       1.000231      0.217216          1144              1360            731         1.770931              1416         1632          2045      0.032605

```
* use `--format` arguments to change the format of the table,
 could be `raw` `html` `csv` or `latex`. The default one is raw.
