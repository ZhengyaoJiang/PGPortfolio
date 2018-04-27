## Traditional algorithms
The library contains the implementation of the following traditional algorithms and benchmarks:

#### Benchmarks
* `ubah`: Uniform Buy And Hold. Equally spread the total fund into the preselected assets and hold them without making any purchases or selling until the end.
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
* `m0`: Markov of order zero [(Borodin et al., 2000)](https://pdfs.semanticscholar.org/5693/0f8457aa5e612db8f25d2b0d2f8a989344a5.pdf).
* `rmr`: Robust Median Reversion [(Huang et al., 2013)](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=3326). Exploit median reversion
via robust L1-estimator and Passive Aggressive online learning.
* `sp`: TODO
* `wmamr`: Weighted Moving Average Mean Reversion [(Gao and Zhang, 2013)](https://ieeexplore.ieee.org/document/6643896/)

For more information see the [OLPS toolbox manual](http://www.mysmu.edu.sg/faculty/chhoi/olps/OLPS_toolbox_manual.pdf).
