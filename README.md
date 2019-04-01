# stock-trader
## Introduction
The goal of this project is to create a stock trader capable of learning from the market variables, generating (buy, sell, sit) actions, and evaluating the performance of itself. The tasks involved are as the follows:
- Fetch and preprocess the historical stock data from yahoo!finance using the 'pandas_datareader' package
- Train a trader that can decide which action to take given the current stock market environment
- Evaluate performance of the trader

Metric to evaluate the trader performance:
profitability = (total capital at the end of trading - invested capital)/invested capital
The benchmark for the trader, I used the simple scenario where buy at the first day and sell at the last day of the given data set to compute the profitability of a given stock. The benchmark profitability of the train data is 106.8%.

## Files
- data_process.py: contains all the data preprocessing, such as create financial stock indicators Bollinger band width and close price to SMA ratio
- stock-trader.ipynb: implemented Q-learning algorithm to train on apple stock and tune training parameters to achieve the best profitability on the test data.


## Results
For the newest 20% of the stock data for Apple(APPL) from 2014/01/01 to 2018/01/01, the model achieves a profitability of 755.28% on the train data set, and 56.11% on the test data set.
<img src="https://github.com/nyxcat/stock-trader/blob/master/Figures/apple_test.png">

For the short-term Google stock from 2017/11/01 to 2017/12/01, the model yields a return/invest ratio of -0.0097, which is a profitability of -0.97%, which is higher than the benchmark return/invest ratio of 2.16%.
<img src="https://github.com/nyxcat/stock-trader/blob/master/Figures/short-term-google.png">

For th long-term Google stock from 2018/11/01 to 2018/12/01, the model yields a profitability of 34.39%, which is much higher than the benchmark return/invest ratio of 10.51%.
<img src="https://github.com/nyxcat/stock-trader/blob/master/Figures/longterm-google.png">

## Requirements
python 3.6.7
numpy 1.15.4
pandas 0.23.4
matplotlib 3.0.2
pandas_datareader 0.7.0 
data_process
