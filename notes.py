import pandas as pd
import quandl

start = pd.to_datetime('2012-01-01')
end = pd.to_datetime('2017-01-01')

aapl = quandl.get('WIKI/AAPL.11', start_date = start, end_date = end)
cisco = quandl.get('WIKI/CSCO.11', start_date = start, end_date = end)
ibm = quandl.get('WIKI/IBM.11', start_date = start, end_date = end)
amzn = quandl.get('WIKI/AMZN.11', start_date = start, end_date = end)

aapl.iloc[0]['Adj. Close']

for stock_df in (aapl,cisco,ibm,amzn):
    stock_df['Normed Return'] = stock_df['Adj. Close'] / stock_df.iloc[0]['Adj. Close']

aapl.head()
aapl.tail()

#30% in AAPL
#20% in cisco
#40% in amazon
#10% in ibm

for stock_df, allo in zip((aapl,cisco,ibm,amzn), [.3, .2, .4, .1]):
    stock_df['Allocation'] = stock_df['Normed Return'] * allo

aapl.head()

for stock_df in (aapl,cisco,ibm,amzn):
    stock_df['Position Values'] = stock_df['Allocation'] * 1000000

aapl.head()


all_pos_vals = [aapl['Position Values'], cisco['Position Values'], ibm['Position Values'], amzn['Position Values'],]
portfolio_val = pd.concat(all_pos_vals, axis = 1)
portfolio_val.columns = ['AAPL Pos.', 'CISCO Pos.', 'IBM Pos.', 'AMZN Pos.',]
portfolio_val.head()

portfolio_val['Total Pos'] = portfolio_val.sum(axis = 1)
portfolio_val.head()

import matplotlib.pyplot as plt
portfolio_val['Total Pos'].plot(figsize = (10, 8))

portfolio_val.drop('Total Pos', axis = 1).plot(figsize = (10, 8))

portfolio_val.head()

portfolio_val['Daily Return'] = portfolio_val['Total Pos'].pct_change(1)
portfolio_val.head()
portfolio_val['Daily Return'].mean()
portfolio_val['Daily Return'].std()
portfolio_val['Daily Return'].plot(kind = 'hist', bins = 100, figsize = (4, 5))
portfolio_val['Daily Return'].plot(kind = 'kde',figsize = (4, 5))
cumulative_return = 100 * (portfolio_val['Total Pos'][-1] / portfolio_val['Total Pos'][0] - 1)
cumulative_return
portfolio_val['Total Pos'][-1]
SR = portfolio_val['Daily Return'].mean() / portfolio_val['Daily Return'].std()
SR

ASR = (252**0.5) * SR
ASR
#Annualized sharp ratio value of 1 or greater is considered acceptable to good
# 2 or greater is very good
# 3 or higher is excellent

# -=-=-=-=- PORTFOLIO OPTIMIZATION -=-=-=-=-=-=
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

aapl = pd.read_csv('AAPL_CLOSE', index_col = 'Date', parse_dates = True)
cisco = pd.read_csv('CISCO_CLOSE', index_col = 'Date', parse_dates = True)
ibm = pd.read_csv('IBM_CLOSE', index_col = 'Date', parse_dates = True)
amzn = pd.read_csv('AMZN_CLOSE', index_col = 'Date', parse_dates = True)

stocks = pd.concat([aapl, cisco, ibm, amzn], axis = 1)
stocks.columns = ['aapl', 'cisco', 'ibm', 'amzn']
stocks.head()

#calculate mean daily return
stocks.pct_change(1).mean()
#find correlation between stocks using pearson correlation coefficent
stocks.pct_change(1).corr()
#This calculates Pearson Correlation Coefficient
#-----------------------------------------------------
#logarithmic returns vs arithmetic returns
#We will use Log returns more because it allows us to normalize but heres a comparison
#Arithmetically
stocks.pct_change(1).head()
#logarithmically
log_ret = np.log(stocks / stocks.shift(1))
log_ret.head()
# youll see the numbers themselves are really similar but for more complex operations log is preferred
# moving on
log_ret.hist(bins =  100, figsize = (12, 8))
plt.tight_layout()

#this give us average or mean return of stock
log_ret.mean()
#find covariance
#252 business days
log_ret.cov() * 252

print(stocks.columns)

weights = np.array(np.random.random(4))

print("Random Weights: ")
print(weights)
#gotta make sure the weights add up to 100 so:
print('Rebalance')
weights = weights / np.sum(weights)
print(weights)
#so we get the same random numbers every time
np.random.seed(101)

# Expected Return
print('Expected Portfolio Return')
exp_ret = np.sum( (log_ret.mean() * weights) * 252)
print(exp_ret)
np.sum(log_ret.mean() * weights * 252)

#Expected Variance / Volatility
print('Expected Volatility')
exp_vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
print(exp_vol)
#Sharpe ratio
print('Sharpe Ratio')
SR = exp_ret / exp_vol
print(SR)


#=-=-=-=-=-=-==-=-PORTFOLIO ALLOCATION PT 2 (same as pt 1 but with for loop for efficiency)==========----------===============-----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(101)

num_ports = 5000
all_weights = np.zeros((num_ports, len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):
    #Weights
    weights = np.array(np.random.random(4))
    weights = weights / np.sum(weights)

    #Save weights
    all_weights[ind,:] = weights

    #Expected Return
    ret_arr[ind] = np.sum( (log_ret.mean() * weights) * 252)

    #Expected Volatility
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

    #Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]

#check the maximum value the sharpe array can possibly have after testing with FOR loop
sharpe_arr.max()

#Check the index location of that max so we can see the best
sharpe_arr.argmax()

#Find out the best portfolio allocations by calling the index locations
all_weights[1420,:]
#Now we know the optimal allocation of our money in our listed securities

#now lets visualize the data
plt.figure(figsize = (12, 8))
plt.scatter(vol_arr, ret_arr, c = sharpe_arr, cmap = 'plasma')
plt.colorbar(label = 'Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

#we plot the point of the best portfolio allocation so we can see it on the plot
max_sr_ret = ret_arr[1420]
max_sr_vol = vol_arr[1420]
plt.scatter(max_sr_vol, max_sr_ret, c = 'red', s = 50, edgecolors = 'black')
#now we can find and visualize the best portfolio allocations YAHASHUYIDIGAPIFHUA
#this is based off of 5000 random portfolio allocations, next is optimization

#******************************************************************************
#-=-=-=-=-=-=-=-=-PORTFOLIO ALLOCATION PART 3 (OPTIMIZATION WITH MATH BABAY)
#******************************************************************************
def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret, vol, sr])

from scipy.optimize import minimize
#--=-=-=-=-=-==-gives us a guide on how to use scipy-==--=-=-=-=-=
#help(minimize)

#Takes in a weight allocation and then returns the negative sharpe ratio
def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * -1

#help optimization and minimization using constraints because they will allow there to be less stuff to Check
def check_sum(weights):
    #this is going to return 0 if the sum of the weights is 1
    return np.sum(weights) - 1
    # if not it will return how off you are by 1 itself

cons = ({'type' : 'eq', 'fun' : check_sum})
bounds = ((0,1), (0,1), (0,1), (0,1))

init_guess = [0.25, 0.25, 0.25, 0.25]

opt_results = minimize(neg_sharpe, init_guess, method = 'SLSQP', bounds = bounds, constraints = cons)
opt_results.x
get_ret_vol_sr(opt_results.x)

#finding the lowest risk possible for given level of return (efficiency frontier)

frontier_y = np.linspace(0, 0.3, 100)
def minimize_volatility(weights):
    return get_ret_vol_sr(weights)[1]

frontier_volatility = []

for possible_return in frontier_y:
    cons = ({'type' : 'eq', 'fun' : check_sum},
            {'type' : 'eq', 'fun' : lambda w: get_ret_vol_sr(w)[0] - possible_return})

    result = minimize(minimize_volatility, init_guess, method = 'SLSQP', bounds = bounds, constraints = cons)

    frontier_volatility.append(result['fun'])

plt.figure(figsize = (12, 8))
plt.scatter(vol_arr, ret_arr, c = sharpe_arr, cmap = 'plasma')
plt.colorbar(label = 'Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

plt.plot(frontier_volatility, frontier_y, 'g--', linewidth = 3)
# Now we have a line that tells us the best values for return given volatility and vice versa
#This process is known as marcowitz portfolio optimization.
#
#
#
#
#==============================FINANCIAL MARKET CONCEPTS=============================
#
#
#
#

#^^^^^^^^^^^^^^^^^^^^^^^^^^CAPM CODE^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
from scipy import stats
# help(stats.linregress)
import pandas as pd
import pandas_datareader as web

start = pd.to_datetime('2010-01-04')
end = pd.to_datetime('2017-07-25')

spy_etf = web.DataReader('SPY','yahoo', start, end)
spy_etf.info()
spy_etf.head()

aapl = web.DataReader('AAPL', 'yahoo', start, end)
aapl.head()

import matplotlib.pyplot as plt

aapl['Close'].plot(label = 'AAPL', figsize = (10,8))
spy_etf['Close'].plot(label = 'SPY Index')
plt.legend()

aapl['Cumulative'] = aapl['Close'] / aapl['Close'].iloc[0]
spy_etf['Cumulative'] = spy_etf['Close'] / spy_etf['Close'].iloc[0]

#This graph shows how much I would have made in 2017 had I invested a dollar into
#Each security before 2011.
aapl['Cumulative'].plot(label = 'AAPL', figsize = (10, 8))
spy_etf['Cumulative'].plot(label = 'SPY')
plt.legend()


aapl['Daily Return'] = aapl['Close'].pct_change(1)
spy_etf['Daily Return'] = spy_etf['Close'].pct_change(1)
plt.scatter(aapl["Daily Return"], spy_etf['Daily Return'], alpha = 0.25)

#using tuple unpacking to get some values
beta,alpha,r_value,p_value,std_err = stats.linregress(aapl['Daily Return'].iloc[1:],
                                                      spy_etf['Daily Return'].iloc[1:])
#Now we can call Beta and Alpha by itself

beta

alpha

r_value

#if our stock is in line with the market you will expect a high Beta value
spy_etf['Daily Return'].head()

import numpy as np
#Now we create some noise
#use random.normal so we can set paramaters like mean
noise = np.random.normal(0, 0.001, len(spy_etf['Daily Return'].iloc[1:]))

noise

#we make a fake stock as a way to demonstrate high Beta. This stock will follow SPY trends
fake_stock = spy_etf['Daily Return'].iloc[1:] + noise
plt.scatter(fake_stock, spy_etf['Daily Return'].iloc[1:], alpha = 0.25)
#Now we see obvious linear behavior and clear relationship
#Now we use CAPM to prove it and to prove CAPM works
beta,alpha,r_value,p_value,std_err = stats.linregress(fake_stock,
                                                      spy_etf['Daily Return'].iloc[1:])
#now we check beta value
beta
#very close to 1 making it very correlated, CAPM works

#also means alpha should be very small
alpha
#note the euler number showing its tiny

#NOTE ALWAYS USE ADJ CLOSE AND OPEN NEVER THE PURE ONES
#you want a survivor bias free data set (quandl premium has this)
