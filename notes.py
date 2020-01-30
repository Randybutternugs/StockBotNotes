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
