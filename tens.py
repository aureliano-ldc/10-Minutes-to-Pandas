"""
http://pandas.pydata.org/pandas-docs/stable/10min.html
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Object Creation"""
# Creating a Series by passing a list of values, letting pandas create a default integer index:
s = pd.Series([1, 3, 5, np.nan, 6, 8])

# Creating a DataFrame by passing a numpy array, with a datetime index and labeled columns:
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

# Creating a DataFrame by passing a dict of objects that can be converted to series-like:
df2 = pd.DataFrame({'A' : 1.,
					'B' : pd.Timestamp('20130102'),
					'C' : pd.Series(1, index=list(range(4)), dtype='float32'),
					'D' : np.array([3] * 4, dtype='int32'),
					'E' : pd.Categorical(["test", "train", "test", "train"]),
					'F' : 'foo'})

# Having specific dtypes:
df2.dtypes


"""Viewing Data"""
# See the top & bottom rows of the frame:
df.head()
df.tail()
df.head(1)
df.tail(3)

# Display the index, columns, and the underlying numpy data:
df.index
df.columns
df.values

# Describe shows a quick statistic summary of your data:
df.describe()

# Transposing your data
df.T

# Sorting by an axis:
df.sort_index(axis=0, ascending=False)
df.sort_index(axis=1, ascending=False)

# Sorting by values:
df.sort_values(by='B')
df.sort_values(by='B', ascending=False)


"""Selection"""
# Selecting a single column, which yields a Series, equivalent to df.A
df['A']

# Selecting via [], which slices the rows.
df[0:3]
df['20130102':'20130104']

# For getting a cross section using a label
df.loc[dates[0]]

# Selecting on a multi-axis by label
df.loc[:,['A','B']]

# Showing label slicing, both endpoints are included
df.loc['20130102':'20130104',['A', 'B']]

# For getting fast access to a scalar
df.at[dates[0],'A']

# Select via the position of the passed integers
df.iloc[3]

# By integer slices, acting similar to numpy/python
df.iloc[3:5,0:2]

# By lists of integer position locations, similar to the numpy/python style
df.iloc[[1,2,4],[0,2]]

# For getting a value explicitly
df.iloc[1,1]

# For getting fast access to a scalar (equiv to the prior method)
df.iat[1,1]

# Using a single column’s values to select data.
df[df.A > 0]

# A where operation for getting.
df[df > 0]

# Using the isin() method for filtering:
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
df2[df2['E'].isin(['two', 'four'])]

# Setting a new column automatically aligns the data by the indexes
s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
df['F'] = s1

# Setting values by label
df.at[dates[0], 'A'] = 0

# Setting values by position
df.iat[0, 1] = 0

# Setting by assigning with a numpy array
df.loc[:, 'D'] = np.array([5] * len(df))

# A where operation with setting.
df2 = df.copy()
df2[df2 > 0] = -df2


"""Missing Data"""
# Reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of the data.
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1], 'E'] = 1

# To drop any rows that have missing data.
df1.dropna(how='any')

# Filling missing data
df1.fillna(value=5)

# To get the boolean mask where values are nan
pd.isnull(df1)


"""Operations"""
# Performing a descriptive statistic
df.mean()

# Same operation on the other axis
df.mean(1)

# Operating with objects that have different dimensionality and need alignment. In addition, pandas automatically broadcasts along the specified dimension.
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=dates).shift(2)
df.sub(s, axis='index')

# Applying functions to the data
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())

# Histogramming
s = pd.Series(np.random.randint(1, 7, size=10))
s.value_counts()

# String Methods
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s.str.lower()


"""Merge"""
# Concatenating pandas objects together with concat():
df = pd.DataFrame(np.random.randn(10, 4))
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)

# SQL style merges
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
pd.merge(left, right, on='key')
# Another example
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
pd.merge(left, right, on='key')

# Appemd
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
s = df.iloc[3]
df.append(s, ignore_index=True)


"""Grouping"""
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
							'foo', 'bar', 'foo', 'foo'],
					'B' : ['one', 'one', 'two', 'three',
							'two', 'two', 'one', 'three'],
					'C' : np.random.randn(8),
					'D' : np.random.randn(8)})

# Grouping and then applying a function sum to the resulting groups.
df.groupby('A').sum()

# Grouping by multiple columns forms a hierarchical index, which we then apply the function.
df.groupby(['A','B']).sum()


"""Reshaping"""
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz',
					'foo', 'foo', 'qux', 'qux'],
					['one', 'two', 'one', 'two',
					 'one', 'two', 'one', 'two']]))

"""
example:
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
xyz = list(zip(x, y, z))
xyz
运行的结果是：
[(1, 4, 7), (2, 5, 8), (3, 6, 9)]

list(zip(*xyz))
运行的结果是：
[(1, 2, 3), (4, 5, 6), (7, 8, 9)]
一般认为这是一个unzip的过程
"""

index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]

# The stack() method “compresses” a level in the DataFrame’s columns.
stacked = df2.stack()

# the inverse operation of stack() is unstack()
stacked.unstack()
stacked.unstack(1)
stacked.unstack(0)

# Pivot Tables
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,
					'B' : ['A', 'B', 'C'] * 4,
					'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
					'D' : np.random.randn(12),'E' : np.random.randn(12)})

pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])


"""Time Series"""
# performing resampling operations during frequency conversion
rng = pd.date_range('1/1/2012', periods=100, freq='S')
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min').sum()

# Time zone representation
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
ts_utc = ts.tz_localize('UTC')
ts_utc.tz_convert('US/Eastern')

# Converting between time span representations
rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ps = ts.to_period()
ps.to_timestamp()

"""
Converting between period and timestamp enables some convenient arithmetic functions to be used. In the following 
example, we convert a quarterly frequency with year ending in November to 9am of the end of the month following the 
quarter end:
"""
 prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
 ts = pd.Series(np.random.randn(len(prng)), prng)
 ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
 ts.head


 """Categoricals"""
df = pd.DataFrame({"id":[1,2,3,4,5,6],
					"raw_grade":['a', 'b', 'b', 'a', 'a', 'e']})

df["grade"] = df["raw_grade"].astype("category")

# Rename the categories to more meaningful names
df["grade"].cat.categories = ["very good", "good", "very bad"]

# Reorder the categories and simultaneously add the missing categories 
df["grade"] = df["grade"].cat.set_categories(["very bad", "bad", "medium", "good", "very good"])

# Sorting is per order in the categories, not lexical order.
df.sort_values(by="grade")

# Grouping by a categorical column shows also empty categories.
df.groupby("grade").size()


"""Plotting"""
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
plt.show()

# On DataFrame, plot() is a convenience to plot all of the columns with labels:
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, 
					columns=['A', 'B', 'C', 'D'])

df = df.cumsum()
plt.figure()
df.plot()
plt.legend(loc='best')


"""Getting Data In/Out"""
# Writing to a csv file
df.to_csv('foo.csv')

# Reading from a csv file
pd.read_csv('foo.csv')

# Writing to a HDF5 Store
df.to_hdf('foo.h5','df')

# Reading from a HDF5 Store
pd.read_hdf('foo.h5','df')

# Writing to an excel file
df.to_excel('foo.xlsx', sheet_name='Sheet1')

# Reading from an excel file
pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])