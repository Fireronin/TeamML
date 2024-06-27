#%%
import pandas as pd
import numpy as np

file1 = '../stats/accuracies_per_percent_linear.csv'
file2 = '../stats/accuracies_per_percent_xgb.csv'
file3 = '../stats/gold_advantage_per_percent.csv'

# read csv files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3) 

# df3 rename correct to Accuracy
df3.rename(columns = {'correct':'Accuracy'}, inplace = True)



# add columns to dataframes
df1['Accuracy_linear'] = df1['Accuracy']
df1['Accuracy_xgb'] = df2['Accuracy']
df1['Accuracy_gold'] = df3['Accuracy']

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
x = np.arange(101)
ax.plot(x, df1['Accuracy_linear'], label='Linear Regression')
ax.plot(x, df1['Accuracy_xgb'], label='XGB')
ax.plot(x, df1['Accuracy_gold'], label='Gold Advantage')
ax.set(xlabel='Time Percentage', ylabel='Accuracy',
       title='Accuracy per Time Percentage')
ax.legend() 
ax.grid()

# %% same graph with plotly
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=df1['Accuracy_linear'], name='Linear Regression'))
fig.add_trace(go.Scatter(x=x, y=df1['Accuracy_xgb'], name='XGB'))
fig.add_trace(go.Scatter(x=x, y=df1['Accuracy_gold'], name='Gold Advantage'))
fig.update_layout(title='Accuracy per Time Percentage')
fig.show()


# %%
