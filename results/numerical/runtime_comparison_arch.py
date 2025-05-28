import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import json
import glob

# load the runtime_exp_1.csv file 
df = pd.read_csv("/home/luka/repo/JAXOvercooked/results/numerical/runtime_exp_1.csv")

print(df.head())

# average over the runtime_1 and runtime_2 columns
df['avg_runtime'] = (df['runtime_1'] + df['runtime_2']) / 2
# add the average runtime to the DataFrame
df['avg_runtime'] = df['avg_runtime'].astype(float)
# print the DataFrame
print(df.head())

# split the data on cnn= true and cnn=false
means = df.groupby('cnn')['avg_runtime'].agg(['mean','std'])
print(means)


# group the data by multihead
means_multihead = df.groupby('multihead')['avg_runtime'].agg(['mean', 'std'])
print(means_multihead)
# group the data by shared
means_shared = df.groupby('shared_backbone')['avg_runtime'].agg(['mean', 'std'])
print(means_shared)
# group the data by task_id
means_task_id = df.groupby('task_id')['avg_runtime'].agg(['mean', 'std'])
print(means_task_id)