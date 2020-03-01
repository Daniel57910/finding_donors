import numpy as np
import pandas as pd
from time import time
from IPython.display import display
# import visuals as vs
import os
from sklearn.preprocessing import MinMaxScaler
from time import time

FILE_NAME = os.path.join(os.getcwd(), 'census.csv')

data = pd.read_csv(FILE_NAME)

over_50k = data.loc[data['income'] == '>50K']
under_50k = data.loc[data['income'] != '>50K']

n_records = len(data)
n_over_50k = len(over_50k)
n_under_50k = len(under_50k)
percentage_50k = n_over_50k / (n_over_50k + n_under_50k)

print(f'Number of record => {n_records}')
print(f'Number of earners over 50k => {n_over_50k}')
print(f'Number of earners under 50k => {n_under_50k}')
print(f'Ratio => {percentage_50k}')


# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

print(int(time()))