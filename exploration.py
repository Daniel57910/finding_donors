import numpy as np
import pandas as pd
from time import time
from IPython.display import display
import os

FILE_NAME = os.path.join(os.getcwd(), 'census.csv')

census_data = pd.read_csv(FILE_NAME)

over_50k = census_data.loc[census_data['income'] == '>50K']
under_50k = census_data.loc[census_data['income'] != '>50K']

n_records = len(census_data)
n_over_50k = len(over_50k)
n_under_50k = len(under_50k)
percentage_50k = n_over_50k / (n_over_50k + n_under_50k)

print(f'Number of record => {n_records}')
print(f'Number of earners over 50k => {n_over_50k}')
print(f'Number of earners under 50k => {n_under_50k}')
print(f'Ratio => {percentage_50k}')




