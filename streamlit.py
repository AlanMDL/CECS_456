import hdbscan
import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans # for KMeans algorithm
from kneed import KneeLocator # To find elbow point
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import streamlit as st
data = pd.read_csv("SouthGermanCredit.csv")
import plotly.express as px
import warnings

# These featrues are chosen due to being qualitive features of the CSV file
# Outliers sorted based on amount 
# Can compare other features to these qualtiive features such as credit amount vs risk etc...... 

warnings.filterwarnings('ignore')
# generating statistics of data
data.describe()[['age','amount','duration']]
# Plot distribution plot for features

data.sample(5)

plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
sns.distplot(data['amount'])
plt.subplot(1,3,2)
sns.distplot(data['duration'])
plt.subplot(1,3,3)
sns.distplot(data['age'])
plt.show()

# Due to all features having a skewed disbruition, z-score treatment is not ideal
# IQR is a better approach

print("Old Shape: ", data.shape)

q_low = data["amount"].quantile(0.05)
q_hi  = data["amount"].quantile(0.95)

data_filtered = data[(data['amount'] < q_hi) & (data['amount'] > q_low)]


print("New Shape: ", data_filtered.shape)

plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
sns.distplot(data_filtered['amount'])
plt.subplot(1,3,2)
sns.distplot(data_filtered['duration'])
plt.subplot(1,3,3)
sns.distplot(data_filtered['age'])
plt.show()

# Drops around 100 rows of data
# Maybe only drop top 5% of data due to bottom 5% being prominent to data and patterns
