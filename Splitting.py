import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# Define the columns to normalize
cols_to_normalize = ['xrsa_flux', 'xrsb_flux', 'xrsa_cumsum', 'xrsb_cumsum']

# Load in the data
df = pd.read_csv('/Users/raymondblahajr/Desktop/PDS/Machine_Learning_Stuff/goes16.csv', low_memory=False)

# Convert the 'time' column to datetime and remove timezone info
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)

# Group by day and compute the cumulative sum for 'xrsa' and 'xrsb' columns within each group
df['date'] = df['time'].dt.date
df['xrsa_cumsum'] = df.groupby('date')['xrsa_flux'].cumsum()
df['xrsb_cumsum'] = df.groupby('date')['xrsb_flux'].cumsum()


def status_to_numeric(status):
    mapping = {np.nan: 0, '': 0, 'EVENT_PEAK': 1}
    return mapping.get(status, 0)

df['status'] = df['status'].apply(status_to_numeric)

# Normalize data using MinMaxScaler
scaler = MinMaxScaler()
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# Save the scaler to a pickle file
with open("/Users/raymondblahajr/Desktop/PDS/Machine_Learning_Stuff/scaler_Da_One.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Drop the date column
df.drop(columns=['date'], inplace=True)

# Train-test split
train_ratio = 0.8
train_size = int(train_ratio * len(df))
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

# Save train and test csvs
df_train.to_csv("/Users/raymondblahajr/Desktop/PDS/Machine_Learning_Stuff/train_data.csv", index=False)
df_test.to_csv("/Users/raymondblahajr/Desktop/PDS/Machine_Learning_Stuff/test_data.csv", index=False)

