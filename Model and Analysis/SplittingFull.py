import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# Define the columns to normalize
cols_to_normalize = ['xrsa_flux', 'xrsb_flux']

# Load in the data
df = pd.read_csv('/Users/raymondblahajr/Desktop/PDS2.0/goes_13to18.csv', low_memory=False)

# Convert the 'time' column to datetime and remove timezone info
df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)

# Normalize data using MinMaxScaler
scaler = MinMaxScaler()
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# Save the scaler to a pickle file
with open("/Users/raymondblahajr/Desktop/PDS2.0/scaler_Da_One.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Split data into training, validation, and test sets
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2  # This is implicitly determined as (1 - train_ratio - val_ratio)

train_size = int(train_ratio * len(df))
val_size = int(val_ratio * len(df))

df_train = df.iloc[:train_size]
df_val = df.iloc[train_size:train_size + val_size]
df_test = df.iloc[train_size + val_size:]

# Save train, validation, and test csvs
df_train.to_csv("/Users/raymondblahajr/Desktop/PDS2.0/train_datafull.csv", index=False)
df_val.to_csv("/Users/raymondblahajr/Desktop/PDS2.0/val_datafull.csv", index=False)
df_test.to_csv("/Users/raymondblahajr/Desktop/PDS2.0/test_datafull.csv", index=False)
