import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained autoencoder model
autoencoder = load_model('autoencoder_model2.h5')

# Load the validation data
df = pd.read_csv('/Users/raymondblahajr/Desktop/PDS/Machine_Learning_Stuff/test_data.csv', low_memory=False)

norm_cols = ['xrsa_flux', 'xrsb_flux', 'xrsa_cumsum', 'xrsb_cumsum']
SEQUENCE_LENGTH = 1440
BATCH_SIZE = 128

def sequence_generator(df, batch_size=BATCH_SIZE):
    while True:
        for day_start in range(0, len(df) - SEQUENCE_LENGTH, SEQUENCE_LENGTH * batch_size):
            sequences = []
            for batch_num in range(batch_size):
                start = day_start + (batch_num * SEQUENCE_LENGTH)
                end = start + SEQUENCE_LENGTH
                if end <= len(df):
                    sequences.append(df.iloc[start:end][norm_cols].values)
            if len(sequences) == batch_size:
                yield np.array(sequences)

# Predict on the validation data using the autoencoder
test_steps = (len(df) - SEQUENCE_LENGTH) // (SEQUENCE_LENGTH * BATCH_SIZE)
reconstructed = autoencoder.predict(sequence_generator(df), steps=test_steps)

# Calculate the reconstruction error
val_reshaped = df[norm_cols].values[:BATCH_SIZE * test_steps * SEQUENCE_LENGTH]
reconstructed_reshaped = reconstructed.reshape(-1, len(norm_cols))
errors = np.mean(np.power(val_reshaped - reconstructed_reshaped, 2), axis=1)

# Define the anomaly threshold
threshold = np.percentile(errors, 99)

# Extract date and time for the corresponding indices
date_times = df['time'].values[:BATCH_SIZE * test_steps * SEQUENCE_LENGTH]

# Create a DataFrame to save predictions, errors, and anomaly status along with the date and time
result_df = pd.DataFrame({
    'DateTime': date_times,
    'Error': errors,
    'Is_Anomaly': errors > threshold
})

# Save the results to CSV
result_df.to_csv('/Users/raymondblahajr/Desktop/PDS/Machine_Learning_Stuff/predictions_with_datetime2.csv', index=False)

# Detect anomalies
anomaly_indices = np.where(errors > threshold)[0]

# Print anomalies with corresponding date and time
for idx in anomaly_indices:
    anomaly_time = df.iloc[idx]['time']  # Assuming 'time' column has the date-time information
    print(f"Anomaly detected at index {idx} corresponding to time: {anomaly_time}")
