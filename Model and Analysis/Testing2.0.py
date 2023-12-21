import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from scipy.signal import find_peaks

# Load the trained autoencoder model
autoencoder = load_model('autoencoder_model2.h5')

# Load the validation data
df = pd.read_csv('/Users/raymondblahajr/Desktop/PDS2.0/test_data.csv', low_memory=False)

norm_cols = ['xrsa_flux', 'xrsb_flux', 'xrsa_cumsum', 'xrsb_cumsum']
SEQUENCE_LENGTH = 1440
BATCH_SIZE = 128

def sequence_generator(df, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH):
    start_idx = 0
    while True:
        if start_idx + sequence_length >= len(df):
            remainder = len(df) - start_idx
            if remainder >= sequence_length:  # If there's enough data for at least one more sequence
                yield np.array([df.iloc[start_idx:start_idx+sequence_length][norm_cols].values])
            break
        sequences = []
        for _ in range(batch_size):
            end_idx = start_idx + sequence_length
            if end_idx > len(df):
                break
            sequences.append(df.iloc[start_idx:end_idx][norm_cols].values)
            start_idx += sequence_length
        if sequences:
            yield np.array(sequences)

# Update the steps to account for the last partial batch
total_sequences = (len(df) - len(df) % SEQUENCE_LENGTH) // SEQUENCE_LENGTH
test_steps = total_sequences // BATCH_SIZE
last_batch = total_sequences % BATCH_SIZE > 0  # Check if there is a last partial batch

# When predicting, handle the last batch separately if it's a partial batch
reconstructed = []
for i, batch in enumerate(sequence_generator(df)):
    if i < test_steps:
        reconstructed.append(autoencoder.predict(batch))
    elif last_batch and i == test_steps:  # This is the last partial batch
        reconstructed.append(autoencoder.predict(batch))
        break
reconstructed = np.concatenate(reconstructed, axis=0)


# Calculate the reconstruction error
sequences_in_reconstructed = reconstructed.shape[0] * SEQUENCE_LENGTH
val_reshaped = df[norm_cols].values[:sequences_in_reconstructed]
reconstructed_reshaped = reconstructed.reshape(-1, len(norm_cols))
errors = np.mean(np.power(val_reshaped - reconstructed_reshaped, 2), axis=1)

# Make sure the date_times array is the same length as the val_reshaped array
date_times = df['time'].values[:sequences_in_reconstructed]

# Define the anomaly threshold
threshold = np.percentile(errors, 95)

# Detect peaks
peaks, _ = find_peaks(errors, height=threshold)

# Create a DataFrame to save predictions, errors, and anomaly status along with the date and time
result_df = pd.DataFrame({
    'DateTime': date_times,
    'xrsa_flux': val_reshaped[:, 0],
    'xrsb_flux': val_reshaped[:, 1],
    'Error': errors,
    'Is_Peak': np.isin(np.arange(len(errors)), peaks)
})

# Save the results to CSV
result_df.to_csv('/Users/raymondblahajr/Desktop/PDS2.0/Classifications.csv', index=False)


# Detect anomalies
anomaly_indices = np.where(errors > threshold)[0]

# Print peaks with corresponding date and time
for idx in peaks:
    peak_time = df.iloc[idx]['time']  # Assuming 'time' column has the date-time information
    print(f"Peak detected at index {idx} corresponding to time: {peak_time}")
