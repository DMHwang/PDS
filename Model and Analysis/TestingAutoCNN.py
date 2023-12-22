import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from scipy.signal import find_peaks
import tensorflow.keras.backend as K

def custom_loss(y_true, y_pred):
    false_negatives = K.sum(K.square((y_true - y_pred) * y_true))
    false_positives = K.sum(K.square((y_true - y_pred) * (1 - y_true)))
    return K.mean(K.square(y_true - y_pred)) + 2 * false_negatives + 0.5 * false_positives

autoencoder = load_model('autoencoder_modelfull.h5', custom_objects={'custom_loss': custom_loss})

# Load the test dataset
df_test = pd.read_csv('/Users/raymondblahajr/Desktop/PDS2.0/test_datafull.csv', low_memory=False)
norm_cols = ['xrsa_flux', 'xrsb_flux']

SEQUENCE_LENGTH = 1440
BATCH_SIZE = 128

def sequence_generator(df, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH):
    start_idx = 0
    while start_idx + sequence_length <= len(df):
        sequences = []
        for _ in range(batch_size):
            end_idx = start_idx + sequence_length
            if end_idx > len(df):
                break
            sequences.append(df.iloc[start_idx:end_idx][norm_cols].values)
            start_idx += sequence_length // 2  # 50% overlap
        yield np.array(sequences)
        if end_idx >= len(df):
            break

# Generate predictions
reconstructed = []
for batch in sequence_generator(df_test):
    reconstructed.append(autoencoder.predict(batch))
reconstructed = np.concatenate(reconstructed, axis=0)

num_samples_reconstructed = ((reconstructed.shape[0] - 1) * (SEQUENCE_LENGTH // 2)) + SEQUENCE_LENGTH
reconstructed_reshaped = reconstructed.reshape(-1, len(norm_cols))[:num_samples_reconstructed]
test_reshaped = df_test[norm_cols].values[:num_samples_reconstructed]

errors = np.mean(np.power(test_reshaped - reconstructed_reshaped, 2), axis=1)

# Adjust the threshold for anomaly detection
threshold = np.percentile(errors, 93.5)
peaks, _ = find_peaks(errors, height=threshold)

# Post-processing to refine anomaly detection
date_times = df_test['time'].values[:num_samples_reconstructed]
result_df = pd.DataFrame({
    'DateTime': date_times,
    'xrsa_flux': test_reshaped[:, 0],
    'xrsb_flux': test_reshaped[:, 1],
    'Error': errors,
    'Is_Anomaly': np.isin(np.arange(len(errors)), peaks)
})

result_df.to_csv('/Users/raymondblahajr/Desktop/PDS2.0/Anomaly_Detections_Final.csv', index=False)

for idx in peaks:
    anomaly_time = df_test.iloc[idx]['time']
    print(f"Anomaly detected at index {idx} corresponding to time: {anomaly_time}")
