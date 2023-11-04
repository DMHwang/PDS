import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import regularizers

# Load in the data
df = pd.read_csv('/Users/raymondblahajr/Desktop/PDS/Machine_Learning_Stuff/train_data.csv', low_memory=False)

norm_cols = ['xrsa_flux', 'xrsb_flux', 'xrsa_cumsum', 'xrsb_cumsum']

# Instead of re-normalizing, simply split the already normalized data into training and validation sets if needed
train_ratio = 0.8
train_size = int(train_ratio * len(df))
df_train = df.iloc[:train_size].copy()
df_val = df.iloc[train_size:].copy()

# No need to normalize again since it's already normalized in the splitting script
# If you introduce new data (like a separate validation set) which isn't normalized, then load the saved scaler and use it

# Sequence Length
SEQUENCE_LENGTH = 1440 # 24 hours * 60 minutes


# Sequence Generator Update for 24-hour Windows
def sequence_generator(df, batch_size=64, is_autoencoder=True):
    """Yield sequences of data and, if is_autoencoder, the same data as labels."""
    while True:
        for day_start in range(0, len(df) - SEQUENCE_LENGTH, SEQUENCE_LENGTH * batch_size):
            sequences = []

            for batch_num in range(batch_size):
                start = day_start + (batch_num * SEQUENCE_LENGTH)
                end = start + SEQUENCE_LENGTH

                if end <= len(df):
                    sequences.append(df.iloc[start:end][norm_cols].values)

            if len(sequences) == batch_size:  # Ensure we only yield full batches
                if is_autoencoder:
                    yield np.array(sequences), np.array(sequences)
                else:
                    yield np.array(sequences)



# Model Architecture
latent_dim = 20  # You can tweak this

# Mapping the status to numeric values
def status_to_numeric(status):
    mapping = {'': 0, 'EVENT_PEAK': 1}
    return mapping.get(status, 0)

df_train['status_label'] = df_train['status'].apply(status_to_numeric)
df_val['status_label'] = df_val['status'].apply(status_to_numeric)

# Computing class weights for training data only
class_weights = compute_class_weight('balanced', classes=np.unique(df_train['status_label']), y=df_train['status_label'])
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

BATCH_SIZE = 128
train_dataset = tf.data.Dataset.from_generator(
    lambda: sequence_generator(df_train, BATCH_SIZE),
    output_signature=(
        tf.TensorSpec(shape=(BATCH_SIZE, SEQUENCE_LENGTH, len(norm_cols)), dtype=tf.float32),
        tf.TensorSpec(shape=(BATCH_SIZE, SEQUENCE_LENGTH, len(norm_cols)), dtype=tf.float32)  # This should match the shape of the sequences
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: sequence_generator(df_val, BATCH_SIZE),
    output_signature=(
        tf.TensorSpec(shape=(BATCH_SIZE, SEQUENCE_LENGTH, len(norm_cols)), dtype=tf.float32),
        tf.TensorSpec(shape=(BATCH_SIZE, SEQUENCE_LENGTH, len(norm_cols)), dtype=tf.float32)  # This should match the shape of the sequences
    )
)



print(df.isnull().sum())
print((df_train[norm_cols] == np.inf).sum())
print((df_train[norm_cols] == -np.inf).sum())
print("Infinite values in df_train:", (np.isinf(df_train[norm_cols])).sum().sum())
print("Infinite values in df_val:", (np.isinf(df_val[norm_cols])).sum().sum())


# Encoder
encoder_input = layers.Input(shape=(SEQUENCE_LENGTH, len(norm_cols)))

# Consider using different activation functions in the LSTM layers
x = layers.LSTM(50, return_sequences=True, activation='tanh')(encoder_input) # Modify activation as needed

# Add Dropout layers or L1/L2 regularizations to prevent overfitting
x = layers.Dropout(0.5)(x) # Adjust dropout rate as needed
x = layers.LSTM(latent_dim, return_sequences=False, kernel_regularizer=regularizers.l2(0.01))(x) # Add L1/L2 regularizations

encoder = Model(inputs=encoder_input, outputs=x)

# Decoder
decoder_input = layers.Input(shape=(latent_dim,))

# Incorporate Batch Normalization to stabilize and possibly accelerate the training process
x = layers.BatchNormalization()(decoder_input)

x = layers.RepeatVector(SEQUENCE_LENGTH)(x)
x = layers.LSTM(latent_dim, return_sequences=True, activation='tanh')(x) # Modify activation as needed
x = layers.Dropout(0.5)(x) # Adjust dropout rate as needed
x = layers.LSTM(50, return_sequences=True)(x)
x = layers.TimeDistributed(layers.Dense(len(norm_cols)))(x)

decoder = Model(inputs=decoder_input, outputs=x)

# Autoencoder = Encoder + Decoder
autoencoder_input = layers.Input(shape=(SEQUENCE_LENGTH, len(norm_cols)))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = Model(inputs=autoencoder_input, outputs=decoded)

# Compile & Train
autoencoder.compile(optimizer='adam', loss='mse')

# Compute the number of steps per epoch
def compute_steps(df, batch_size):
    return (len(df) - SEQUENCE_LENGTH) // (SEQUENCE_LENGTH * batch_size)


train_steps = compute_steps(df_train, BATCH_SIZE)
test_steps = compute_steps(df_val, BATCH_SIZE)


# Train the model
autoencoder.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    steps_per_epoch=train_steps,
    validation_steps=test_steps
)

# Save the model
autoencoder.save('autoencoder_model2.h5')

# Detect Anomalies on Validation Data
reconstructed = autoencoder.predict(val_dataset, steps=test_steps)

# Reshape reconstructed and validation datasets for error calculation
total_val_samples = BATCH_SIZE * test_steps * SEQUENCE_LENGTH
val_reshaped = df_val[norm_cols].values[:total_val_samples]
reconstructed_reshaped = reconstructed.reshape(-1, len(norm_cols))

# Calculate the error
errors = np.mean(np.power(val_reshaped - reconstructed_reshaped, 2), axis=1)

# Set a threshold based on your understanding of the error distribution. Chould change this to increase model sensitivity
threshold = np.percentile(errors, 95)

# Print the anomalies
anomalies = np.where(errors > threshold)
print("Anomalies found at indices:", anomalies)