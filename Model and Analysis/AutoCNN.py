import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import tensorflow.keras.backend as K

# Load in the data
df_train = pd.read_csv('/Users/raymondblahajr/Desktop/PDS2.0/train_datafull.csv', low_memory=False)
df_val = pd.read_csv('/Users/raymondblahajr/Desktop/PDS2.0/val_datafull.csv', low_memory=False)

norm_cols = ['xrsa_flux', 'xrsb_flux']

# Sequence Length - 1440 mins * 128 batches = 24 hours
SEQUENCE_LENGTH = 1440
BATCH_SIZE = 128

# Sequence Generator for 6-hour Windows
def sequence_generator(df, batch_size=128, is_autoencoder=True):
    while True:
        for start in range(0, len(df) - SEQUENCE_LENGTH, SEQUENCE_LENGTH // 2):
            sequences = []
            for batch_num in range(batch_size):
                end = start + SEQUENCE_LENGTH
                if end <= len(df):
                    sequences.append(df.iloc[start:end][norm_cols].values)
                start += SEQUENCE_LENGTH // 2
            if len(sequences) == batch_size:
                if is_autoencoder:
                    yield np.array(sequences), np.array(sequences)
                else:
                    yield np.array(sequences)

train_dataset = tf.data.Dataset.from_generator(
    lambda: sequence_generator(df_train, BATCH_SIZE),
    output_signature=(
        tf.TensorSpec(shape=(BATCH_SIZE, SEQUENCE_LENGTH, len(norm_cols)), dtype=tf.float32),
        tf.TensorSpec(shape=(BATCH_SIZE, SEQUENCE_LENGTH, len(norm_cols)), dtype=tf.float32)
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: sequence_generator(df_val, BATCH_SIZE),
    output_signature=(
        tf.TensorSpec(shape=(BATCH_SIZE, SEQUENCE_LENGTH, len(norm_cols)), dtype=tf.float32),
        tf.TensorSpec(shape=(BATCH_SIZE, SEQUENCE_LENGTH, len(norm_cols)), dtype=tf.float32)
    )
)

# Custom Loss Function
def custom_loss(y_true, y_pred):
    false_negatives = K.sum(K.square((y_true - y_pred) * y_true))
    false_positives = K.sum(K.square((y_true - y_pred) * (1 - y_true)))
    return K.mean(K.square(y_true - y_pred)) + 2 * false_negatives + 0.5 * false_positives

# Model Architecture
latent_dim = 10

# Encoder
encoder_input = layers.Input(shape=(SEQUENCE_LENGTH, len(norm_cols)))
x = layers.Conv1D(16, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(encoder_input)
x = layers.MaxPooling1D(2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv1D(32, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Flatten()(x)
encoder_output = layers.Dense(latent_dim, activation='relu')(x)
encoder = Model(inputs=encoder_input, outputs=encoder_output)

# Decoder
decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense((SEQUENCE_LENGTH//4)*32, activation='relu')(decoder_input)
x = layers.Reshape((SEQUENCE_LENGTH//4, 32))(x)
x = layers.BatchNormalization()(x)
x = layers.Conv1DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
x = layers.Conv1DTranspose(16, 3, strides=2, activation='relu', padding='same')(x)
decoder_output = layers.Conv1D(len(norm_cols), 3, activation='linear', padding='same')(x)
decoder = Model(inputs=decoder_input, outputs=decoder_output)

# Autoencoder
autoencoder_input = layers.Input(shape=(SEQUENCE_LENGTH, len(norm_cols)))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = Model(inputs=autoencoder_input, outputs=decoded)

# Compute the number of steps per epoch
def compute_steps(df, batch_size):
    return (len(df) - SEQUENCE_LENGTH) // (SEQUENCE_LENGTH * batch_size)

# Compile and train the model
autoencoder.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001), loss=custom_loss)
train_steps = compute_steps(df_train, BATCH_SIZE)
test_steps = compute_steps(df_val, BATCH_SIZE)

# Removed the learning rate scheduler callback
autoencoder.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    steps_per_epoch=train_steps,
    validation_steps=test_steps
)

# Save the model
autoencoder.save('autoencoder_modelfull.h5')

# Detect Anomalies on Validation Data
reconstructed = autoencoder.predict(val_dataset, steps=test_steps)
total_val_samples = BATCH_SIZE * test_steps * SEQUENCE_LENGTH
val_reshaped = df_val[norm_cols].values[:total_val_samples]
reconstructed_reshaped = reconstructed.reshape(-1, len(norm_cols))
errors = np.mean(np.power(val_reshaped - reconstructed_reshaped, 2), axis=1)
threshold = np.percentile(errors, 95) 
anomalies = np.where(errors > threshold)
print("Anomalies found at indices:", anomalies)