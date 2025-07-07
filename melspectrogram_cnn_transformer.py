
import os
import math
import json
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)

def save_melspectrogram():
    """Enhanced Mel-spectrogram extraction with better parameters"""
    # Constants
    SAMPLE_RATE = 22050
    SAMPLES_PER_TRACK = SAMPLE_RATE * 30

    # Enhanced parameters for mel-spectrogram extraction
    num_segments = 10
    n_mels = 128  # Number of mel frequency bins
    n_fft = 2048
    hop_length = 512
    fmin = 0      # Minimum frequency
    fmax = 8000   # Maximum frequency (reduced from Nyquist for music)

    dataset_path = r"../input/gtzan-dataset-music-genre-classification/Data/genres_original"
    json_path = r"data_melspec_lstm.json"

    # Data storage dictionary
    data = {
        "mapping": [],
        "melspectrogram": [],
        "labels": [],
    }

    samples_ps = int(SAMPLES_PER_TRACK / num_segments)
    expected_vects_ps = math.ceil(samples_ps / hop_length)

    print(f"Expected vectors per segment: {expected_vects_ps}")
    print(f"Samples per segment: {samples_ps}")
    print(f"Mel frequency bins: {n_mels}")
    print(f"Frequency range: {fmin}-{fmax} Hz")

    # Loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            # Save the semantic label
            dirpath_comp = dirpath.split("/")
            semantic_label = dirpath_comp[-1]
            data["mapping"].append(semantic_label)
            print(f"Processing: {semantic_label}")

            # Process files for specific genre
            for f in tqdm(filenames, desc=f"Processing {semantic_label}"):
                if f == "jazz.00054.wav":
                    print(f"Skipping {f} - known issue")
                    continue

                try:
                    # Load audio file
                    file_path = os.path.join(dirpath, f)
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    # Process each segment
                    for s in range(num_segments):
                        start_sample = samples_ps * s
                        finish_sample = start_sample + samples_ps

                        # Extract the segment
                        segment = signal[start_sample:finish_sample]

                        # Extract Mel-spectrogram features for this segment
                        melspec = librosa.feature.melspectrogram(
                            y=segment,
                            sr=sr,
                            n_mels=n_mels,
                            n_fft=n_fft,
                            hop_length=hop_length,
                            fmin=fmin,
                            fmax=fmax,
                            power=2.0  # Power spectrogram
                        )
                        
                        # Convert to log scale (dB)
                        melspec_db = librosa.power_to_db(melspec, ref=np.max)

                        # Transpose to get time x features format
                        melspec_db = melspec_db.T

                        # Store mel-spectrogram if it has expected length
                        if len(melspec_db) == expected_vects_ps:
                            data["melspectrogram"].append(melspec_db.tolist())
                            data["labels"].append(i - 1)
                        else:
                            print(f"Skipping {file_path}, segment {s+1} - Wrong length: {len(melspec_db)} vs {expected_vects_ps}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

    # Save data to JSON file
    print(f"Saving {len(data['melspectrogram'])} Mel-spectrogram features to {json_path}")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print("Mel-spectrogram extraction completed!")
    return data

def load_data(json_path="data_melspec_lstm.json"):
    """Load and preprocess the Mel-spectrogram data"""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Convert to numpy arrays
    X = np.array(data["melspectrogram"])
    y = np.array(data["labels"])

    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of genres: {len(data['mapping'])}")
    print(f"Genres: {data['mapping']}")

    return X, y, data["mapping"]

class PositionalEncoding(layers.Layer):
    """Fixed Positional encoding layer for transformer"""
    
    def __init__(self, max_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model
        
    def build(self, input_shape):
        super(PositionalEncoding, self).build(input_shape)
        # Create positional encoding matrix
        pe = np.zeros((self.max_length, self.d_model))
        position = np.arange(0, self.max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = self.add_weight(
            name='positional_encoding',
            shape=(1, self.max_length, self.d_model),
            initializer='zeros',
            trainable=False
        )
        self.pe.assign(pe[np.newaxis, :, :])
        
    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'max_length': self.max_length,
            'd_model': self.d_model
        })
        return config

class MultiHeadSelfAttention(layers.Layer):
    """Fixed Multi-head self-attention layer"""
    
    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
    def build(self, input_shape):
        super(MultiHeadSelfAttention, self).build(input_shape)
        self.wq = layers.Dense(self.d_model)
        self.wk = layers.Dense(self.d_model)
        self.wv = layers.Dense(self.d_model)
        self.dense = layers.Dense(self.d_model)
        self.dropout = layers.Dropout(self.dropout_rate)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Add mask if provided
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = tf.matmul(attention_weights, v)
        return output, attention_weights
    
    def call(self, inputs, training=None, mask=None):
        batch_size = tf.shape(inputs)[0]
        
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        concat_attention = tf.reshape(scaled_attention,
                                    (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        return output
    
    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config

class TransformerBlock(layers.Layer):
    """Fixed Transformer encoder block"""
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)
        self.mha = MultiHeadSelfAttention(self.d_model, self.num_heads, self.dropout_rate)
        self.ffn = tf.keras.Sequential([
            layers.Dense(self.dff, activation='relu'),
            layers.Dense(self.d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(self.dropout_rate)
        self.dropout2 = layers.Dropout(self.dropout_rate)
    
    def call(self, inputs, training=None, mask=None):
        attn_output = self.mha(inputs, training=training, mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config

class ImprovedMusicGenreClassifier:
    """Improved CNN-based model for music genre classification"""
    
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def create_model(self):
        """Create an improved CNN model with better architecture"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Reshape for CNN: (time_steps, features) -> (time_steps, features, 1)
        x = layers.Reshape((*self.input_shape, 1))(inputs)
        
        # First CNN block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.1)(x)
        
        # Second CNN block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.15)(x)
        
        # Third CNN block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Fourth CNN block
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers for classification
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

def train_model():
    """Train the improved CNN model"""
    
    # Check if processed data exists, if not create it
    json_path = "data_melspec_lstm.json"
    if not os.path.exists(json_path):
        print("Processed data not found. Creating mel-spectrogram features...")
        save_melspectrogram()
    
    # Load data
    print("Loading data...")
    X, y, genre_mapping = load_data(json_path)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Genre mapping: {genre_mapping}")
    
    # Check for any issues in the data
    print(f"NaN values in X: {np.isnan(X).sum()}")
    print(f"Inf values in X: {np.isinf(X).sum()}")
    print(f"Label distribution: {np.bincount(y)}")
    
    # Improved normalization - per sample normalization
    print("Normalizing data...")
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        sample = X[i]
        # Standardize each sample
        mean = np.mean(sample)
        std = np.std(sample)
        if std > 0:
            X_normalized[i] = (sample - mean) / std
        else:
            X_normalized[i] = sample - mean
    
    X = X_normalized
    
    # Clip extreme values
    X = np.clip(X, -3, 3)
    
    num_classes = len(genre_mapping)
    print(f"Number of classes: {num_classes}")
    
    # Split data with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Convert to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
    
    # Create improved model
    model_builder = ImprovedMusicGenreClassifier(
        input_shape=X_train.shape[1:],
        num_classes=num_classes
    )
    model = model_builder.create_model()
    
    # Use Adam optimizer with learning rate scheduling
    initial_learning_rate = 0.001
    optimizer = Adam(learning_rate=initial_learning_rate)
    
    # Compile model without label smoothing initially
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Improved callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        ModelCheckpoint(
            'best_improved_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
            cooldown=2
        )
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model for evaluation
    try:
        model = tf.keras.models.load_model('best_improved_model.h5')
    except:
        print("Using current model (couldn't load saved model)")
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Predictions
    print("Making predictions...")
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    final_accuracy = np.mean(y_pred_classes == y_test)
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=genre_mapping))
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss', alpha=0.8, linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', alpha=0.8, linewidth=2)
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', alpha=0.8, linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', alpha=0.8, linewidth=2)
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate plot
    plt.subplot(1, 3, 3)
    if 'learning_rate' in history.history:
        plt.plot(history.history['learning_rate'], label='Learning Rate', alpha=0.8, linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_cnn_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Enhanced confusion matrix
    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(y_test, y_pred_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=genre_mapping, yticklabels=genre_mapping,
                cbar_kws={'label': 'Normalized Count'})
    plt.title(f'Improved CNN Model - Normalized Confusion Matrix\nTest Accuracy: {test_accuracy:.4f}', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Genre', fontsize=12)
    plt.ylabel('Actual Genre', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('improved_cnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model and preprocessing parameters
    model.save('gtzan_improved_cnn_model.h5')
    
    print(f"\nModel saved as 'gtzan_improved_cnn_model.h5'")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    # Show per-class accuracy
    print("\nPer-class Accuracy:")
    for i, genre in enumerate(genre_mapping):
        class_mask = y_test == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.sum((y_pred_classes == i) & class_mask) / np.sum(class_mask)
            print(f"{genre}: {class_accuracy:.4f}")
    
    return model, history, genre_mapping
        
          
if __name__ == "__main__":
    # Train the model
    model, history, genre_mapping = train_model()
    print("Training completed successfully!")
