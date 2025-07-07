import os
import json
import math
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def save_mfcc():
    """Enhanced MFCC extraction with better parameters"""
    # Constants
    SAMPLE_RATE = 22050
    SAMPLES_PER_TRACK = SAMPLE_RATE * 30

    # Enhanced parameters for better feature extraction
    num_segments = 10
    n_mfcc = 20
    n_fft = 2048
    hop_length = 512

    dataset_path = r"../input/gtzan-dataset-music-genre-classification/Data/genres_original"
    json_path = r"data_lstm.json"

    # Data storage dictionary
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
    }

    samples_ps = int(SAMPLES_PER_TRACK / num_segments)
    expected_vects_ps = math.ceil(samples_ps / hop_length)

    print(f"Expected vectors per segment: {expected_vects_ps}")
    print(f"Samples per segment: {samples_ps}")
    print(f"MFCC coefficients: {n_mfcc}")

    # Loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            # Save the semantic label
            dirpath_comp = dirpath.split("/")
            semantic_label = dirpath_comp[-1]
            data["mapping"].append(semantic_label)
            print(f"Processing: {semantic_label}")

            # Process files for specific genre
            for f in filenames:
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

                        # Extract MFCC features for this segment
                        mfcc_features = librosa.feature.mfcc(
                            y=segment,
                            sr=sr,
                            n_mfcc=n_mfcc,
                            n_fft=n_fft,
                            hop_length=hop_length
                        )

                        # Transpose to get time x features format
                        mfcc_features = mfcc_features.T

                        # Store MFCC if it has expected length
                        if len(mfcc_features) == expected_vects_ps:
                            data["mfcc"].append(mfcc_features.tolist())
                            data["labels"].append(i - 1)
                        else:
                            print(f"Skipping {file_path}, segment {s+1} - Wrong length: {len(mfcc_features)} vs {expected_vects_ps}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

    # Save data to JSON file
    print(f"Saving {len(data['mfcc'])} MFCC features to {json_path}")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print("MFCC extraction completed!")
    return data

def load_data(json_path="data_lstm.json"):
    """Load and preprocess the MFCC data"""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Convert to numpy arrays
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Number of genres: {len(data['mapping'])}")
    print(f"Genres: {data['mapping']}")

    return X, y, data["mapping"]

class WaveNetBlock(layers.Layer):
    """Enhanced WaveNet-inspired residual block with stronger regularization"""
    
    def __init__(self, filters, dilation_rate, dropout_rate=0.4):
        super(WaveNetBlock, self).__init__()
        self.filters = filters
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        
        # Dilated causal convolution with L2 regularization
        self.conv = layers.Conv1D(
            filters=filters,
            kernel_size=2,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='tanh',
            kernel_regularizer=keras.regularizers.l2(0.001)
        )
        
        # Gated activation
        self.gate_conv = layers.Conv1D(
            filters=filters,
            kernel_size=2,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='sigmoid',
            kernel_regularizer=keras.regularizers.l2(0.001)
        )
        
        # Residual and skip connections
        self.residual_conv = layers.Conv1D(
            filters=filters, 
            kernel_size=1,
            kernel_regularizer=keras.regularizers.l2(0.001)
        )
        self.skip_conv = layers.Conv1D(
            filters=filters, 
            kernel_size=1,
            kernel_regularizer=keras.regularizers.l2(0.001)
        )
        
        # Increased dropout and layer normalization
        self.dropout = layers.Dropout(dropout_rate)
        self.layer_norm = layers.LayerNormalization()
        
    def call(self, inputs, training=None):
        # Dilated convolutions with gated activation
        tanh_out = self.conv(inputs)
        sigmoid_out = self.gate_conv(inputs)
        
        # Gated activation: tanh * sigmoid
        gated = tanh_out * sigmoid_out
        gated = self.dropout(gated, training=training)
        
        # Skip connection
        skip = self.skip_conv(gated)
        
        # Residual connection
        residual = self.residual_conv(gated)
        
        # Add residual connection if dimensions match
        if inputs.shape[-1] == residual.shape[-1]:
            output = self.layer_norm(inputs + residual)
        else:
            output = self.layer_norm(residual)
            
        return output, skip

def create_regularized_wavenet_model(input_shape, num_classes, num_blocks=8, filters=64):
    """
    Create WaveNet-inspired model with enhanced regularization
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution with regularization
    x = layers.Conv1D(
        filters=filters, 
        kernel_size=1, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.001)
    )(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Early dropout
    
    # Stack of WaveNet blocks with increasing dilation
    skip_connections = []
    
    for i in range(num_blocks):
        # Exponentially increasing dilation rates
        dilation_rate = 2 ** (i % 6)  # Reduced to prevent excessive dilation
        
        # Gradually increase dropout through the network
        dropout_rate = 0.3 + (i / num_blocks) * 0.2
        
        x, skip = WaveNetBlock(
            filters=filters,
            dilation_rate=dilation_rate,
            dropout_rate=dropout_rate
        )(x)
        
        skip_connections.append(skip)
    
    # Combine all skip connections
    skip_sum = layers.Add()(skip_connections)
    
    # Final processing layers with strong regularization
    x = layers.Activation('relu')(skip_sum)
    x = layers.Conv1D(
        filters=filters, 
        kernel_size=1, 
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.001)
    )(x)
    x = layers.Dropout(0.5)(x)
    
    # Global average pooling to reduce sequence dimension
    x = layers.GlobalAveragePooling1D()(x)
    
    # Reduced dense layers with stronger regularization
    x = layers.Dense(
        128,  # Reduced from 256
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.6)(x)
    
    x = layers.Dense(
        64,   # Reduced from 128
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.01)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Enhanced data augmentation
def augment_mfcc_advanced(mfcc_features, noise_factor=0.02, time_shift_factor=0.1):
    """Apply multiple augmentation techniques to MFCC features"""
    augmented = mfcc_features.copy()
    
    # Add noise
    if np.random.random() > 0.5:
        noise = np.random.normal(0, noise_factor, mfcc_features.shape)
        augmented = augmented + noise
    
   
    
    return augmented

def train_enhanced_wavenet_model():
    """Train the enhanced WaveNet model with better regularization"""
    
    # Load data
    print("Loading MFCC data...")
    try:
        X, y, genre_mapping = load_data()
    except FileNotFoundError:
        print("Data file not found. Extracting MFCC features...")
        save_mfcc()
        X, y, genre_mapping = load_data()
    
    # Data preprocessing with enhanced normalization
    print("Preprocessing data...")
    
    # More robust normalization per feature
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[2]):  # For each MFCC coefficient
        feature_data = X[:, :, i]
        mean = np.mean(feature_data)
        std = np.std(feature_data)
        X_normalized[:, :, i] = (feature_data - mean) / (std + 1e-8)
    X = X_normalized
    
    # Convert labels to categorical
    num_classes = len(genre_mapping)
    y_categorical = keras.utils.to_categorical(y, num_classes)
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create augmented training data
    print("Creating augmented training data...")
    X_train_aug = []
    y_train_aug = []
    
    for i in range(len(X_train)):
        # Original sample
        X_train_aug.append(X_train[i])
        y_train_aug.append(y_train[i])
        
        # Add 1 augmented version (reduced from 2 to prevent overfitting)
        augmented = augment_mfcc_advanced(X_train[i])
        X_train_aug.append(augmented)
        y_train_aug.append(y_train[i])
    
    X_train_aug = np.array(X_train_aug)
    y_train_aug = np.array(y_train_aug)
    
    print(f"Augmented training set: {X_train_aug.shape}")
    
    # Create model with better regularization
    print("Creating enhanced WaveNet model...")
    model = create_regularized_wavenet_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        num_classes=num_classes,
        num_blocks=10,  # Slightly reduced complexity
        filters=96      # Balanced filter size
    )
    
    # Compile model with label smoothing and advanced optimizer
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.0005,  # Reduced learning rate
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999
    )
    
    # Use label smoothing to prevent overconfidence
    loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    print(model.summary())
    
    # Enhanced callbacks for better training control
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,  # Increased patience
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,   # More aggressive reduction
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_enhanced_wavenet_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Cosine annealing for learning rate
        keras.callbacks.LearningRateScheduler(
            lambda epoch: 0.0005 * (0.5 * (1 + np.cos(np.pi * epoch / 100)))
        )
    ]
    
    # Train model with augmented data
    print("Training enhanced model...")
    history = model.fit(
        X_train_aug, y_train_aug,
        batch_size=64,  # Increased batch size for stability
        epochs=100,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy, test_top_k = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Test Top-K Accuracy: {test_top_k:.4f}")
    
    # Detailed evaluation
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_classes, y_pred_classes, target_names=genre_mapping))
    
    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=genre_mapping, yticklabels=genre_mapping)
    plt.title('Confusion Matrix - Enhanced WaveNet MFCC Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    # Plot training history with validation gap analysis
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', alpha=0.8)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', alpha=0.8)
    # Add gap visualization
    gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
    plt.fill_between(range(len(gap)), history.history['val_accuracy'], 
                     history.history['accuracy'], alpha=0.3, color='red', 
                     label='Overfitting Gap')
    plt.title('Model Accuracy with Overfitting Analysis')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss', alpha=0.8)
    plt.plot(history.history['val_loss'], label='Validation Loss', alpha=0.8)
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(gap, label='Accuracy Gap (Train - Val)', color='red', alpha=0.8)
    plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Target Gap (<5%)')
    plt.title('Overfitting Monitor')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print overfitting analysis
    final_gap = history.history['accuracy'][-1] - history.history['val_accuracy'][-1]
    print(f"\nOverfitting Analysis:")
    print(f"Final accuracy gap: {final_gap:.4f} ({final_gap*100:.2f}%)")
    if final_gap < 0.05:
        print(" Good generalization (gap < 5%)")
    elif final_gap < 0.10:
        print(" Moderate overfitting (gap 5-10%)")
    else:
        print(" Significant overfitting (gap > 10%)")
    
    return model, history, test_accuracy

if __name__ == "__main__":
    # Train the enhanced model
    print(" Enhanced Music Genre Classification with WaveNet")
    print("=" * 60)
    
    model, history, accuracy = train_enhanced_wavenet_model()
    
    
    print(f"\n Achieved {accuracy*100:.2f}% accuracy.")
      