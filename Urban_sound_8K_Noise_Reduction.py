import keras 
import librosa
import os
import pandas as pd 
import tensorflow as tf
import numpy as np
import noisereduce as nr
import soundfile as sf
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.signal import butter, filtfilt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pywt
import noisereduce as nr
import graphviz
import pydot
import pydotplus
# Apply noise reduction

MAX_TIME_STEPS = 500 


def highpass_filter(data, cutoff=300, fs=44100, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def spectral_subtraction(y, sr, noise_frames=1):
    """ Apply spectral subtraction for noise reduction """
    noise_sample = y[:sr * noise_frames]  
    noise_stft = np.mean(np.abs(librosa.stft(noise_sample)), axis=1, keepdims=True)

    stft = librosa.stft(y)
    magnitude, phase = np.abs(stft), np.angle(stft)

    # Ensure no negative values after subtraction
    clean_magnitude = np.maximum(magnitude - noise_stft, 1e-10)

    cleaned_signal = librosa.istft(clean_magnitude * np.exp(1j * phase))

    return cleaned_signal

def wavelet_denoise(y, wavelet='db1', level=3):
    """ Apply wavelet thresholding for noise reduction """
    coeffs = pywt.wavedec(y, wavelet, level=min(level, pywt.dwt_max_level(len(y), wavelet)))
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745
    new_coeffs = [pywt.threshold(c, threshold) for c in coeffs]
    return pywt.waverec(new_coeffs, wavelet)

def gen_mfcc(file_name, num_mfcc=120, max_time_steps=MAX_TIME_STEPS):
    y, sr = librosa.load(file_name, sr=None)

    # Apply spectral subtraction
    # y = spectral_subtraction(y, sr)
  
# Apply noise reduction
    y = nr.reduce_noise(y=y, sr=sr)
    # Apply wavelet-based denoising
    y = wavelet_denoise(y)

    # Ensure the signal is finite
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize safely
    if np.any(np.isfinite(y)) and np.max(np.abs(y)) > 0:
        y = librosa.util.normalize(y)
    else:
        raise ValueError("Processed signal contains only zero or NaN values.")


    # # Extract MFCC
    
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)

    # Pad or truncate
    mfccs = mfccs.T
    if mfccs.shape[0] < max_time_steps:
        mfccs = np.pad(mfccs, ((0, max_time_steps - mfccs.shape[0]), (0, 0)), mode='constant')
    else:
        mfccs = mfccs[:max_time_steps, :]

    return mfccs

def gen_cnn_model():
    
    input_shape = (500, 128)  # (time_steps, num_mfcc)

    # Build the model
    model = keras.models.Sequential()

    # Conv1D Layers
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling1D(pool_size=3, padding='same'))

    
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling1D(pool_size=3, padding='same'))

    model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))

    # Global Average Pooling
    model.add(keras.layers.GlobalAveragePooling1D())

    # Dropout layer
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Reshape((1, 128)))
    # RNN Layer (GRU)
    model.add(keras.layers.GRU(128, return_sequences=False))
    # model.add(keras.layers.LSTM(128, return_sequences=True))
    # model.add(keras.layers.LSTM(128))
    # Fully Connected Output Layer
    model.add(keras.layers.Dense(10, activation='softmax'))  # 10-class classification
    optimzer = keras.optimizers.Adam(learning_rate=0.001)
    # Compile the model
    model.compile(optimizer=optimzer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()
    return model

metadata_df = pd.read_csv('/Users/dangriffith/Library/CloudStorage/OneDrive-CarletonUniversity/COMP4107 - Final/archive/UrbanSound8K.csv')
audio_path = '/Users/dangriffith/Library/CloudStorage/OneDrive-CarletonUniversity/COMP4107 - Final/archive'
folds_dir = '/Users/dangriffith/Library/CloudStorage/OneDrive-CarletonUniversity/COMP4107 - Final/archive'

plot_model  = gen_cnn_model()
keras.utils.plot_model(plot_model, to_file='GRU_model.png', show_shapes=True)

# Store results
accuracies = []
f1s = []
precisions = []
recalls = []

X_train = []
y_train = []

X_val = []
y_val = []

X_hold_out = []
y_hold_out = []


folds = [1,2,3,4,5,6,7,8,9, 10]

for i in range(len(folds)):
    X_train = []
    y_train = []
    temp = []
    X_val = []
    y_val = []

    temp = folds.copy()
    temp.remove(i+1)
    
    print(temp)
    print(i+1)
    
    train_folds = temp
    val_folds = i+1
    
    print(train_folds)
    print(val_folds)
    
    for j in train_folds:
        folder_name = f"fold{j}"
        folder_path = os.path.join(folds_dir, folder_name)
        for file in os.listdir(folder_path):
            class_id = file.split('-')[1]
            y_train.append(class_id)
            X_train.append(gen_mfcc(os.path.join(folder_path, file),128))

    val_file_name = f"fold{val_folds}"
    folder_path = os.path.join(folds_dir, val_file_name)
    
    for file in os.listdir(folder_path):
        class_id = file.split('-')[1]
        y_val.append(class_id)
        X_val.append(gen_mfcc(os.path.join(folder_path, file), 128))

    metric_y = y_val.copy()
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_val = np.array(X_val)
    y_val = np.array(y_val)

   
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    y_val = keras.utils.to_categorical(y_val, num_classes=10)
   
    model = gen_cnn_model()
    history = model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=1)
    
    result = model.evaluate(X_val, y_val)
    y_pred = model.predict(X_val)
    metric_y_pred = np.argmax(y_pred, axis=1)

    metric_y = list(map(int, metric_y))
    precision = precision_score(metric_y, metric_y_pred, average=None) 
    recall = recall_score(metric_y, metric_y_pred, average=None)
    f1 = f1_score(metric_y, metric_y_pred, average=None)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(result)

    recalls.append(recall)
    precisions.append(precision)
    f1s.append(f1)
    accuracies.append(result)
   

print(accuracies)
print(f1s)
print(precisions)
print(recalls)

