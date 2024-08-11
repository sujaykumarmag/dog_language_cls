import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class DogLanguageDataloader:
    def __init__(self, data_dir, num_mfcc=13, test_size=0.2, random_state=12, batch_size=32):
        self.data_dir = data_dir
        self.num_mfcc = num_mfcc
        self.test_size = test_size
        self.random_state = random_state
        self.batch_size = batch_size

    def _load_data(self):
        X = []
        y = []
        
        for class_label in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_label)
            if os.path.isdir(class_dir):
                for audio_file in os.listdir(class_dir):
                    if audio_file.endswith(".wav"):
                        audio_path = os.path.join(class_dir, audio_file)
                        audio_data, _ = librosa.load(str(audio_path), sr=44100)
                        X.append(audio_data)
                        y.append(class_label)
        
        return X, y

    def _extract_mfcc_features(self, audio_data):
        mfccs = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=self.num_mfcc)
        return mfccs

    def _pad_or_truncate_mfcc(self, X_mfcc):
        max_frames = max(mfcc.shape[1] for mfcc in X_mfcc)
        X_mfcc_padded = np.zeros((len(X_mfcc), X_mfcc[0].shape[0], max_frames))
        for i, mfcc in enumerate(X_mfcc):
            if mfcc.shape[1] < max_frames:
                X_mfcc_padded[i, :, :mfcc.shape[1]] = mfcc
            else:
                X_mfcc_padded[i, :, :] = mfcc[:, :max_frames]
        return X_mfcc_padded

    def _normalize_mfcc(self, X_mfcc_padded):
        mean = np.mean(X_mfcc_padded, axis=2, keepdims=True)
        std = np.std(X_mfcc_padded, axis=2, keepdims=True)
        X_mfcc_normalized = (X_mfcc_padded - mean) / (std + 1e-6)
        return X_mfcc_normalized

    def _encode_labels(self, y_train, y_test):
        self.label_encoder = LabelEncoder()
        y_train_encoded = to_categorical(self.label_encoder.fit_transform(y_train))
        y_test_encoded = to_categorical(self.label_encoder.transform(y_test))
        return y_train_encoded, y_test_encoded

    def prepare_data(self):
        X, y = self._load_data()
        X_mfcc = [self._extract_mfcc_features(audio) for audio in X]
        X_mfcc_padded = self._pad_or_truncate_mfcc(X_mfcc)
        X_mfcc_normalized = self._normalize_mfcc(X_mfcc_padded)

        X_train, X_test, y_train, y_test = train_test_split(X_mfcc_normalized, y, test_size=self.test_size, random_state=self.random_state, shuffle=True)
        
        y_train_encoded, y_test_encoded = self._encode_labels(y_train, y_test)

    

        return X_train, y_train_encoded, X_test, y_test_encoded

