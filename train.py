import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from src.dataloader import DogLanguageDataloader
from models.bilstm import BiLSTM
from models.bilstm_fcn import BiLSTM_FCN
from models.lstm import LSTM
from models.lstm_fcn import LSTM_FCN
from models.fcn import FCN



data_dir = "data/"
loader = DogLanguageDataloader(data_dir)
X_train,y_train_encoded, X_test,y_test_encoded = loader.prepare_data()


X_train_reshaped = np.array(X_train)
X_test_reshaped = np.array(X_test)
label_encoder =  loader.label_encoder


models = []
names = []


# FCN Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 1
num_filters = 64
kernel_size = 3
lstm_units = 64

model = FCN(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("FCN_1")


# FCN-2 Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 3
lstm_units = 64

model = FCN(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("FCN_2")

# FCN-3 Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 3
num_filters = 64
kernel_size = 3
lstm_units = 64

model = FCN(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("FCN_3")


# FCN-4 Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 3
lstm_units = 64

model = FCN(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("FCN_4")


# FCN Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 5
num_filters = 64
kernel_size = 3
lstm_units = 64

model = FCN(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("FCN_5")


# FCN Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 6
num_filters = 64
kernel_size = 3
lstm_units = 64

model = FCN(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("FCN_6")





# LSTM Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 2
lstm_units = 64

model = LSTM(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("LSTM_64")






# LSTM-124 Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 2
lstm_units = 124

model = LSTM(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("LSTM_124")

# LSTM-212 Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 2
lstm_units = 212

model = LSTM(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("LSTM_212")

# LSTM-212 Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 2
lstm_units = 300

model = LSTM(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("LSTM_300")






# BiLSTM 64 Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 2
lstm_units = 64

model = BiLSTM(num_classes, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("BiLSTM-64")



# BiLSTM_124 Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 2
lstm_units = 124

model = BiLSTM(num_classes, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("BiLSTM-124")

# BiLSTM_212 Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 2
lstm_units = 212

model = BiLSTM(num_classes, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("BiLSTM-212")


# BiLSTM_212 Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 2
lstm_units = 300

model = BiLSTM(num_classes, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("BiLSTM-300")



#  LSTM_FCN Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 2
lstm_units = 64

model = LSTM_FCN(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("LSTM_FCN_4_64")


#  LSTM_FCN Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 2
lstm_units = 212

model = LSTM_FCN(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("LSTM_FCN_4_212")



#  LSTM_FCN Model
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 2
lstm_units = 300

model = LSTM_FCN(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("LSTM_FCN_4_300")

# BiLSTM FCN
num_classes = len(label_encoder.classes_)
num_conv_layers = 4
num_filters = 64
kernel_size = 3
lstm_units = 64

model = BiLSTM_FCN(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("BiLSTM_FCN_4_64")



# BiLSTM FCN
num_classes = len(label_encoder.classes_)
num_conv_layers = 5
num_filters = 64
kernel_size = 3
lstm_units = 212

model = BiLSTM_FCN(num_classes, num_conv_layers, num_filters, kernel_size, lstm_units)
model.build(input_shape=(None, X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy',
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall'),
                       tf.keras.metrics.F1Score(name='f1_score')])
models.append(model)
names.append("BiLSTM_FCN_5_212")





num_epochs = 25
batch_size = 32
histories = []
metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1_score', 'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1_score']

results_dict = {metric: [] for metric in metrics}

for model in models:
    history = model.fit(X_train_reshaped, y_train_encoded, validation_data=(X_test_reshaped, y_test_encoded), epochs=num_epochs, batch_size=batch_size)
    histories.append(history)
    for metric in metrics:
        results_dict[metric].append(history.history[metric][-1])

results_df = pd.DataFrame(results_dict)
results_df["names"] = names
train_results = results_df[["loss",	"accuracy",	"precision",	"recall","f1_score","names"]]
val_results = results_df[["val_loss",	"val_accuracy",	"val_precision",	"val_recall",	"val_f1_score","names"]]
