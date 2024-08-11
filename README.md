# Dog barking Classification

## Table of Contents

- [Installation](#installation)
- [Architecture](#architecture)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
  - [Visualizing the Model](#visualizing-the-model)
- [File Structure](#file-structure)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BiLSTM_FCN_Model.git
   cd BiLSTM_FCN_Model
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   



## Architecture

The BiLSTM_FCN model consists of the following components:

- **Fully Convolutional Block**: A series of 1D convolutional layers followed by batch normalization and global average pooling to extract spatial features.
- **BiLSTM Block**: A Bidirectional LSTM layer to capture temporal dependencies in the input sequence.
- **Concatenation**: Outputs from the Fully Convolutional Block and the BiLSTM Block are concatenated.
- **Softmax Classification Layer**: A dense layer with softmax activation for classification.

### Model Parameters

The model can be customized with the following parameters:
- `num_classes`: Number of output classes.
- `num_conv_layers`: Number of convolutional layers in the FCN block.
- `num_filters`: Number of filters in each convolutional layer.
- `kernel_size`: Size of the convolutional kernel.
- `lstm_units`: Number of units in the LSTM layer.


## Usage

### Training the Model
The following code demonstrates how to instantiate and train the BiLSTM_FCN model:

```python

import tensorflow as tf
from src.dataloader import DogLanguageDataloader
from models.bilstm_fcn import BiLSTM_FCN

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
```
### Evaluating the Model
Evaluate the trained model on the test dataset:

```python
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
```

### Visualizing the Model
To visualize the architecture of the model, use the `plot_model` function:

```python
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='BiLSTM_FCN_model.png', show_shapes=True, show_layer_names=True)
```

## File Structure
```bash
BiLSTM_FCN_Model/
├── assets/
│   └── model_architecture.docx
│
├── data/
│   ├── barking/
│   ├── gowling/
│   ├── howling/
│   └── whining/
│
├── models/
│   ├── bilstm_fcn.py
│   ├── bilstm.py
│   ├── fcn.py
│   ├── lstm_fcn.py
│   └── lstm.py
│
├── notebooks/
│   ├── model_vis.ipynb
│   └── main.ipynb
│
├── src/
│   └── dataloader.py
│
├── results/
│   └── model.pt
│
├── train.py
│
├── model.py
│
├── requirements.txt
│
└── README.md
```

