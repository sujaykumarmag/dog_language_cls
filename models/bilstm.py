
import tensorflow as tf


class BiLSTM(tf.keras.Model):
    def __init__(self, num_classes, lstm_units):
        super(BiLSTM, self).__init__()

        # Define the BiLSTM block
        self.bilstm_block = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=lstm_units))

        # Softmax classification layer
        self.classification_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Apply the BiLSTM block to the input
        x = self.bilstm_block(inputs)

        # Pass the output through the classification layer
        output = self.classification_layer(x)

        return output