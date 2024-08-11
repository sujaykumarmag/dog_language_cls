
import tensorflow as tf

class LSTM(tf.keras.Model):
    def __init__(self, num_classes, num_conv_layers, num_filters, kernel_size, lstm_units):
        super(LSTM, self).__init__()

        self.lstm_block = tf.keras.Sequential()
        self.lstm_block.add(tf.keras.layers.Permute((2, 1)))
        self.lstm_block.add(tf.keras.layers.LSTM(units=lstm_units))

        # Softmax classification layer
        self.classification_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):

        # Apply the LSTM block to the input
        x2 = self.lstm_block(inputs)
        x = tf.keras.layers.concatenate([x2], axis=-1)
        output = self.classification_layer(x)

        return output
    

