import tensorflow as tf


class FCN(tf.keras.Model):
    def __init__(self, num_classes, num_conv_layers, num_filters, kernel_size, lstm_units):
        super(FCN, self).__init__()

        # Define the fully convolutional block
        self.conv_block = tf.keras.Sequential()
        for _ in range(num_conv_layers):
            self.conv_block.add(tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same'))
            self.conv_block.add(tf.keras.layers.BatchNormalization())
        self.conv_block.add(tf.keras.layers.GlobalAveragePooling1D())


        # Softmax classification layer
        self.classification_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Apply the fully convolutional block to the input
        x1 = self.conv_block(inputs)

        # Concatenate the outputs from both blocks
        x = tf.keras.layers.concatenate([x1], axis=-1)
        output = self.classification_layer(x)

        return output
    