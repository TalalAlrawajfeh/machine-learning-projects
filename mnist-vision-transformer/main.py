import cv2
import numpy as np
import tensorflow as tf


def multi_layer_perceptron(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier(input_shape,
                          patch_size,
                          num_patches,
                          transformer_layers,
                          projection_dim,
                          num_heads,
                          transformer_units,
                          mlp_head_units,
                          num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    for _ in range(transformer_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,
                                                              key_dim=projection_dim,
                                                              dropout=0.1)(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = multi_layer_perceptron(x3,
                                    hidden_units=transformer_units,
                                    dropout_rate=0.1)
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = tf.keras.layers.Flatten()(representation)
    representation = tf.keras.layers.Dropout(0.5)(representation)

    features = multi_layer_perceptron(representation,
                                      hidden_units=mlp_head_units,
                                      dropout_rate=0.5)
    logits = tf.keras.layers.Dense(num_classes)(features)
    return tf.keras.Model(inputs=inputs, outputs=logits)


def main():
    num_classes = 10
    input_shape = (72, 72, 1)
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 100
    image_size = 72
    patch_size = 6
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [projection_dim * 2, projection_dim]
    transformer_layers = 8
    mlp_head_units = [2048, 1024]

    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    x_train = [255 - cv2.resize(255 - x_train[i], (image_size, image_size)) for i in range(x_train.shape[0])]
    x_train = np.array(x_train, np.float32)
    x_train = x_train.astype(np.float32) / 255.0
    x_train = np.expand_dims(x_train, -1)

    vit = create_vit_classifier(input_shape,
                                patch_size,
                                num_patches,
                                transformer_layers,
                                projection_dim,
                                num_heads,
                                transformer_units,
                                mlp_head_units,
                                num_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    vit.compile(optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])

    vit.summary()

    vit.fit(x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=0.1)


if __name__ == '__main__':
    main()
