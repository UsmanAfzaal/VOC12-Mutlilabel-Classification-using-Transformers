import keras
from tensorflow import keras
from tensorflow.keras import layers

def custom_model(image_shape: tuple):
    inputs = keras.Input(shape=image_shape+(3,), name="img")

    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.Conv2D(64, 3, activation="relu")(x)

    x = layers.SeparableConv2D(64, 3, activation="relu")(x)
    x = layers.SeparableConv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.SeparableConv2D(64, 3, activation="relu")(x)
    x = layers.SeparableConv2D(64, 3, activation="relu")(x)
    x = layers.SeparableConv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.SeparableConv2D(128, 3, activation="relu")(x)
    x = layers.SeparableConv2D(128, 3, activation="relu")(x)
    x = layers.SeparableConv2D(128, 3, activation="relu")(x)

    MSA_layer = layers.MultiHeadAttention(num_heads=5, key_dim=3)
    MSA_1 = MSA_layer(x, x)
    MSA_2 = MSA_layer(MSA_1, MSA_1)

    x = layers.GlobalAveragePooling2D()(MSA_2)
    outputs = layers.Dense(20, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs, name="custom_model")
    return model

def tiny_resnet(image_shape: tuple):
    inputs = keras.Input(shape=image_shape+(3,), name="img")
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    block_1_output = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(20, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs, name="tiny_resnet")
    return model


