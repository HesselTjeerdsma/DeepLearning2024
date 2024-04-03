# https://github.com/JackonYang/captcha-tensorflow/blob/master/captcha-solver-tf2-4digits-AlexNet-98.8.ipynb

from tensorflow.keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Conv2D,
    MaxPool2D,
    GlobalMaxPool2D,
    Dropout,
)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model

H = 46
W = 160
C = 3
N_LABELS = 10
D = 4

input_layer = tf.keras.Input(shape=(H, W, C))
x = layers.Conv2D(32, 3, activation="relu")(input_layer)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
# x = layers.Dropout(0.5)(x)

x = layers.Dense(D * N_LABELS, activation="softmax")(x)
x = layers.Reshape((D, N_LABELS))(x)

model = models.Model(inputs=input_layer, outputs=x)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()
