import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

translate = [
    "negative",
    "neutral",
    "positive"
]

filepath = "./modelsave"

def fromDistribution(output):
    argmax = np.argmax(output, axis=1)
    conv = map(
        lambda n: translate[n],
        argmax)

    print(output)
    return str(list(conv)[0]) + ", " + str(output[0][argmax[0]])

model = tf.keras.models.load_model(filepath)

for i in range(0, 1000):
    x = input("> ")
    print(fromDistribution(model.predict([x])))