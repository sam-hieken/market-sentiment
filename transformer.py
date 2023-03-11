import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer

translate = [
    "negative",
    "neutral",
    "positive"
]

def fromDistribution(output):
    conv = map(
        lambda n: translate[n],
        np.argmax(output, axis=1))

    return list(conv)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)



class TokenAndPositionEmbeddingLayer(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        #embeddings_initializer="uniform"
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


#
#
# Data preprocessing begins below
#
#

vocab_size = 24000
maxlen = 500

data = pd.read_csv("data.csv", quotechar='"', skipinitialspace=True, encoding="ISO-8859-1")

# Extract the sentiment and text columns
sentiment = data.iloc[:, 0]
text = data.iloc[:, 1]

# Quantitative conversion
sentiment = np.unique(sentiment, return_inverse=True)[1];
# One hop
sentiment = tf.keras.utils.to_categorical(sentiment, num_classes=3)

print(sentiment)

#
#
# Model begins below
#
#

embed_dim = 32  # Embedding size for each token
num_heads = 6  # Number of attention heads
ff_dim = 48  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(1,), dtype=tf.string)

x = layers.TextVectorization(output_mode='int')
x.adapt(text)
x = x(inputs)

x = layers.Masking(mask_value=0)(x)

embedding_layer = TokenAndPositionEmbeddingLayer(maxlen, vocab_size, embed_dim)
x = embedding_layer(x)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout=0.5)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    text, sentiment, batch_size=32, epochs=8
)



# TEST

test = [
    "AAPL fell 61 points today after the CEO announced poor earnings",
    "The value of stock increased by 5 points today",
    "The market is looking pretty bearish"
]

print(fromDistribution(model.predict(test)))

# Save.
model.save("modelsave", save_format="tf")