import tensorflow as tf
import numpy as np
from sklearn.datasets import make_circles
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# Generate data
X, y = make_circles(n_samples=1000, noise=0.1, factor=0.2, random_state=1)



# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Fit the model
# model.fit(X, y, epochs=100)
