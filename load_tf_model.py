import tensorflow as tf
# print(tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
# operating on each element in x_"", normalize the values by dividing by its upperbound
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), #converting 2D array into 1D vector
  tf.keras.layers.Dense(128, activation='relu'), #a main hidden layer of 128 neurons
  tf.keras.layers.Dropout(0.2), #during training, randomly set 20% of inputs to 0 to prevent overfitting
  tf.keras.layers.Dense(10, activation='softmax') #output layer with 10 neurons (0-9) and softmax activation, outputting activations
  #that represent probability adding to 1
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)