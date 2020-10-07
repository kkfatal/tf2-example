# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

print (tf.__version__)

encoder_input = keras.Input(shape=(28, 28, 1), name='original_img')
x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(16, 3, activation='relu')(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()
#keras.utils.plot_model(encoder, 'my_first_keras_model.png')


if 0:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print (x_train.shape)
    print (y_train.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

if 1:
    x_train = np.random.uniform(0,1,(100,28,28,1)).astype(float) # train examples 
    y_train = np.random.uniform(0,1,(100,16)).astype(float) # train_labels
    print (" training examples : ", x_train.shape)
    print (" training labels : ", y_train.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))


BATCH_SIZE = 8
SHUFFLE_BUFFER_SIZE = 100
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
#test_dataset = test_dataset.batch(BATCH_SIZE)

encoder.compile( optimizer=tf.keras.optimizers.RMSprop(0.001), loss=tf.keras.losses.MeanSquaredError() )

#  train 
#history = encoder.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.2)
history = encoder.fit(train_dataset, epochs=10)



# inference 
pred = encoder(x_train).numpy()
print(pred.shape)

result = encoder.predict(x_train, steps = 10)
print(result.shape)


encoder.save('model1')
#model = keras.models.load_model('path_to_my_model')


# tf lite
# Converting a SavedModel to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_saved_model("model1")
tflite_model = converter.convert()

## Converting a tf.Keras model to a TensorFlow Lite model.
#converter = lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()

file = open( 'model.tflite' , 'wb' ) 
file.write( tflite_model )





print(" End !")



