import tensorflow as tf
from tensorflow.keras import layers
 
def create_model(class_names):
  input_shape = (224, 224, 3)
  base_model = tf.keras.applications.efficientnet.EfficientNetB1(include_top=False)

  inputs = layers.Input(shape=input_shape, name="input_layer")
  
  x = base_model(inputs)
  x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
  x = layers.Dropout(0.3)(x)
  x = layers.Dense(len(class_names))(x)
  outputs = layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)
  model = tf.keras.Model(inputs, outputs)

  return model