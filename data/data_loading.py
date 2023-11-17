import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def load_and_preprocess_data():
    datasets_list = tfds.list_builders()
    target_dataset = "food101"

    (train_data, test_data), ds_info = tfds.load(name=target_dataset,
                                                 split=["train", "validation"],
                                                 shuffle_files=False,
                                                 as_supervised=True,
                                                 with_info=True)

    def preprocess_img(image, label, img_shape=224):
        image = tf.image.resize(image, [img_shape, img_shape])
        return tf.cast(image, tf.float32), label

    train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

    test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

    return (train_data, test_data), ds_info

