import tensorflow as tf
import os
from tensorflow.keras import mixed_precision
import requests
from utils.helper_functions import create_tensorboard_callback, plot_loss_curves, compare_historys
from models.model_definition import create_model

def train_model(model, train_data, test_data, class_names):
    checkpoint_path = "model_checkpoints/cp.ckpt"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                          monitor="val_accuracy",
                                                          save_best_only=True,
                                                          save_weights_only=True,
                                                          verbose=0)

    mixed_precision.set_global_policy(policy="mixed_float16")

    tf.get_logger().setLevel('ERROR')

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    history_101_food_classes_feature_extract = model.fit(train_data,
                                                         epochs=3,
                                                         steps_per_epoch=len(train_data),
                                                         validation_data=test_data,
                                                         validation_steps=int(0.15 * len(test_data)),
                                                         callbacks=[create_tensorboard_callback("training_logs",
                                                                                                "efficientnetb0_101_classes_all_data_feature_extract"),
                                                                    model_checkpoint])

    results_created_model = model.evaluate(test_data)

    save_dir = "07_efficientnetb0_feature_extract_model_mixed_precision"
    model.save(save_dir)

