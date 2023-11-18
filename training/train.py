import tensorflow as tf
import os
from tensorflow.keras import mixed_precision
import requests
from utils.helper_functions import create_tensorboard_callback, plot_loss_curves, compare_historys
from models.model_definition import create_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model(model, train_data, test_data, class_names):
    
    for layer in model.layers[-20:]:
        layer.trainable = True
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  restore_best_weights=True, 
                                                  patience=3, 
                                                  verbose=1)

    checkpoint_path = "fine_tune_checkpoints/"
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_best_only=True,
                                                      monitor="val_loss")

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",
                                                 factor=0.5,
                                                 patience=0,
                                                 verbose=1,
                                                 min_lr=1e-7)

    mixed_precision.set_global_policy(policy="mixed_float16")

    tf.get_logger().setLevel('ERROR')

    model.compile(loss="sparse_categorical_crossentropy",
                            optimizer=tf.keras.optimizers.Adam(0.0001),
                            metrics=["accuracy"])

    history_101_food_classes_all_data_fine_tune = model.fit(train_data,
                                                        epochs=100,
                                                        steps_per_epoch=len(train_data),
                                                        validation_data=test_data,
                                                        validation_steps=int(0.15 * len(test_data)),
                                                        callbacks=[create_tensorboard_callback("training_logs", "efficientb1_101_classes_all_data_fine_tuning"),
                                                                   model_checkpoint,
                                                                   early_stopping,
                                                                   reduce_lr])
    results_created_model = model.evaluate(test_data)
    
    plot_loss_curves(history_101_food_classes_all_data_fine_tune)

    save_dir = "/kaggle/working/Food_Vision_Big/models/07_efficientnetb1_feature_extract_model_mixed_precision_fine_tuning"
    model.save(save_dir)
    print("Save Directory:", save_dir)

