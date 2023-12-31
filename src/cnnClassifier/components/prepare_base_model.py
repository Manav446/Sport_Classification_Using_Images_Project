import os
import zipfile
import gdown
from src.cnnClassifier.utils.common import *
from src.cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from src.cnnClassifier.constants import constants

import tensorflow as tf

from src.logger import logging

logger = logging.getLogger("BaseModelPrepration")

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        # self.model = tf.keras.applications.vgg16.VGG16(
        #     weights=self.config.params_weights,
        #     input_shape=self.config.parmas_image_size,
        #     include_top=self.config.params_include_top
        # )

        self.model = tf.keras.applications.efficientnet.EfficientNetB0(
            weights=self.config.params_weights,
            input_shape=self.config.parmas_image_size,
            include_top=self.config.params_include_top
        )

        self.save_model(
            path = self.config.base_model_path,
            model = self.model
        )

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        preprocessed_input = tf.keras.applications.efficientnet.preprocess_input(
            tf.keras.layers.Input(shape=(224, 224, 3))
        )
        efficient_X = model(preprocessed_input)
        efficient_X = tf.keras.layers.GlobalAveragePooling2D()(efficient_X)
        efficient_X = tf.keras.layers.Dropout(0.2)(efficient_X)
        efficient_X = tf.keras.layers.Dense(
            units=512,
            activation="relu"
        )(efficient_X)
        efficient_X = tf.keras.layers.Dropout(0.2)(efficient_X)
        efficient_X = tf.keras.layers.Dense(
            units = classes,
            activation="softmax"
        )(efficient_X)

        full_model = tf.keras.models.Model(
            inputs=preprocessed_input,
            outputs=efficient_X
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)