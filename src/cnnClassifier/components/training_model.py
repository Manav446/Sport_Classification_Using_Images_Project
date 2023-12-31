import os
import tensorflow as tf
import time
from pathlib import Path
import sys

from cnnClassifier.entity.config_entity import TrainingConfig
from src.logger import logging
from src.exception import CustomException

logger = logging.getLogger("ModelTrainingClass")

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    def get_train_validation_data(self):
        logger.info("Entering into get_train_validation_data()......")
        try:
            self.train_generator = tf.keras.utils.image_dataset_from_directory(
                directory = self.config.training_data,
                labels = "inferred",
                label_mode="categorical",
                shuffle = True,
                class_names = self.config.params_classes,
                seed = 42,
                image_size=tuple(self.config.params_image_size[0:2]),
                batch_size=self.config.params_batch_size
            )
        except Exception as exe:
            logger.exception(exe)
            raise CustomException(exe, sys)

        try:
            self.valid_generator = tf.keras.utils.image_dataset_from_directory(
                directory = self.config.validation_data,
                labels="inferred",
                label_mode="categorical",
                class_names= self.config.params_classes,
                seed=42,
                image_size=tuple(self.config.params_image_size[0:2]),
                batch_size=self.config.params_batch_size
            )
        except Exception as exe:
            logger.exception(exe)
            raise CustomException(exe, sys)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)


    def model_training(self):
        logger.info("Entering into model_training().........")

        try:
            self.model.fit(
                self.train_generator,
                epochs = self.config.params_epochs,
                validation_data = self.valid_generator,
                verbose = 1
            )
            self.save_model(
                path=self.config.trained_model_path,
                model = self.model
            )
        except Exception as exe:
            logger.exception(exe)
            raise CustomException(exe, sys)
    
    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
