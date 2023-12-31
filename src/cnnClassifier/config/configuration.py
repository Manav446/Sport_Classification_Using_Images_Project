from pathlib import Path
import os
import sys

from src.cnnClassifier.constants import constants
from src.cnnClassifier.utils.common import read_yaml, create_directories
from src.cnnClassifier.entity.config_entity import (DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig)
from src.logger import logging
from src.exception import CustomException

logger = logging.getLogger("ConfigurationManager")

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = constants.CONFIG_FILE_PATH,
        params_filepath = constants.PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        logger.info("Entering into get_data_ingestion_config()........")
        config = self.config.data_ingestion

        create_directories([config.root_dir])
        try:
            data_ingestion_config = DataIngestionConfig(
                root_dir=config.root_dir,
                source_URL=config.source_URL,
                local_data_file=config.local_data_file,
                unzip_dir=config.unzip_dir 
            )
        except Exception as exe:
            logger.exception(exe)
            raise CustomException(exe, sys)

        return data_ingestion_config
    
    def prepare_base_model(self) -> PrepareBaseModelConfig:
        logger.info("Entering into prepare_base_model().......")
        config = self.config.prepare_base_model

        try:
            prepare_base_model_config = PrepareBaseModelConfig(
                root_dir=Path(config.root_dir),
                base_model_path=Path(config.base_model_path),
                updated_base_model_path=Path(config.updated_base_model_path),
                params_include_top=self.params.INCLUDE_TOP,
                params_learning_rate=self.params.LEARNING_RATE,
                params_weights=self.params.WEIGHTS,
                params_classes=self.params.CLASSES,
                parmas_image_size=self.params.IMAGE_SIZE
            )
        except Exception as exe:
            logger.exception(exe)
            raise CustomException(exe, sys)
        
        return prepare_base_model_config 
    
    def get_training_config(self) -> TrainingConfig:
        logger.info("Entering into get_training_config().........")
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, constants.IMAGE_DATA_FOLDER_NAME, "train")
        validation_data = os.path.join(self.config.data_ingestion.unzip_dir, constants.IMAGE_DATA_FOLDER_NAME, "valid")
        create_directories([
            Path(training.root_dir)
        ])
        total_classes = os.listdir(os.path.join(self.config.data_ingestion.unzip_dir, constants.IMAGE_DATA_FOLDER_NAME, "train"))

        try:
            training_config = TrainingConfig(
                root_dir=Path(training.root_dir),
                trained_model_path=Path(training.trained_model_path),
                updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
                training_data=Path(training_data),
                validation_data=Path(validation_data),
                params_epochs=params.EPOCHS,
                params_batch_size=params.BATCH_SIZE,
                params_is_augmentation=params.AUGMENTATION,
                params_image_size=params.IMAGE_SIZE,
                params_total_classes = len(total_classes),
                params_classes=total_classes
            )
        except Exception as exe:
            logger.exception(exe)
            raise CustomException(exe, sys)

        return training_config