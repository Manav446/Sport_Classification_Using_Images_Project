import sys

from src.logger import logging
from src.exception import CustomException
from src.cnnClassifier.pipeline import (stage_01_data_ingestion, stage_02_prepare_base_model, stage_03_model_training)

logger = logging.getLogger("MainFile")

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = stage_01_data_ingestion.DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise CustomException(e, sys)

STAGE_NAME = "Prepare Base Model stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   prepare_base_model = stage_02_prepare_base_model.PrepareBaseModelPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as exe:
   logger.exception(exe)
   raise CustomException(exe, sys)

STAGE_NAME = "Model Training"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_training = stage_03_model_training.ModelTrainingPipeline()
   model_training.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
   logger.exception(e)
   raise CustomException(e, sys)