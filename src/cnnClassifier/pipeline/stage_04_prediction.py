import os
import tensorflow as tf
import numpy as np
import sys

from src.exception import CustomException
from src.logger import logging
from src.cnnClassifier.entity.config_entity import TrainingConfig
from src.cnnClassifier.constants import constants

logger = logging.getLogger("Prediction Pipeline")

class PredictionPipeline:
    def __init__(self, fileName):
        self.fileName = fileName
        
    
    def predict_image(self):
        logger.info("Entering into predict_image()............")
        classes = []
        result = None
        try:
            model = tf.keras.models.load_model(os.path.join("artifacts/training/model.h5"))
        
            logger.info("FileName: {}".format(self.fileName))
            test_image = tf.keras.preprocessing.image.load_img(
                self.fileName, 
                target_size = (224, 224)
            )

            test_image = tf.keras.preprocessing.image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            
            result = np.argmax(
                model.predict(test_image),
                axis=1
            )
            logger.info("After prediction result is {}".format(result))
            classes = os.listdir(os.path.join(constants.UNZIP_DIR, constants.IMAGE_DATA_FOLDER_NAME, "train"))
            
            logger.info("Actual Name: {}".format(classes[result[0]]))
        except Exception as e:
            logger.exception(e)
            raise CustomException(e, sys)
        
        return [{
            "image": classes[result[0]]
        }]
        