import sys
from datetime import datetime
import numpy as np
import os
import pandas as pd
import re

from typing import List, Union

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.components.data_ingestion import DataIngestion
from src.constant.training_pipeline import *
from src.entity.config_entity import SimpleImputerConfig
from src.exception import SpamhamException
from src.logger import logging
from src.utils.main_utils import MainUtils

import warnings
warnings.filterwarnings('ignore')


class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):

        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config
        self.data_ingestion = DataIngestion()
        self.imputer_config = SimpleImputerConfig()
        self.utils = MainUtils()

        # Preload stopwords to avoid repeated calls
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise SpamhamException(e, sys)

    def get_stemmed_data(self, data: pd.DataFrame) -> List[str]:
        try:
            corpus = [
                ' '.join(
                    [self.stemmer.stem(word) for word in re.sub(r"[^a-zA-Z]", ' ', text).lower().split() if word not in self.stop_words]
                )
                for text in data[FEATURE_COLUMN]
            ]
            return corpus
        except Exception as e:
            raise SpamhamException(e, sys)

    def get_vectorized_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, vectorizer=None) -> Union[np.ndarray, np.ndarray, object]:
        try:
            train_data = self.get_stemmed_data(train_df)
            test_data = self.get_stemmed_data(test_df)

            if vectorizer is None:
                vectorizer = CountVectorizer(max_features=3000)

            logging.info("Applying CountVectorizer with max_features=3000")
            x_train = vectorizer.fit_transform(train_data)
            x_test = vectorizer.transform(test_data)

            return x_train, x_test, vectorizer

        except Exception as e:
            raise SpamhamException(e, sys)

    def get_encoded_target_column(self, train_df: pd.DataFrame, test_df: pd.DataFrame, encoder=None) -> Union[np.ndarray, np.ndarray, object]:
        try:
            y_train = train_df[[TARGET_COLUMN]]
            y_test = test_df[[TARGET_COLUMN]]

            if encoder is None:
                encoder = OrdinalEncoder()

            encoded_y_train = encoder.fit_transform(y_train)
            encoded_y_test = encoder.transform(y_test)

            logging.info("Target column encoded.")
            return encoded_y_train, encoded_y_test, encoder

        except Exception as e:
            raise SpamhamException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:

                train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
                test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

                x_train, x_test, vectorizer = self.get_vectorized_data(train_df, test_df)
                y_train, y_test, encoder = self.get_encoded_target_column(train_df, test_df)

                os.makedirs(os.path.dirname(self.data_transformation_config.transformed_vectorizer_object_file_path), exist_ok=True)

                self.utils.save_object(self.data_transformation_config.transformed_encoder_object_file_path, encoder)
                logging.info("Encoder saved.")

                self.utils.save_object(self.data_transformation_config.transformed_vectorizer_object_file_path, vectorizer)
                logging.info("Vectorizer saved.")

                from scipy.sparse import save_npz

                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)

                    logging.info(f"Saving transformed train features to: {self.data_transformation_config.transformed_train_file_path}")
                    save_npz(self.data_transformation_config.transformed_train_file_path, x_train)

                    logging.info(f"Saving transformed test features to: {self.data_transformation_config.transformed_test_file_path}")
                    save_npz(self.data_transformation_config.transformed_test_file_path, x_test)

                    logging.info(f"Saving transformed train labels to: {self.data_transformation_config.transformed_train_label_path}")
                    self.utils.save_numpy_array_data(self.data_transformation_config.transformed_train_label_path, y_train)

                    logging.info(f"Saving transformed test labels to: {self.data_transformation_config.transformed_test_label_path}")
                    self.utils.save_numpy_array_data(self.data_transformation_config.transformed_test_label_path, y_test)

                    logging.info("All transformed files saved successfully.")

                except Exception as e:
                    logging.error("Failed during saving transformed features or labels.", exc_info=True)
                    raise SpamhamException(e, sys)

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_vectorizer_object_file_path=self.data_transformation_config.transformed_vectorizer_object_file_path,
                    transformed_encoder_object_file_path=self.data_transformation_config.transformed_encoder_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                    transformed_train_label_path=self.data_transformation_config.transformed_train_label_path,
                    transformed_test_label_path=self.data_transformation_config.transformed_test_label_path
                )

                logging.info("Data transformation completed.")
                return data_transformation_artifact

        except Exception as e:
            raise SpamhamException(e, sys)
