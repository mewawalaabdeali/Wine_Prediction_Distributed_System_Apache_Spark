import findspark
findspark.find()
findspark.init()

from datetime import datetime
from io import StringIO
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.tree import RandomForest, RandomForestModel, DecisionTree
from pyspark.mllib.util import MLUtils
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.session import SparkSession
import boto3
import numpy as np
import os
import pandas as pd
import pyspark
import shutil
import sys
from urllib.parse import urlparse

def main():
    print(">>>>>> PROGRAM STARTS")

    # Spark Setup
    conf = SparkConf().setAppName('WineQuality-Training')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    # Training CSV path
    trainPath = "s3a://winepredictionabdeali/TrainingDataset.csv"
    print(">>>> Importing: " + trainPath)
    
    # Model Path Creation
    s3ModelPath = "s3a://winepredictionabdeali/models"
    print(">>>> Model Path set: " + s3ModelPath)

    # Importing Training CSV
    df_train = spark.read.csv(trainPath, header=True, sep=";")
    df_train.printSchema()  # Column info
    df_train.show()

    # Data Cleaning
    df_train = df_train.withColumnRenamed('""""quality"""""', "myLabel")
    # Removing the quotes from column names
    for column in df_train.columns:
        df_train = df_train.withColumnRenamed(column, column.replace('"', ''))

    # Converting to double/integer
    for idx, col_name in enumerate(df_train.columns):
        if idx not in [6 - 1, 7 - 1, len(df_train.columns) - 1]:
            df_train = df_train.withColumn(col_name, col(col_name).cast("double"))
        elif idx in [6 - 1, 7 - 1, len(df_train.columns) - 1]:
            df_train = df_train.withColumn(col_name, col(col_name).cast("integer"))

    # Convert DataFrame to RDD
    df_train = df_train.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))

    # Model 1 - Decision Tree
    print("Training DecisionTree model...")
    model_dt = DecisionTree.trainClassifier(df_train, numClasses=10, categoricalFeaturesInfo={},
                                            impurity='gini', maxDepth=10, maxBins=32)
    print("Model - DecisionTree Created")

    # Model 2 - RandomForest
    print("Training RandomForest model...")
    model_rf = RandomForest.trainClassifier(df_train, numClasses=10, categoricalFeaturesInfo={},
                                    numTrees=10, featureSubsetStrategy="auto",
                                    impurity='gini', maxDepth=10, maxBins=32)
    print("Model - RandomForest Created")

    # Evaluate both models
    evaluator = MulticlassClassificationEvaluator(labelCol="myLabel", predictionCol="prediction", metricName="accuracy")

    accuracy_dt = evaluator.evaluate(model_dt.transform(df_train))
    accuracy_rf = evaluator.evaluate(model_rf.transform(df_train))

    # Select the best performing model based on accuracy
    if accuracy_dt > accuracy_rf:
        best_model = model_dt
        model_name = "model_dt.model"
        print(f"DecisionTree model performs better with accuracy: {accuracy_dt}")
    else:
        best_model = model_rf
        model_name = "model_rf.model"
        print(f"RandomForest model performs better with accuracy: {accuracy_rf}")

    # Save the best model to S3
    model_path = os.path.join(s3ModelPath, model_name)
    save_model_to_s3(best_model, model_path)

    print(">>>>> Best model saved to S3")

    print(">>>>> TRAINING PROGRAM COMPLETE")

def save_model_to_s3(model, model_path):
    """
    Save the trained model to the specified S3 path.
    """
    try:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    main()
