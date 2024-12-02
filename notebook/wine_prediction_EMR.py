from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel, DecisionTreeClassificationModel
import boto3
import os

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_Prediction") \
    .config("spark.master", "yarn") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

# Step 2: Download Model and Validation Dataset from S3
s3 = boto3.client('s3')
model_key = "models/latest_model.model"  # Replace with dynamic logic to get the latest model
validation_data_key = "ValidationDataset.csv"

local_model_path = "/tmp/latest_model.model"
local_validation_path = "/tmp/ValidationDataset.csv"

s3.download_file(Bucket="winepredictionabdeali", Key=model_key, Filename=local_model_path)
s3.download_file(Bucket="winepredictionabdeali", Key=validation_data_key, Filename=local_validation_path)

# Step 3: Load Validation Data
validation_data = spark.read.csv(local_validation_path, header=True, inferSchema=True, sep=";")
validation_data = validation_data.toDF(*[col.strip().replace('"', '') for col in validation_data.columns])

# Assemble Features
feature_cols = [col for col in validation_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
validation_data = assembler.transform(validation_data).select("features", "quality")

# Step 4: Load Model Dynamically
if "RandomForest" in model_key:
    model = RandomForestClassificationModel.load(local_model_path)
elif "DecisionTree" in model_key:
    model = DecisionTreeClassificationModel.load(local_model_path)
else:
    raise ValueError("Unsupported model type.")

# Step 5: Make Predictions
predictions = model.transform(validation_data)
results = predictions.select("features", "quality", "prediction", "probability")

# Step 6: Upload Predictions to S3
local_prediction_path = "/tmp/WinePredictions.csv"
results.write.csv(local_prediction_path, header=True, mode="overwrite")

s3.upload_file(Filename=local_prediction_path, Bucket="winepredictionabdeali", Key="output/WinePredictions.csv")

# Stop Spark Session
spark.stop()
