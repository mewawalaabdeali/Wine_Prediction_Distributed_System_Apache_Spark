import findspark
findspark.find()
findspark.init()

from datetime import datetime
from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import boto3
import os
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
    s3ModelPath = "s3a://winepredictionabdeali/Testing_models"
    print(">>>> Model Path set: " + s3ModelPath)

    # Importing Training CSV
    df_train = spark.read.csv(trainPath, header=True, sep=";")
    df_train.printSchema()  # Column info
    df_train.show()

    # Data Cleaning
    df_train = df_train.withColumnRenamed('""""quality"""""', "myLabel")
    for column in df_train.columns:
        df_train = df_train.withColumnRenamed(column, column.replace('"', ''))

    # Converting to appropriate data types
    for column in df_train.columns:
        if column != "myLabel":
            df_train = df_train.withColumn(column, col(column).cast("double"))
    df_train = df_train.withColumn("myLabel", col("myLabel").cast("integer"))

    # Feature Engineering
    feature_cols = [col for col in df_train.columns if col != "myLabel"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df_train = assembler.transform(df_train)

    # Train-Test Split
    train_data, test_data = df_train.randomSplit([0.8, 0.2], seed=42)

    # Model 1 - Decision Tree
    print("Training DecisionTree model...")
    dt = DecisionTreeClassifier(labelCol="myLabel", featuresCol="features", maxDepth=10)
    dt_model = dt.fit(train_data)
    print("Model - DecisionTree Created")

    # Model 2 - RandomForest
    print("Training RandomForest model...")
    rf = RandomForestClassifier(labelCol="myLabel", featuresCol="features", numTrees=10, maxDepth=10)
    rf_model = rf.fit(train_data)
    print("Model - RandomForest Created")

    # Evaluate Models
    evaluator = MulticlassClassificationEvaluator(labelCol="myLabel", predictionCol="prediction", metricName="accuracy")
    accuracy_dt = evaluator.evaluate(dt_model.transform(test_data))
    accuracy_rf = evaluator.evaluate(rf_model.transform(test_data))

    # Select the best performing model based on accuracy
    if accuracy_dt > accuracy_rf:
        best_model = dt_model
        model_name = "decision_tree_model"
        print(f"DecisionTree model performs better with accuracy: {accuracy_dt:.4f}")
    else:
        best_model = rf_model
        model_name = "random_forest_model"
        print(f"RandomForest model performs better with accuracy: {accuracy_rf:.4f}")

    # Save the best model to S3
    save_model_to_s3(best_model, s3ModelPath, model_name)

    print(">>>>> Best model saved to S3")

    print(">>>>> TRAINING PROGRAM COMPLETE")


def save_model_to_s3(model, s3_path, model_name):
    """
    Save the trained model to the specified S3 path.
    """
    local_path = f"/tmp/{model_name}"
    model.write().overwrite().save(local_path)
    print(f"Model saved locally at {local_path}")

    # Upload model to S3
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(local_path):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, local_path)
            s3_key = f"{model_name}/{relative_path}"
            s3_client.upload_file(full_path, urlparse(s3_path).netloc, s3_key)
    print(f"Model uploaded to S3 at {s3_path}/{model_name}")


if __name__ == "__main__":
    main()
