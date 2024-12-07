import findspark
findspark.find()
findspark.init()

from datetime import datetime
import hashlib
import json
import os
import shutil
import boto3
from pyspark import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession

def main():
    print(">>>>>> PROGRAM STARTS")

    # Spark Setup
    conf = SparkConf().setAppName('WineQuality-Training')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    # Training CSV path
    trainPath = "s3a://winepredictionabdeali/TrainingDataset.csv"
    print(f">>>> Importing training dataset from: {trainPath}")

    # Model Path Creation
    bucket_name = "winepredictionabdeali"
    model_s3_path = "Testing_models"
    print(f">>>> Model will be uploaded to S3 at: s3://{bucket_name}/{model_s3_path}/")

    # Importing Training CSV
    df_train = spark.read.csv(trainPath, header=True, sep=";")
    df_train = df_train.toDF(*[col.strip().replace('"', '') for col in df_train.columns])
    print(f"Loaded training dataset with {df_train.count()} rows and {len(df_train.columns)} columns.")

    # Train-Test Split
    train_data, test_data = df_train.randomSplit([0.8, 0.2], seed=42)

    # Feature Engineering
    feature_cols = [col for col in train_data.columns if col != "quality"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

    # Model and Hyperparameter Tuning
    rf = RandomForestClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 50]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()

    crossval = CrossValidator(estimator=rf,
                              estimatorParamMaps=paramGrid,
                              evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                              numFolds=3)

    # Create Pipeline
    pipeline = Pipeline(stages=[assembler, scaler, crossval])

    # Train Model
    print("Starting model training with hyperparameter tuning...")
    pipeline_model = pipeline.fit(train_data)
    print("Model training completed.")

    # Generate a unique model name based on parameters
    param_str = json.dumps([{k.name: v for k, v in param.items()} for param in paramGrid], sort_keys=True)
    model_name = f"RandomForest_{hashlib.md5(param_str.encode()).hexdigest()[:8]}"
    local_model_dir = f"/tmp/{model_name}"
    shutil.rmtree(local_model_dir, ignore_errors=True)  # Clear existing directory

    # Save Model Locally
    pipeline_model.write().overwrite().save(local_model_dir)
    print(f"Model saved locally at: {local_model_dir}")

    # Upload the model directory to S3
    upload_model_to_s3(local_model_dir, bucket_name, model_s3_path, model_name)

    # Evaluate Model
    print("Starting model evaluation...")
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
    train_prediction = pipeline_model.transform(train_data)
    test_prediction = pipeline_model.transform(test_data)

    train_accuracy = evaluator.evaluate(train_prediction, {evaluator.metricName: "accuracy"})
    test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})

    # Enhanced Metrics Output
    print("\n")
    print("***********************************************************************")
    print("+++++++++++++++++++++++ Model Evaluation Metrics ++++++++++++++++++++++")
    print("***********************************************************************")
    print(f"[Train] Accuracy      : {train_accuracy:.4f}")
    print("-----------------------------------------------------------------------")
    print(f"[Test]  Accuracy      : {test_accuracy:.4f}")
    print("***********************************************************************")

    print(">>>>> TRAINING PROGRAM COMPLETE")

def upload_model_to_s3(local_model_dir, bucket_name, model_s3_path, model_name):
    """
    Upload the trained model to S3.
    """
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(local_model_dir):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, local_model_dir)
            s3_key = f"{model_s3_path}/{model_name}/{relative_path}"
            s3_client.upload_file(full_path, bucket_name, s3_key)
    print(f"Model uploaded to S3: s3://{bucket_name}/{model_s3_path}/{model_name}/")

if __name__ == "__main__":
    main()
