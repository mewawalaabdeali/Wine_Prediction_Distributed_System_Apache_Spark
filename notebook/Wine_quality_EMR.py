import findspark
findspark.init()

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import hashlib
import boto3
import os
import shutil
from datetime import datetime

def main():
    print(">>>>>> PROGRAM STARTS")

    # Spark Setup
    conf = SparkConf().setAppName('WineQuality-Training')
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    # Training CSV path
    trainPath = "s3://winepredictionabdeali/TrainingDataset.csv"
    print(f">>>> Importing: {trainPath}")

    # Model Path Creation
    s3ModelPath = "s3://winepredictionabdeali/Testing_models"
    print(f">>>> Model Path set: {s3ModelPath}")

    # Step 1: Load and Clean Data
    df_train = spark.read.csv(trainPath, header=True, sep=";")
    df_train = df_train.toDF(*[col.strip().replace('"', '') for col in df_train.columns])
    
    # Explicitly cast feature columns to double
    numeric_columns = [col for col in df_train.columns if col != "quality"]
    for col_name in numeric_columns:
        df_train = df_train.withColumn(col_name, col(col_name).cast("double"))
    
    # Cast label column to integer
    df_train = df_train.withColumn("quality", col("quality").cast("integer"))

    print("Schema after type casting:")
    df_train.printSchema()

    # Train-Test Split
    train_data, test_data = df_train.randomSplit([0.8, 0.2], seed=42)
    print(f"Loaded training dataset with {df_train.count()} rows and {len(df_train.columns)} columns.")

    # Step 2: Feature Engineering
    assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

    # Step 3: Model and Hyperparameter Tuning
    rf = RandomForestClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 50]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()
    crossval = CrossValidator(estimator=rf,
                              estimatorParamMaps=paramGrid,
                              evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                              numFolds=3)

    # Step 4: Create Pipeline
    pipeline = Pipeline(stages=[assembler, scaler, crossval])

    # Step 5: Train Model
    print("Starting model training with hyperparameter tuning...")
    pipeline_model = pipeline.fit(train_data)
    print("Model training completed.")

    # Step 6: Save Model with Unique Identifier
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name = f"RandomForest_{timestamp}"
    local_model_dir = f"/tmp/{model_name}"
    shutil.rmtree(local_model_dir, ignore_errors=True)
    pipeline_model.write().overwrite().save(local_model_dir)
    print(f"Model saved locally at: {local_model_dir}")

    # Upload model to S3
    print(f"Uploading model to S3: {s3ModelPath}/{model_name}/")
    s3_client = boto3.client('s3')
    for root, dirs, files in os.walk(local_model_dir):
        for file in files:
            full_path = os.path.join(root, file)
            s3_key = f"{model_name}/{os.path.relpath(full_path, local_model_dir)}"
            s3_client.upload_file(full_path, "winepredictionabdeali", f"Testing_models/{s3_key}")
    print(f"Model uploaded successfully to S3: {s3ModelPath}/{model_name}/")

    # Step 7: Evaluate Model
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
    train_prediction = pipeline_model.transform(train_data)
    test_prediction = pipeline_model.transform(test_data)

    train_f1 = evaluator.evaluate(train_prediction, {evaluator.metricName: "f1"})
    test_f1 = evaluator.evaluate(test_prediction, {evaluator.metricName: "f1"})
    train_accuracy = evaluator.evaluate(train_prediction, {evaluator.metricName: "accuracy"})
    test_accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})

    # Print Evaluation Metrics
    print("\n")
    print("***********************************************************************")
    print("+++++++++++++++++++++++ Model Evaluation Metrics ++++++++++++++++++++++")
    print("***********************************************************************")
    print(f"[Train] F1 Score      : {train_f1:.4f}")
    print(f"[Train] Accuracy      : {train_accuracy:.4f}")
    print("-----------------------------------------------------------------------")
    print(f"[Test]  F1 Score      : {test_f1:.4f}")
    print(f"[Test]  Accuracy      : {test_accuracy:.4f}")
    print("***********************************************************************")

    print(">>>>> TRAINING PROGRAM COMPLETE")
    spark.stop()
    print("Spark session stopped.")

if __name__ == "__main__":
    main()
