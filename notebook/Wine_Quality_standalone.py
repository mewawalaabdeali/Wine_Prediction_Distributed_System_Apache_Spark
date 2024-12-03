from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from datetime import datetime
import os
import shutil
import boto3

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_Training_ParamGrid") \
    .master("local[*]") \
    .getOrCreate()

print("Spark session initialized.")

# Step 2: Load and Clean Data
data_path = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/data/TrainingDataset.csv"
data = spark.read.csv(data_path, header=True, inferSchema=True, sep=";")
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])  # Clean column names
print(f"Data loaded from {data_path} with {data.count()} rows and {len(data.columns)} columns.")

# Step 3: Inspect Data
data.groupBy("quality").count().show()  # Display target distribution

# Step 4: Train-Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
print(f"Training set: {train_data.count()} rows, Test set: {test_data.count()} rows.")

# Step 5: Feature Engineering
feature_cols = [col for col in train_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

# Step 6: Model and Hyperparameter Tuning
rf = RandomForestClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 50, 100]) \
    .addGrid(rf.maxDepth, [5, 10, 20]) \
    .build()

crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                          numFolds=5)  # 5-fold cross-validation

# Step 7: Create Pipeline
pipeline = Pipeline(stages=[assembler, scaler, crossval])

# Step 8: Train Model
pipeline_model = pipeline.fit(train_data)
print("Model training completed with hyperparameter tuning.")

# Step 9: Evaluate Model
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction")
train_predictions = pipeline_model.transform(train_data)
test_predictions = pipeline_model.transform(test_data)

metrics = {
    "Training Accuracy": evaluator.evaluate(train_predictions, {evaluator.metricName: "accuracy"}),
    "Training F1 Score": evaluator.evaluate(train_predictions, {evaluator.metricName: "f1"}),
    "Test Accuracy": evaluator.evaluate(test_predictions, {evaluator.metricName: "accuracy"}),
    "Test F1 Score": evaluator.evaluate(test_predictions, {evaluator.metricName: "f1"}),
}

print("\nModel Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Step 10: Save Model
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_name = f"PipelineModel_Grid_{timestamp}"
model_dir = f"/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/models/{model_name}"
shutil.rmtree(model_dir, ignore_errors=True)  # Clear existing directory
pipeline_model.write().overwrite().save(model_dir)
print(f"Model saved locally at: {model_dir}")

# Step 11: Upload Model to S3
s3_client = boto3.client('s3')
bucket_name = "winepredictionabdeali"
for root, dirs, files in os.walk(model_dir):
    for file in files:
        full_path = os.path.join(root, file)
        s3_key = os.path.relpath(full_path, model_dir)
        s3_client.upload_file(full_path, bucket_name, f"Wine_models/{model_name}/{s3_key}")
print(f"Model uploaded to S3: s3://{bucket_name}/Wine_models/{model_name}/")

# Stop Spark Session
spark.stop()
print("Spark session stopped.")
