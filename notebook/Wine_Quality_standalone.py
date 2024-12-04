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

# Step 3: Train-Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 4: Feature Engineering
feature_cols = [col for col in train_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)

# Step 5: Model and Hyperparameter Tuning
rf = RandomForestClassifier(labelCol="quality", featuresCol="scaled_features", seed=42)
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 50, 100]) \
    .addGrid(rf.maxDepth, [5, 10, 20]) \
    .build()

crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                          numFolds=5)  # 5-fold cross-validation

# Step 6: Create Pipeline
pipeline = Pipeline(stages=[assembler, scaler, crossval])

# Step 7: Train Model
pipeline_model = pipeline.fit(train_data)
print("Model training completed with hyperparameter tuning.")

# Step 8: Evaluate Model on Test Data
test_predictions = pipeline_model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol="quality")
accuracy = evaluator.evaluate(test_predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(test_predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(test_predictions, {evaluator.metricName: "weightedRecall"})
f1_score = evaluator.evaluate(test_predictions, {evaluator.metricName: "f1"})

print(f"Test Metrics: Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1_score:.4f}")

# Step 9: Save Model Locally
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_name = f"PipelineModel_{timestamp}"
model_dir = f"/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/models/{model_name}"
shutil.rmtree(model_dir, ignore_errors=True)  # Clear existing directory
pipeline_model.write().overwrite().save(model_dir)
print(f"Model saved locally at: {model_dir}")

# Print the model directory name for Jenkins
print(f"MODEL_DIR={model_name}")  # This is the key line for Jenkins

# Step 10: Upload Model to S3
s3_client = boto3.client('s3')
bucket_name = "winepredictionabdeali"
for root, dirs, files in os.walk(model_dir):
    for file in files:
        full_path = os.path.join(root, file)
        s3_key = os.path.relpath(full_path, model_dir)
        s3_client.upload_file(full_path, bucket_name, f"Wine_models/{model_name}/{s3_key}")
print(f"Model uploaded to S3: s3://{bucket_name}/Wine_models/{model_name}/")

# Step 11: Stop Spark Session
spark.stop()
print("Spark session stopped.")
