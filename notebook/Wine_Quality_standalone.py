from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
import boto3
import os

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_Single_Machine_Enhanced") \
    .master("local[*]") \
    .getOrCreate()

# Step 2: Load and Clean Data
data = spark.read.csv("/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/data/TrainingDataset.csv", 
                      header=True, inferSchema=True, sep=";")

# Clean column names
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])

# Verify 'quality' column exists
if 'quality' not in data.columns:
    raise ValueError("The dataset must contain a 'quality' column.")

# Step 3: Train-Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 4: Assemble Features
feature_cols = [col for col in train_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
train_data = assembler.transform(train_data).select("features", "quality")
test_data = assembler.transform(test_data).select("features", "quality")

# Step 5: Feature Scaling
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
scaler_model = scaler.fit(train_data)
train_data = scaler_model.transform(train_data).select("scaled_features", "quality").withColumnRenamed("scaled_features", "features")
test_data = scaler_model.transform(test_data).select("scaled_features", "quality").withColumnRenamed("scaled_features", "features")

# Step 6: Initialize Models
rf = RandomForestClassifier(labelCol="quality", featuresCol="features", seed=42)
dt = DecisionTreeClassifier(labelCol="quality", featuresCol="features", seed=42)

# Step 7: Hyperparameter Tuning for Random Forest
paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

train_val_split = TrainValidationSplit(estimator=rf,
                                       estimatorParamMaps=paramGrid_rf,
                                       evaluator=MulticlassClassificationEvaluator(labelCol="quality", metricName="accuracy"),
                                       trainRatio=0.8)

# Train Random Forest
rf_model = train_val_split.fit(train_data).bestModel

# Train Decision Tree (without hyperparameter tuning for simplicity)
dt_model = dt.fit(train_data)

# Step 8: Evaluate Models
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")

rf_predictions = rf_model.transform(test_data)
dt_predictions = dt_model.transform(test_data)

rf_accuracy = evaluator.evaluate(rf_predictions)
dt_accuracy = evaluator.evaluate(dt_predictions)

print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")
print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")

# Step 9: Select Best Model
best_model = rf_model if rf_accuracy >= dt_accuracy else dt_model
best_model_name = "RandomForest" if rf_accuracy >= dt_accuracy else "DecisionTree"
print(f"Best Model: {best_model_name}")

# Initialize S3 client
s3_client = boto3.client('s3')
bucket_name = "winepredictionabdeali"

# Define paths
model_dir = "/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/models"
model_filename = f"{best_model_name}_model"
local_model_path = os.path.join(model_dir, model_filename)  # Save in the models directory
s3_model_path = f"Wine_models/{model_filename}"  # Path in S3

# Ensure the local directory exists
os.makedirs(model_dir, exist_ok=True)

# Save the model locally
best_model.write().overwrite().save(local_model_path)
print(f"Model saved locally at: {local_model_path}")

# Upload the model to S3
s3_client.upload_file(local_model_path, bucket_name, s3_model_path)
print(f"Best model saved to S3 at: s3://{bucket_name}/{s3_model_path}")

# Step 11: Stop Spark Session
spark.stop()