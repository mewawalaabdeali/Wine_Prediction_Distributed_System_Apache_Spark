from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_single_machine") \
    .master("local[*]") \
    .getOrCreate()

# Step 2: Load and Clean Data
data = spark.read.csv("/home/ubuntu/Wine_Prediction_Distributed_System_Apache_Spark/notebook/TrainingDataset.csv", header=True, inferSchema=True, sep=";")

# Clean column names to remove extra quotes
data = data.toDF(*[col.strip().replace('"', '') for col in data.columns])

# Verify 'quality' column exists
if 'quality' not in data.columns:
    raise ValueError("The dataset must contain a 'quality' column.")

# Step 3: Train-Test Split
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Step 4: Assemble Features and Labels
feature_cols = [col for col in train_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

train_data = assembler.transform(train_data).select("features", "quality")
test_data = assembler.transform(test_data).select("features", "quality")

# Step 5: Train Random Forest Model
rf = RandomForestClassifier(labelCol="quality", featuresCol="features", numTrees=10)
model = rf.fit(train_data)

# Step 6: Evaluate Model on Test Data
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# Step 7: Evaluate Model on Training Data (Optional)
train_predictions = model.transform(train_data)
train_accuracy = evaluator.evaluate(train_predictions)
print(f"Model Accuracy on Training Data: {train_accuracy * 100:.2f}%")

# Stop Spark Session
spark.stop()