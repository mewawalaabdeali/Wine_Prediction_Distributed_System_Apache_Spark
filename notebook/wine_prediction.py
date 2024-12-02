from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel

# Step 1: Initialize Spark Session
spark = SparkSession.builder \
    .appName("Wine_Quality_Prediction") \
    .config("spark.executor.memory", "2g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

# Step 2: Load Validation Data
# Adjust the path to your validation dataset
validation_data = spark.read.csv("hdfs://172.31.37.198:9000/dataset/ValidationDataset.csv", 
                                 header=True, inferSchema=True, sep=";")

# Clean column names to remove extra quotes
validation_data = validation_data.toDF(*[col.strip().replace('"', '') for col in validation_data.columns])

# Step 3: Assemble Features
feature_cols = [col for col in validation_data.columns if col != "quality"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
validation_data = assembler.transform(validation_data).select("features", "quality")

# Step 4: Load Trained Model
# Adjust the path to the saved model
model = RandomForestClassificationModel.load("hdfs://172.31.37.198:9000/models/WineQualityRFModel")

# Step 5: Make Predictions
predictions = model.transform(validation_data)

# Select relevant columns
results = predictions.select("features", "quality", "prediction", "probability")

# Step 6: Save Predictions
# Adjust the path to save predictions
results.write.csv("hdfs://172.31.37.198:9000/output/WinePredictions.csv", header=True, mode="overwrite")

# Stop Spark Session
spark.stop()
