# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("WineQualityModel").getOrCreate()

# File location and type
file_location = "/FileStore/tables/TrainingDataset.csv"
file_type = "csv"

df = spark.read.format('csv').load(path=file_location, header=True, inferSchema=True, sep=';')



# COMMAND ----------

df.printSchema()

# COMMAND ----------

for col_names in df.columns:
    clean_names=col_names.replace('"','').strip()
    df = df.withColumnRenamed(col_names, clean_names)

df.printSchema()


# COMMAND ----------

df.show(5)

# COMMAND ----------

df.columns

# COMMAND ----------

df.groupBy("quality").count().show()

# COMMAND ----------

from pyspark.sql.functions import col
majority_classes=[5,6]
minority_classes=[3,4,7,8]
df_majority = df.filter(col("quality").isin(majority_classes))
oversampled_dataframes = [df_majority]
majority_count=df_majority.count()
for minority_class in minority_classes:
    df_minority=df.filter(col("quality")==minority_class)
    minority_count=df_minority.count()

    if minority_count>0:
        oversample_ratio=majority_count/minority_count
        df_oversampled=df_minority.sample(withReplacement=True, fraction=oversample_ratio)
        oversampled_dataframes.append(df_oversampled)

    

df_balanced = oversampled_dataframes[0]
for oversampled_df in oversampled_dataframes[1:]:
    df_balanced=df_balanced.union(oversampled_df)

df_balanced.groupby("quality").count().show()


# COMMAND ----------

df=df_balanced

# COMMAND ----------

feature_cols = df.columns[:-1]
from pyspark.ml.feature import VectorAssembler, StandardScaler
vect_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# COMMAND ----------

df_features = vect_assembler.transform(df)
df_features.show(5)

# COMMAND ----------

df_final=df_features.select("features","quality")
df_final.show(5)

# COMMAND ----------

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scaler_model=scaler.fit(df_final)
df_train_scaled=scaler_model.transform(df_final)
df_scaled_final=df_train_scaled.select("scaledFeatures", "quality")

# COMMAND ----------

df_scaled_final.describe().show()

# COMMAND ----------

df_scaled_final.printSchema()

# COMMAND ----------

df_scaled_final.show(5)

# COMMAND ----------

df_scaled_final.groupBy("quality").count().show()

# COMMAND ----------

print(f"Training feature vector size: {len(df_scaled_final.select('scaledFeatures').first()[0])}")

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="quality")
model=lr.fit(df_scaled_final)

# COMMAND ----------

file_location = "/FileStore/tables/ValidationDataset.csv"
file_type = "csv"

df_test = spark.read.format('csv').load(path=file_location, header=True, inferSchema=True, sep=';')

# COMMAND ----------

for col_name in df_test.columns:
    new_clean_names = col_name.replace('"','').strip()
    df_test = df_test.withColumnRenamed(col_name, new_clean_names)

df_test.printSchema()

# COMMAND ----------

df_test.columns

# COMMAND ----------

from pyspark.ml.feature import StandardScaler
feature_cols_test=df_test.columns[:-1]
from pyspark.ml.feature import VectorAssembler
vect_assembler_test = VectorAssembler(inputCols=feature_cols_test, outputCol="features_assembled")
df_test = vect_assembler_test.transform(df_test)
df_features_final=df_test.select("features_assembled", "quality")


df_features_final.show(5)

# COMMAND ----------

df_features_final.printSchema()


# COMMAND ----------

scaler_test = StandardScaler(inputCol="features_assembled", outputCol="scaledFeatures", withStd=True, withMean=True)
scaler_model_test=scaler_test.fit(df_features_final)
df_test_scaled=scaler_model_test.transform(df_features_final)
df_test_final=df_test_scaled.select("scaledFeatures", "quality")

# COMMAND ----------

df_test_final.describe().show()

# COMMAND ----------

df_test_final.printSchema()

# COMMAND ----------

df_test_final.show(5)

# COMMAND ----------

print(f"Test feature vector size: {len(df_test_final.select('scaledFeatures').first()[0])}")

# COMMAND ----------

predictions = model.transform(df_test_final)
predictions.printSchema()


# COMMAND ----------

predictions.select("quality", "prediction").show(10)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
accuracy=evaluator.evaluate(predictions)
print(f"Validation Accuracy: {accuracy:.2f}")


# COMMAND ----------

predictions.select("quality", "prediction", "scaledFeatures").show(10)



# COMMAND ----------

evaluator.setMetricName("f1")
f1_score=evaluator.evaluate(predictions)
print(f"Validation F1 Score: {f1_score:.2f}")

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

rf=RandomForestClassifier(featuresCol="scaledFeatures", labelCol="quality", numTrees=50, maxDepth=10, seed=42)

# COMMAND ----------

rf_model=rf.fit(df_scaled_final)
rf_predictions=rf_model.transform(df_test_final)
rf_predictions.printSchema()

# COMMAND ----------

rf_predictions.select("quality", "prediction", "scaledFeatures").show(10)

# COMMAND ----------

rf_evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="rf_prediction", metricName="rf_accuracy")
rf_accuracy = evaluator.evaluate(rf_predictions)
print(f"Random Forest Validation Accuracy: {rf_accuracy:.2f}")

# COMMAND ----------

rf_evaluator.setMetricName("f1")
rf_f1_score = evaluator.evaluate(rf_predictions)
print(f"Random Forest Validation F1 Score: {rf_f1_score:.2f}")

# COMMAND ----------

rf_predictions.select("quality", "prediction", "scaledFeatures")

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

gbt = GBTClassifier(featuresCol="scaledFeatures", labelCol="quality", maxIter=100, maxDepth=5)

gbt_model=gbt.fit(df_scaled_final)

predictions_gbt = gbt_model.transform(df_test_final)



