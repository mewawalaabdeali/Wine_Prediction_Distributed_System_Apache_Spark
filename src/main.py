from pyspark.sql import SparkSession

#Initialize the spark session
spark = SparkSession.builder.appname("WineQualityData").getorCreate()