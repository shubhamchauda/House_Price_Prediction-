from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt




spark = SparkSession.builder.appName('house_pridiction').getOrCreate()

from pyspark.ml.regression import LinearRegression

dataset =  spark.read.csv('./data/data.csv',header=True,inferSchema=True)
print(dataset.show())
fig,ax = plt.subplots()
ax.scatter(dataset['sqft_basement'], dataset['price'])

feature_input = VectorAssembler(inputCols=["bedrooms","sqft_living","sqft_lot","floors","view","condition","sqft_above","sqft_basement","yr_renovated",],outputCol="New_Feature")

output=feature_input.transform(dataset)
print(output.show())
finalized_data=output.select("New_Feature","price")

train_data,test_data=finalized_data.randomSplit([0.75,0.25])
regrassor  = LinearRegression(featuresCol="New_Feature",labelCol='price')

regrassor = regrassor.fit(train_data)


print(regrassor.coefficients)


pred_results=regrassor.evaluate(test_data)
print(pred_results.predictions.show(40))





