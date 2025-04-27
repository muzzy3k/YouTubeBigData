# %%
import pyspark.sql.functions as F

# %%
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = (
    SparkSession.builder
        .appName("YouTube_Model_USA")
        # allocate 8 GB to the driver JVM:
        .config("spark.driver.memory", "8g")
        # when running in local mode this also inflates the executor memory:
        .config("spark.executor.memory", "8g")
        # limit the size of results Spark will try to pull back to Python
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
)


# %%
# Load the data
import os
current_path = os.getcwd()
full_path = f'file://{current_path}/US_youtube_trending_data_clean.csv'
df = spark.read.csv(full_path, inferSchema = True, header = True)
df = df.drop("_c15")


# %%
df = df.filter(F.col("view_count") != 0)
df = df.filter(F.col("comments_disabled") != 'true')
# adding title statistics and seeing statistics 
# Create a column for the length of each title (number of characters)
df = df.withColumn("title_length", F.length("title"))

# Create a column for the word count (splitting on whitespace)
df = df.withColumn("word_count", F.size(F.split("title", " ")))
df = df.withColumn("publish_year", F.year("publishedAt")).withColumn("publish_month", F.month("publishedAt")).withColumn("publish_dayofweek", F.dayofweek("publishedAt")).withColumn("publish_hour", F.hour("publishedAt"))

# %%
#Catagorical columns
# mapping categoryid to its value through joining json file
categories_df = spark.read.option("multiLine", "true").json("US_category_id.json")
categories_df = categories_df.select(F.explode("items").alias("item"))
categories_df = categories_df.select(
    F.col("item.id").alias("categoryId"),
    F.col("item.snippet.title").alias("categoryTitle")
)

# Cast categoryId to integer so it matches your main DataFrame
categories_df = categories_df.withColumn("categoryId", F.col("categoryId").cast("integer"))

# %%
df_with_categories = df.join(categories_df, on="categoryId", how="left")
df = df.withColumn("dislikes_ratio", F.round(F.col("dislikes") / F.col("view_count"), 10))
df = df.withColumn("comments_ratio", F.col("comment_count")/F.col("view_count"))
df = df.withColumn("likes_ratio", F.round(F.col("likes") / F.col("view_count"), 2))
df_tags_split = df.withColumn("tag_array", F.split(F.col("tags"), r"\|"))

# %%
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import HashingTF, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# Define UDF to lowercase each element in the tag array
def lower_array(tags):
    if tags is None:
        return []
    return [tag.lower() for tag in tags]

lower_udf = F.udf(lower_array, ArrayType(StringType()))

# Create a new DataFrame so that original data is not overwritten.
# Lowercase the tag array column and store it in a new column 'tag_array_lower'
df_ml = df_tags_split.withColumn("tag_array_lower", lower_udf(F.col("tag_array")))


# %%
# Cast the selected numerical columns to string so we can treat them as categorical
df_ml = df_ml.withColumn("categoryID_str", F.col("categoryId").cast("string")) \
             .withColumn("title_length_str", F.col("title_length").cast("string")) \
             .withColumn("word_count_str", F.col("word_count").cast("string")) \
             .withColumn("publish_month_str", F.col("publish_month").cast("string")) \
             .withColumn("publish_dayofweek_str", F.col("publish_dayofweek").cast("string")) \
             .withColumn("publish_hour_str", F.col("publish_hour").cast("string"))

# Define the list of our categorical columns (as strings)
cat_cols = ["categoryID_str", "title_length_str", "word_count_str", 
            "publish_month_str", "publish_dayofweek_str", "publish_hour_str"]


# %%
# getting unique tags to use as buckets in hashingtf
unique_tag_count = df_ml.select(F.explode(F.col("tag_array_lower"))).distinct().count()
print("Unique tags:", unique_tag_count)


# %%
# For each categorical column, create a StringIndexer (with handleInvalid="keep" to avoid errors)
indexers = [StringIndexer(inputCol=col_name, outputCol=col_name + "_idx", handleInvalid="keep")
            for col_name in cat_cols]

# Next, use a OneHotEncoder to convert the indexed categories to one-hot encoded vectors
encoders = [OneHotEncoder(inputCol=col_name + "_idx", outputCol=col_name + "_ohe")
            for col_name in cat_cols]

# Create the HashingTF stage to convert the preprocessed tag array into a fixed-length feature vector.
hashing_tf = HashingTF(inputCol="tag_array_lower", outputCol="hashing_tf_features", numFeatures=262144)

# Assemble all features into a single vector. We use the HashingTF features
# as well as all the one-hot encoded categorical columns.
assembler_inputs = ["hashing_tf_features"] + [col_name + "_ohe" for col_name in cat_cols]
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# Create the pipeline by combining all stages
pipeline_stages = []
pipeline_stages += indexers
pipeline_stages += encoders
pipeline_stages.append(hashing_tf)
pipeline_stages.append(assembler)

pipeline = Pipeline(stages=pipeline_stages)

# %%
# Rename the target variable 'comments_ratio' to 'label' for cr modleing
df_ml_cr = df_ml.withColumnRenamed("comments_ratio", "label")

# Fit the pipeline model on the data and transform the DataFrame.
pipeline_model = pipeline.fit(df_ml_cr)
df_ml_cr_final = pipeline_model.transform(df_ml_cr)
# linear rgreression predict comments ratio

train, test = df_ml_cr_final.randomSplit([0.7, 0.3], seed=101)

# Build a Linear Regression model to predict the label (comments_ratio)
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol="label")
lr_model_cr = lr.fit(train)

# Evaluate the model on the test set
test_results = lr_model_cr.evaluate(test)
print("Building model for target: comments_ratio")
print("Test RMSE:", test_results.rootMeanSquaredError)
print("Test R2:", test_results.r2)

# %%
# likes ratio for modeling
df_ml_lr = df_ml.withColumnRenamed("likes_ratio", "label")

# Fit the pipeline model on the data and transform the DataFrame.
pipeline_model = pipeline.fit(df_ml_lr)
df_ml_lr_final = pipeline_model.transform(df_ml_lr)

# linear rgreression predict likes ratio

train, test = df_ml_lr_final.randomSplit([0.7, 0.3], seed=101)

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol="label")
lr_model_lr = lr.fit(train)

# Evaluate the model on the test set
test_results = lr_model_lr.evaluate(test)
print("Building model for target: likes_ratio")
print("Test RMSE:", test_results.rootMeanSquaredError)
print("Test R2:", test_results.r2)

# %%
pipeline_model.write().overwrite().save("models_USA/pipeline_model")
lr_model_cr.write().overwrite().save("models_USA/lr_model_cr")
lr_model_lr.write().overwrite().save("models_USA/lr_model_lr")


