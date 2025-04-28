import json
import gradio as gr
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.regression import LinearRegressionModel

# —————————————
# 1) Load category JSON and build name→id map (unchanged)
with open("US_category_id.json", "r") as f:
    cat_json = json.load(f)
name_to_id = {
    item["snippet"]["title"]: int(item["id"])
    for item in cat_json["items"]
    if item["snippet"].get("assignable", False)
}
category_names = sorted(name_to_id.keys())

# —————————————
# 2) Start Spark and load your saved models (unchanged)
spark = (
    SparkSession.builder
        .appName("YouTube_Gradio_Inference")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
)
pipeline_model = PipelineModel.load("models_GB/pipeline_model")
lr_model_cr   = LinearRegressionModel.load("models_GB/lr_model_cr")
lr_model_lr   = LinearRegressionModel.load("models_GB/lr_model_lr")

# —————————————
# 3) Updated predict() to include channel_title
def predict(channel_title: str, title: str, tags: str, category_name: str, publish_dt: str):
    # map category name → ID
    category_id = name_to_id.get(category_name)
    if category_id is None:
        raise ValueError(f"Unknown category: {category_name!r}")

    # parse datetime
    dt = datetime.strptime(publish_dt, "%Y-%m-%d %H:%M:%S")
    spark_dofw = (dt.isoweekday() % 7) + 1

    # build the same feature‐dict you used in training, including channelTitle lowercased
    row = {
        "tag_array_lower":        [t.lower() for t in tags.split("|") if t.strip()],
        "categoryID_str":         str(category_id),
        "title_length_str":       str(len(title)),
        "word_count_str":         str(len(title.split())),
        "publish_month_str":      str(dt.month),
        "publish_dayofweek_str":  str(spark_dofw),
        "publish_hour_str":       str(dt.hour),
        "channelTitle":           channel_title.lower()   
    }

    # assemble & predict
    input_df = spark.createDataFrame([row])
    feats    = pipeline_model.transform(input_df)
    cr_pred  = lr_model_cr.transform(feats).first().prediction
    lr_pred  = lr_model_lr.transform(feats).first().prediction

    return float(cr_pred), float(lr_pred)

# —————————————
# 4) Add a Gradio Textbox for Channel Title
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=1, label="Channel Title"),
        gr.Textbox(lines=2, label="Title"),
        gr.Textbox(lines=1, label="Tags (pipe-separated)"),
        gr.Dropdown(choices=category_names, label="Category"),
        gr.Textbox(lines=1, label="Publish datetime (YYYY-MM-DD HH:MM:SS)")         
    ],
    outputs=[
        gr.Number(label="Predicted Comments Ratio"),
        gr.Number(label="Predicted Likes Ratio"),
    ],
    title="YouTube Engagement Predictor"
)

if __name__ == "__main__":
    iface.launch(share=True)
