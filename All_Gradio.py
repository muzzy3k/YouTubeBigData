import json
import gradio as gr
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.regression import LinearRegressionModel

# —————————————
# 1) Define your regions and point to their files
REGIONS = {
    "CA": {
        "cat_json": "CA_category_id.json",
        "pipeline": "models_CA/pipeline_model",
        "lr_cr":    "models_CA/lr_model_cr",
        "lr_lr":    "models_CA/lr_model_lr",
    },
    "US": {
        "cat_json": "US_category_id.json",
        "pipeline": "models_USA/pipeline_model",
        "lr_cr":    "models_USA/lr_model_cr",
        "lr_lr":    "models_USA/lr_model_lr",
    },
    "GB": {
        "cat_json": "GB_category_id.json",
        "pipeline": "models_GB/pipeline_model",
        "lr_cr":    "models_GB/lr_model_cr",
        "lr_lr":    "models_GB/lr_model_lr",
    }
}

# —————————————
# 2) Preload all the JSON→map and model objects
spark = (
    SparkSession.builder
        .appName("YouTube_Gradio_Inference_AllRegions")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
)

# Containers keyed by region code:
name_to_id_map = {}
category_names_map = {}
pipeline_models = {}
lr_models_cr = {}
lr_models_lr = {}

for region, paths in REGIONS.items():
    # load JSON
    with open(paths["cat_json"], "r") as f:
        cat_json = json.load(f)
    # build name→ID map
    nti = {
        item["snippet"]["title"]: int(item["id"])
        for item in cat_json["items"]
        if item["snippet"].get("assignable", False)
    }
    name_to_id_map[region] = nti
    category_names_map[region] = sorted(nti.keys())

    # load the Spark ML models
    pipeline_models[region] = PipelineModel.load(paths["pipeline"])
    lr_models_cr[region]   = LinearRegressionModel.load(paths["lr_cr"])
    lr_models_lr[region]   = LinearRegressionModel.load(paths["lr_lr"])


# —————————————
# 3) The unified predict() function
def predict(region, title, tags, category_name, publish_dt):
    # pick the right maps + models
    nti    = name_to_id_map[region]
    pipe   = pipeline_models[region]
    lr_cr  = lr_models_cr[region]
    lr_lr  = lr_models_lr[region]

    # category → ID
    category_id = nti.get(category_name)
    if category_id is None:
        raise ValueError(f"Unknown category {category_name!r} for region {region!r}")

    # parse datetime string
    dt = datetime.strptime(publish_dt, "%Y-%m-%d %H:%M:%S")
    spark_dofw = (dt.isoweekday() % 7) + 1

    # build feature dict exactly as in training
    row = {
        "tag_array_lower":       [t.lower() for t in tags.split("|") if t.strip()],
        "categoryID_str":        str(category_id),
        "title_length_str":      str(len(title)),
        "word_count_str":        str(len(title.split())),
        "publish_month_str":     str(dt.month),
        "publish_dayofweek_str": str(spark_dofw),
        "publish_hour_str":      str(dt.hour),
    }

    # spark → features → preds
    input_df = spark.createDataFrame([row])
    feats    = pipe.transform(input_df)
    cr_pred  = lr_cr.transform(feats).first().prediction
    lr_pred  = lr_lr.transform(feats).first().prediction

    return float(cr_pred), float(lr_pred)


# —————————————
# 4) Build the Gradio Blocks interface
with gr.Blocks(title="YouTube Engagement Predictor (All Regions)") as demo:
    # region selector
    region_dd = gr.Dropdown(
        choices=list(REGIONS.keys()),
        value="CA",
        label="Region"
    )
    # category selector, will be updated
    category_dd = gr.Dropdown(
        choices=category_names_map["CA"],
        label="Category"
    )
    title_tb    = gr.Textbox(lines=2, label="Title")
    tags_tb     = gr.Textbox(lines=1, label="Tags (pipe-separated)")
    dt_tb       = gr.Textbox(lines=1, label="Publish datetime (YYYY-MM-DD HH:MM:SS)")
    cr_out      = gr.Number(label="Predicted Comments Ratio")
    lr_out      = gr.Number(label="Predicted Likes Ratio")
    btn         = gr.Button("Predict")

    # when region changes, swap out the category choices
    region_dd.change(
        fn=lambda r: gr.update(choices=category_names_map[r], value=category_names_map[r][0]),
        inputs=[region_dd],
        outputs=[category_dd]
    )

    # wire up the predict button
    btn.click(
        fn=predict,
        inputs=[region_dd, title_tb, tags_tb, category_dd, dt_tb],
        outputs=[cr_out, lr_out]
    )

if __name__ == "__main__":
    demo.launch(share=True)
