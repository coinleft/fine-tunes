# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# from datasets import load_dataset_builder
# from datasets import load_dataset

# ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
# print(ds)

import pandas as pd
df = pd.read_csv(
    "https://huggingface.co/datasets/imodels/credit-card/raw/main/train.csv")
df = pd.DataFrame(df)
print(df)