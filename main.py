from datasets import load_dataset
ds = load_dataset("Yelp/yelp_review_full")
print(ds)

train_ds = ds["train"]
print(train_ds[0])