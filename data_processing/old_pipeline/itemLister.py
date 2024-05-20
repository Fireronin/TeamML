#%%
import polars as pl
import os
from tqdm import tqdm
import json

folder_name = "../timeline_parquets_chunked"
files_list = sorted(os.listdir(folder_name))

items = set()

for file in tqdm(files_list):

    df = pl.scan_parquet(os.path.join(folder_name, file)).select(["itemId"]).unique().collect()

    for row in df.iter_rows():
        items.add(row[0])




items.remove(None)

items_dict = {k: v for k, v in zip(list(sorted(items)), range(1, len(items) + 1))}
items_data = json.dumps(items_dict)

# Write JSON data to a file
with open('../mapping_data/items_data.json', 'w') as file:
    file.write(items_data)
# %%
