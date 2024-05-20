#%%
import polars as pl
import os
from tqdm import tqdm
import json


folder_name = "../timeline_parquets_chunked"
files_list = sorted(os.listdir(folder_name))
items = set()
ct = 0

match_df = pl.scan_parquet(os.path.join("../parquets", "match_basic.parquet"))

# print(match_df.count())

#create folder for new files
if not os.path.exists("../timeline_new"):
    os.makedirs("../timeline_new")

for file in tqdm(files_list):

    df = pl.scan_parquet(os.path.join(folder_name, file))
    # Item lister
    items_df =  df.select(["itemId"]).unique().collect()
    for row in items_df.iter_rows():
        items.add(row[0])


    df = df.with_columns(pl.col("winningTeam").fill_null(strategy="backward"))

    joined = df.lazy().join(match_df.lazy(), on='matchId', how='inner').collect()

    joined = joined.drop('__index_level_0__')

    joined.write_parquet(f"../timeline_new/{file}",compression="zstd",compression_level=10,use_pyarrow=True)



# %%
items.remove(None)

items_dict = {k: v for k, v in zip(list(sorted(items)), range(1, len(items) + 1))}
items_data = json.dumps(items_dict)

# Write JSON data to a file
with open('../mapping_data/items_data.json', 'w') as file:
    file.write(items_data)