#%%
import polars as pl
import os
from tqdm import tqdm


folder_name = "../timeline_parquets_chunked"
files_list = sorted(os.listdir(folder_name))
# dfs = []
ct = 0

match_df = pl.scan_parquet(os.path.join("../parquets", "match_basic.parquet"))

# print(match_df.count())

#create folder for new files
if not os.path.exists("../timeline_new"):
    os.makedirs("../timeline_new")

for file in tqdm(files_list):

    df = pl.scan_parquet(os.path.join(folder_name, file))

    df = df.with_columns(pl.col("winningTeam").fill_null(strategy="backward"))

    joined = df.lazy().join(match_df.lazy(), on='matchId', how='inner').collect()

    joined = joined.drop('__index_level_0__')

    joined.write_parquet(f"../timeline_new/{file}",compression="zstd",compression_level=10,use_pyarrow=True)



# %%
