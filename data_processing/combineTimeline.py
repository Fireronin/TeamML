#%%
import polars as pl
import os
from tqdm import tqdm

# conbine all files in timeline_parquets in one file

folder_name = "../timeline_parquets"
files_list = os.listdir(folder_name)
dfs = []
ct = 0
for file in tqdm(files_list):
    if file.endswith(".parquet"):
        df = pl.scan_parquet(os.path.join(folder_name, file))
        # sort df columns
        columns = df.columns
        columns.sort()
        df = df.select(columns)
        dfs.append(df)
        #print(df.shape)
        
    if len(dfs) >= 10:
        df = pl.concat(dfs)
        #df = df.collect()
        #print(df.shape)
        df.collect().write_parquet(f"../timeline_parquets_chunked/timeline_{ct}.parquet",compression="zstd",compression_level=10,use_pyarrow=True)
        dfs = []
        ct += 1
# %%
