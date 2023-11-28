#%%
import polars as pl
import os
from tqdm import tqdm
import numpy as np

FOLDER_NAME = "new2"
# FILE_NUMBER = 10

# read
files_list = os.listdir(FOLDER_NAME)
files_list.sort()

# number_to_read = 100#os.listdir(folder_name).__len__()

# read first n files
# files_list = files_list[:number_to_read]

df = None

if not os.path.exists("parquets"):
    os.mkdir("parquets")

# n = np.ceil(len(files_list) / FILE_NUMBER)
# accumulated = 0
# saved = 0

tmp_df = pl.DataFrame()
# combine all files in the list
for file in tqdm(files_list):
    
    tmp_df = pl.read_csv(os.path.join(FOLDER_NAME, file))
    # drop all columns that start with summonerName
    tmp_df = tmp_df.drop(["summonerName1","summonerName2","summonerName3","summonerName4","summonerName5","summonerName6","summonerName7","summonerName8","summonerName9","summonerName10"])
    
    if df is None:
        df = tmp_df
        # accumulated = 1
        continue
    # reorder columns to match the order of the first file
    tmp_df = tmp_df.select(df.columns)
    
    # accumulated += 1
    
    df.vstack(tmp_df, in_place=True)

    # if accumulated >= n:
    #     df.write_parquet(f"parquets/new2_{saved}.parquet",compression="zstd",compression_level=5,use_pyarrow=True)
    #     #df.write_parquet(f"parquets/new2_{saved}.parquet",compression="uncompressed")
    #     accumulated = 0
    #     df = None
    #     saved+=1
    #     continue


# if accumulated > 0:
df.write_parquet(f"parquets/new2.parquet",compression="zstd",compression_level=5,use_pyarrow=True)

exit()
# read 

#%%
df = df.fill_null("")
df = df.fill_null(-1)
df = df.fill_nan(0)

# using variable counts divide columns into categorical and numerical
#%%
df.drop_in_place('ASSISTING_CHAMPS')
# get column names
column_names = df.columns

# get column counts but only unique values
column_counts = df.n_unique()

# sort column names by counts
column_counts = sorted([(col, df[col].n_unique()) for col in df.columns], key=lambda x: x[1])

# for each column create a map from value to index that will be used to convert to one hot encoding
maps = {}
for column in column_names:
    # get the unique values
    unique_values = df[column].unique().to_list()
    # create a map from value to index
    maps[column] = {value: index for index, value in enumerate(unique_values)}

cut_off = 100



# %%
# compute which columns are categorical and which are numerical
categorical_columns = []
numerical_columns = []
for column, count in column_counts:
    if count < cut_off:
        categorical_columns.append(column)
    else:
        numerical_columns.append(column)

numerical_columns.remove("matchId")
# %%
# normalize numerical columns
for column in numerical_columns:
    df = df.with_columns((df[column] - df[column].mean()) / df[column].std())
    

# %%
# convert categorical columns to numerical using maps
for column in categorical_columns:
    df = df.with_columns(df[column].map_dict(maps[column]))
# %%

df = df.with_columns(df.to_dummies(categorical_columns))
# remove categorical columns

df = df.drop(categorical_columns)

# %%
# group by matchId, convert to numpy array and save to file

output_array = []
for matchId, group in df.groupby("matchId"):
    group = group.drop("matchId")
    output_array.append(group.to_numpy())
    
#%%
# id of winning_team_0 ,winning_team_1  column
winning_team_0 = df.columns.index("winning_team_0")
winning_team_1 = df.columns.index("winning_team_1")

# split into input and output columns
x_array = []
y_array = []
max_length = max([len(match) for match in output_array])
padded_array_x = np.zeros((len(output_array), max_length, output_array[0].shape[1]-2))
padded_array_y = np.zeros((len(output_array), max_length, 2))

# pad all matches to max length with zeros
for i in range(len(output_array)):
    padded_array_x[i, :output_array[i].shape[0], :] = output_array[i][:, :-2]
    padded_array_y[i, :output_array[i].shape[0], 0] = output_array[i][:, winning_team_0]

#%% convert to float32
padded_array_x = padded_array_x.astype(np.float32)
padded_array_y = padded_array_y.astype(np.float32)

np.save("data.npy", padded_array_x)
np.save("labels.npy", padded_array_y)

# %%
