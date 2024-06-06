#%%
import pandas as pd 

imporantces_file = "feature_importance.csv"
df = pd.read_csv(imporantces_file)
print(df.head())
#   Feature  Importance
# 0    f169     12562.0
# 1    f227      9560.0
# 2    f285      9262.0
# 3    f575      8465.0
# %%
data_file = "transformed_data/timeline_0.parquet"
data_df = pd.read_parquet(data_file)
columns = data_df.columns
print(columns)
# %%
# match columns with feature names f{number} and importance 
# add a column with the feature name

def get_feature_name(feature):
    return columns[int(feature[1:])+1]

df['FeatureName'] = df['Feature'].apply(lambda x: get_feature_name(x))
# %%
df.head(50)
# %%
# save the file
df.to_csv("feature_importance_with_names.csv", index=False)
# %%
