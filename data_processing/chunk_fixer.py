from tqdm import tqdm
import polars as pl
import os

folder_name = "../transformed_data"
files_list = sorted(os.listdir(folder_name))

# MAX_LEN = 2928

# shapes = []

temp_df = pl.DataFrame({})
file_count = 0

for file in tqdm(files_list):

    df = pl.scan_parquet(os.path.join(folder_name, file)).collect()

    df = df.with_columns(pl.exclude('matchId').cast(pl.Float64))

    df = df.filter((pl.col('winningTeam') == 100) | (pl.col('winningTeam') == 200))

    df = df.insert_column(1, df.select((pl.col('1_totalGold') + pl.col('2_totalGold') + pl.col('3_totalGold') + pl.col('4_totalGold') + pl.col('5_totalGold')).alias('100_teamTotalGold')).to_series())
    df = df.insert_column(2, df.select((pl.col('6_totalGold') + pl.col('7_totalGold') + pl.col('8_totalGold') + pl.col('9_totalGold') + pl.col('10_totalGold')).alias('200_teamTotalGold')).to_series())
    df = df.insert_column(3, df.select((pl.col('100_teamTotalGold') - pl.col('200_teamTotalGold')).alias('teamGoldDiff')).to_series())

    df = df.insert_column(4, df.select((pl.col('1_xp') + pl.col('2_xp') + pl.col('3_xp') + pl.col('4_xp') + pl.col('5_xp')).alias('100_teamXp')).to_series())
    df = df.insert_column(5, df.select((pl.col('6_xp') + pl.col('7_xp') + pl.col('8_xp') + pl.col('9_xp') + pl.col('10_xp')).alias('200_teamXp')).to_series())
    df = df.insert_column(6, df.select((pl.col('100_teamXp') - pl.col('200_teamXp')).alias('teamXpDiff')).to_series())

    temp_df = pl.concat([temp_df, df])

    if temp_df['matchId'].n_unique() >= 1000:
        chunk_matches = temp_df['matchId'].unique()[:1000]
        
        chunk = temp_df['matchId'].is_in(chunk_matches)

        new_df = temp_df.filter(chunk)

        temp_df = temp_df.filter(~chunk)

        new_df.write_parquet(f"../filtered_data/timeline_{file_count}.parquet", compression="zstd", compression_level=10, use_pyarrow=True)

        file_count += 1


if not temp_df.is_empty():
    temp_df.write_parquet(f"../filtered_data/timeline_{file_count}.parquet", compression="zstd", compression_level=10, use_pyarrow=True)