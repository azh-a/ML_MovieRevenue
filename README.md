# Movie Revenue Forecasting

# Datasets
Movie dataset: https://www.kaggle.com/datasets/utkarshx27/movies-dataset   <-- `movie_dataset.csv`

Economic dataset: Federal Reserve Economic Data (FRED)

GDP: https://fred.stlouisfed.org/series/GDP#

Inflation: https://fred.stlouisfed.org/series/FPCPITOTLZGUSA


# Data Preprocessing
```
import pandas as pd

df = pd.read_csv("movie_dataset")

df['release_date'] = pd.to_datetime(df['release_date'])
df['Year'] = df['release_date'].dt.year
df['Month'] = df['release_date'].dt.month
df = df.drop('homepage', axis=1)
df = df.drop('original_language', axis=1)
df = df.drop('popularity', axis=1)
df = df.drop('status', axis=1)
df = df.drop('vote_average', axis=1)
df = df.drop('vote_count', axis=1)

# df.to_csv('movie_only.csv', index=False) <-- cleaned movie only dataset

fred = pd.read_csv("FRED_Data.csv")

df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
fred['year'] = pd.to_datetime(fred['observation_date']).dt.year

merged = pd.merge(df, fred, on='year', how='left')

merged.head()

```
