# Movie Revenue Forecasting

# Milestone 1 Progress
- Our first step was to merge our movie dataset with our economic datasets. We got data from 1960-2020 for the US GDP, Inflation Rate, and Interest Rate. We first merged these three datasets into one which you can find in FRED_Data_all.csv file in our datasets folder. Then, we merged this file with our movie dataset by mapping the release year of each movie to its corresponding economic variables. So a movie released in 2015 now has data for our three economic variables from 2015. 
- Our next step was to run a simple Linear Regression. 

# Datasets
Movie dataset: https://www.kaggle.com/datasets/utkarshx27/movies-dataset   <-- `movie_dataset.csv`

Economic dataset: Federal Reserve Economic Data (FRED)

GDP: https://fred.stlouisfed.org/series/GDP#

Inflation: https://fred.stlouisfed.org/series/FPCPITOTLZGUSA

Interest Rates: https://fred.stlouisfed.org/series/INTDSRUSM193N#


# Data Preprocessing
```
import pandas as pd

df = pd.read_csv("movie_dataset.csv")

df['release_date'] = pd.to_datetime(df['release_date'])
df['Year'] = df['release_date'].dt.year
df['Month'] = df['release_date'].dt.month
df = df.drop('homepage', axis=1)
df = df.drop('original_language', axis=1)
df = df.drop('popularity', axis=1)
df = df.drop('status', axis=1)
df = df.drop('vote_average', axis=1)
df = df.drop('vote_count', axis=1)

# df.to_csv('movie_only.csv', index=False) #<-- cleaned movie only dataset
```

```
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
import pandas as pd

fred = pd.read_csv("FRED_Data_all.csv")
df = pd.read_csv('movie_only.csv')

# merging
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
fred['year'] = pd.to_datetime(fred['observation_date']).dt.year
df = pd.merge(df, fred, on='year', how='left')

df = df[df['Year'] >= 1960]  # FRED doesn't have data older than 1960
df = df[df['budget'] > 0]    # weird cols

df.drop("year", axis = 1)


print(df.isna().any())
df['runtime'].fillna(df['runtime'].mean(), inplace=True)  # fill in empty runtime

# deal with genres col
df = df.dropna(subset=['genres']).copy()
df['genres'] = df['genres'].fillna('').astype(str).str.split()
mlb = MultiLabelBinarizer()
genre_dummies = pd.DataFrame(
    mlb.fit_transform(df['genres']),
    columns=mlb.classes_,
    index=df.index
)
df = pd.concat([df, genre_dummies], axis=1)

# past mean revenue of directors
df = df.sort_values(["director", "Year", "Month"])
df["director_past_avg_rev"] = (
    df.groupby("director")["revenue"]
      .apply(lambda s: s.shift().expanding().mean())
      .reset_index(level=0, drop=True)
)
df["director_past_avg_rev"] = df["director_past_avg_rev"].fillna(0)


# normalize
scaler = StandardScaler()
num_cols = ['budget','revenue','runtime','director_past_avg_rev', 'GDP','INFLATION','INTEREST_RATE']
df[num_cols] = scaler.fit_transform(df[num_cols])

df.to_csv('merged_data.csv', index=False) #<-- cleaned movie/econ dataset
df.head()
```

# Linear Regression
```
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


class LinearModel:
    def __init__(self):
        self.model = LinearRegression()
    
    def fit(self, X, y):
        """Train the model on data (X, y)."""
        self.model.fit(X, y)
    
    def predict(self, X):
        """Predict outputs for new inputs X."""
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Compute performance metrics:
          - R² 
          - RMSE 
          - MAE 
        """
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        mae = mean_absolute_error(y, y_pred)
        return {"R²": r2, "RMSE": rmse, "MAE": mae}
    
    def summary(self):
        """Display learned coefficients and intercept."""
        print("Intercept:", self.model.intercept_)
        print("Coefficients:", self.model.coef_)



# features
X = df[["budget","runtime","Year","Month","GDP","INFLATION","INTEREST_RATE","director_past_avg_rev"]]
X = pd.concat([X, genre_dummies], axis=1)
#print(X.head())
#print(X.isna().any())

y = df["revenue"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

lm = LinearModel()
lm.fit(X_train,y_train)
lm.summary()
preds = lm.predict(X_test)

metrics = lm.evaluate(X_test,y_test)
print("Model Performance: ", metrics)
```

# Current Results
Intercept: 33.26534681814592

Coefficients: [ 0.59224283  0.12185578 -0.01660025  0.00134086  0.14073538  0.02538885
 -0.03389948  0.08851087 -0.02201395  0.0870499   0.35864228  0.00997171
 -0.02433448  0.0381424  -0.09421852  0.01157278  0.04141991 -0.00382325
 -0.04324956 -0.30318002  0.10297418  0.09111763 -0.03673845 -0.06920508
  0.05035054 -0.00382325  0.09111763 -0.03687254 -0.13805464 -0.38399001]
  
Model Performance:  {'R²': 0.5543968272471183, 'RMSE': 0.6757270536982963, 'MAE': 0.4061342071169473}
