# Movie Revenue Forecasting

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

df.to_csv('merged_da
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
Intercept: 5560348459.02354

Coefficients: [ 2.48192411e+00  1.26763544e+06 -2.88007926e+06  4.06472184e+04
  5.72218803e+03  1.35238570e+06 -2.11491806e+06  1.96295833e-01
 -3.53904488e+06  1.74434324e+07  5.35743845e+07  1.55404061e+06
 -3.98834331e+06  1.30682933e+07 -1.96612518e+07 -2.66054358e+06
  7.46916723e+06 -2.26433047e+05 -5.21473919e+06 -5.60141726e+07
  1.95073303e+07  1.92630465e+07 -4.66134304e+06 -9.94617139e+06
  9.82219294e+06 -2.26433047e+05  1.92630465e+07 -9.66994864e+06
 -2.07271350e+07 -8.56858496e+07]
 
Model Performance:  {'R²': 0.5478977511369492, 'RMSE': 110901903.66210844, 'MAE': 66885652.0365452}
