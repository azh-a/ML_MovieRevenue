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
fred = pd.read_csv("FRED_Data_all.csv")

df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
fred['year'] = pd.to_datetime(fred['observation_date']).dt.year

merged = pd.merge(df, fred, on='year', how='left')

# merged.to_csv('merged_data.csv', index=False) #<-- cleaned movie/econ dataset

merged.head()
```

# Linear Regression
```
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

class MyLinearModel:
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
          - R² (coefficient of determination)
          - RMSE (root mean squared error)
          - MAE (mean absolute error)
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


df = df = pd.read_csv("merged_data.csv")
X = df[c("budget", "runtime", "Year", "Month", "GDP", "INFLATION", "INTREST_RATE")]
y = df["revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

```
