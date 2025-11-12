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
from sklearn.preprocessing import MultiLabelBinarizer


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


df = pd.read_csv("merged_data.csv")
df.head()
df = df[df['Year'] >= 1960]
df = df[df['budget'] > 0]

df.drop("year", axis = 1)
print(df.isna().any())
df['runtime'].fillna(df['runtime'].mean(), inplace=True)



# Split the string column into lists
df['genres'] = df['genres'].fillna('').apply(lambda x: x.split())

# One-hot encode all unique genres
mlb = MultiLabelBinarizer()
genre_dummies = pd.DataFrame(mlb.fit_transform(df['genres']),
                             columns=mlb.classes_,
                             index=df.index)



# Compute average revenue per director
#director_mean_rev = df.groupby('director')['revenue'].mean()

# Map this back to df as a new numeric column
#df['director_mean_revenue'] = df['director'].map(director_mean_rev)

#global_mean = df['revenue'].mean()
#df['director_mean_revenue'] = df['director_mean_revenue'].fillna(global_mean)



# If you have Year and Month, make a sortable date; otherwise just use Year.
# (Replace Month with 1 if you don't have it.)
df['Month'] = df.get('Month', 1)
df['release_date'] = pd.to_datetime(dict(year=df['Year'], month=df['Month'], day=1))

# Sort within each director by release date so "past" is well-defined.
df = df.sort_values(['director', 'release_date'])

# Compute per-director cumulative mean of past revenues:
# expanding().mean() is over [0..i], so shift(1) makes it [0..i-1] (i.e., strictly past)
df['director_past_mean_revenue'] = (
    df.groupby('director')['revenue']
      .transform(lambda s: s.expanding().mean().shift(1))
)

# Cold-start directors (first movie) -> fill with a sensible prior (e.g., global mean)
global_mean = df['revenue'].mean()
df['director_past_mean_revenue'] = df['director_past_mean_revenue'].fillna(global_mean)

# Use this feature instead of the leaky one:
X = df[["budget","runtime","Year","Month","GDP","INFLATION","INTEREST_RATE","director_past_mean_revenue"]]



#X = df[["budget", "runtime", "Year", "Month", "GDP", "INFLATION", "INTEREST_RATE", "director_mean_revenue"]]


X = pd.concat([X, genre_dummies], axis=1)
print(X.head())


y = df["revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

lm = MyLinearModel()
lm.fit(X_train,y_train)
lm.summary()
preds = lm.predict(X_test)

metrics = lm.evaluate(X_test,y_test)
print("Model Performance: ", metrics)

```
