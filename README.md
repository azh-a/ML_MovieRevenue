# Movie Revenue Forecasting
# Overview
- Our project aims to address the broad issue of future revenue prediction. Specifically, we will focus on the movie industry, and we want to predict how much revenue a film is likely to make based on features including budget, runtime, genre, directors past revenue, and text embeddings of the overview of each movie. We also want to incorporate elements of the US economy into our model to possibly help predict the success of a movie. We incorporated the past 80 years of the US GDP, inflation, and interest rate. We found that in general this task is challenging espicially when excluding variables that relate to the movies success and popularity. It is also difficult to quantify things like directors, casts, plots, and production companies into our model, which we know (from previous studies) would improve model performance. We acheived an R^2 value of 0.6 with our best model, which means our model can make educated guesses, but is still not extremely accurate. We discovered that the economic features we used in our model did not have a significant effect on our predictions. Our most important features were budget, genre, and the text embeddings (plot summaries). 
# Replication Instructions
- download `merged_data.csv` in datasets/final folder
- download and run `replication.ipynb`
# Future Directions
- The most influential next step would be to include more features from the movies into our model. Things like directors, cast, and production companies are all incredibly important factors in a movies success, but are difficult to incorporate into a model since there are so many different people and companies that go into making each movie. The inclusion of factors like this would help our predictions. Another path for the future would be to run this model on different movie and economic datasets from other countries. It would be interesting to see how our model would perform on another countries statistics and data. Another addition could be to find and include the marketing and advertisement features of a movie. Movies that are marketed well may perform very well even if the movie itself is not exceptional. This extra feature would likely improve our results. 
# Contributions
- Vincent - Researched background information on the subject, found similar studies that helped us decide how to make our model and gave us material to compare to. Got economic datasets off FRED for the annual time periods we needed. Merged these economic datasets with our movie dataset so that each movie had economic data based on its release year. Helped test our early Linear Regression model by tweaking features and normalizing data. Helped deal with the genre situation where we ended up one-hot encoding to properly tag movies. Also fought for a while with our past directors revenue feature which was hard to get it to only incorporate a directors previous earnings. Translated a lot of our work and results into our poster, mostly the intro, background, and summary+conclusion sections. We worked every class and lab we were given the opprotunity, and out of class work were mostly small touch ups and nothing major.            
- Angie - Wrote introduction section (for project proposal) and refined research aim/goals. Researched viable movie datasets we could use in our project. Outlined most of the methodology for experimental setup and model development. Wrote most of the code for data preprocessing, MPNet embeddings extraction, and model training/testing. Created results table/figures, and did a lot of code debugging. Helped write summary+conclusion and future directions section.

# Everything Below This Point are Previous Steps

# Milestone 1 Progress
- Our first step was to merge our movie dataset with our economic datasets. We got data from 1960-2020 for the US GDP, Inflation Rate, and Interest Rate. We first merged these three datasets into one which you can find in FRED_Data_all.csv file in our datasets folder. Then, we merged this file with our movie dataset by mapping the release year of each movie to its corresponding economic variables. So a movie released in 2015 now has data for our three economic variables from 2015. 
- Our next step was to clean our data and run a Linear Regression model. We dropped a handfull of values that we found unimportant, and then merged our data as described above. We one-hot encoded our genres so that we can tag movies approprately, then added these values to the merged dataset as well. Along with that, we calculated the past revenue for the directors to avoid leakage but still include a directors past success in our model. And then lastly we normalized our numerical values. Then we ran our Linear Regression, which had a medicore performace. The results are at the bottom.
- Moving forward, we want to run multiple different models and explore other variables we can add to our model.

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
X = df[['budget','runtime','Year','Month','GDP','INFLATION','INTEREST_RATE','director_past_avg_rev', 
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 
        'Fiction', 'Foreign', 'History', 'Horror', 'Movie', 'Music', 'Mystery', 'Romance', 'Science', 'TV', 
        'Thriller', 'War', 'Western']]
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

XGBoost
```
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

class XGBModel:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = mean_squared_error(y, y_pred, squared=False)
        mae = mean_absolute_error(y, y_pred)
        return {"R²": r2, "RMSE": rmse, "MAE": mae}
    
    def summary(self, feature_names=None, top_k=20):
        importances = self.model.feature_importances_
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importances))]
        
        feats = list(zip(feature_names, importances))
        feats = sorted(feats, key=lambda x: x[1], reverse=True)
        
        print(f"Top {top_k} features by importance:")
        for name, imp in feats[:top_k]:
            print(f"{name}: {imp:.4f}")

# Training with hyperparameter tuning & cross-validation 
xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

# Hyperparameter grid
param_dist = {
    "n_estimators":     [200, 400, 600, 800],
    "max_depth":        [3, 4, 5, 6, 8],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.001, 0.005],
    "subsample":        [0.6, 0.8, 1.0, 0.7],
    "colsample_bytree": [0.6, 0.8, 1.0, 0.7],
    "min_child_weight": [1, 3, 5, 10, 2, 8],
    "reg_alpha":        [0, 0.01, 0.1, 1, 0.05],   
    "reg_lambda":       [0.1, 1, 5, 10, 0.5, 2],      
}

# 3-fold CV (randomized)
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=60,                      
    scoring="neg_root_mean_squared_error",
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=42,
)

random_search.fit(X_train, y_train)

print("Best params:", random_search.best_params_)
print("Best CV RMSE:", -random_search.best_score_)

# evaluate best model on test 
best_xgb = random_search.best_estimator_

y_pred = best_xgb.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

print("Test R²:  ", r2)
print("Test RMSE:", rmse)
print("Test MAE: ", mae)

# see top 20 most important features 
importances = best_xgb.feature_importances_
feat_importance = sorted(
    zip(X.columns, importances),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 20 features:")
for name, imp in feat_importance[:20]:
    print(f"{name:30s} {imp:.4f}")
```

# Current Results
Linear Regression:

Intercept: 33.26534681814592

Coefficients: [ 0.59224283  0.12185578 -0.01660025  0.00134086  0.14073538  0.02538885
 -0.03389948  0.08851087 -0.02201395  0.0870499   0.35864228  0.00997171
 -0.02433448  0.0381424  -0.09421852  0.01157278  0.04141991 -0.00382325
 -0.04324956 -0.30318002  0.10297418  0.09111763 -0.03673845 -0.06920508
  0.05035054 -0.00382325  0.09111763 -0.03687254 -0.13805464 -0.38399001]
  
Model Performance:  {'R²': 0.5543968272471183, 'RMSE': 0.6757270536982963, 'MAE': 0.4061342071169473}

-----------------------------------------------------------------------
XGBoost:

Best params: {'subsample': 0.7, 'reg_lambda': 5, 'reg_alpha': 0.05, 'n_estimators': 600, 'min_child_weight': 8, 'max_depth': 3, 'learning_rate': 0.01, 'colsample_bytree': 0.6}

Best CV RMSE: 0.6825894206926216

Test R²:   0.5993791525289482

Test RMSE: 0.6407135996465826

Test MAE:  0.38540209135000125

Top 20 features:

- budget                         0.2057
- Adventure                      0.1001
- director_past_avg_rev          0.0710
- Fantasy                        0.0593
- Animation                      0.0562
- runtime                        0.0555
- Drama                          0.0521
- Science                        0.0497
- Family                         0.0402
- Action                         0.0377
- INTEREST_RATE                  0.0321
- Year                           0.0303
- INFLATION                      0.0270
- GDP                            0.0262
- Fiction                        0.0261
- Thriller                       0.0256
- Month                          0.0217
- History                        0.0183
- Comedy                         0.0173
- Mystery                        0.0143
