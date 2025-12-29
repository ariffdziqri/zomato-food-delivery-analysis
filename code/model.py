import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


class BuildModel:
    def __init__(self, df, features, numeric_f, cat_f,
                 target_col='Time_taken (min)', random_state=0, skip_split=False):

        self.df = df
        self.features = features
        self.numeric_f = numeric_f
        self.cat_f = cat_f
        self.target_col = target_col
        self.random_state = random_state

        self.clf = None

        if not skip_split:
            self.X = self.df[self.features]
            self.y = self.df[self.target_col]

            self.X_train, self.X_temp, self.y_train, self.y_temp = train_test_split(
                self.X, self.y, random_state=self.random_state, test_size=0.1
            )
            self.X_valid, self.X_test, self.y_valid, self.y_test = train_test_split(
                self.X_temp, self.y_temp, random_state=self.random_state, test_size=0.5
            )

        self.estimators = [
            ('GradientBoosting', GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)),
            ('DecisionTree', DecisionTreeRegressor(max_depth=10, random_state=self.random_state)),
            ('Knn', KNeighborsRegressor(metric='euclidean', n_neighbors=6))
        ]

        self.clf = Pipeline(steps=[
            ('preprocess', self.transform()),
            ('model', StackingRegressor(
                estimators=self.estimators,
                final_estimator=RandomForestRegressor(
                    max_depth=10, n_estimators=100, random_state=self.random_state, n_jobs=-1
                ),
                n_jobs=-1
            ))
        ])

        # NOTE: removed auto-fit from __init__ for reproducibility/reuse
        # Call self.fit() explicitly

    def split(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def transform(self):
        preprocess = ColumnTransformer(transformers=[
            ('num', StandardScaler(), self.numeric_f),
            ('cat', OneHotEncoder(handle_unknown="ignore"), self.cat_f)
        ])
        return preprocess

    def fit(self):
        self.clf.fit(self.X_train, self.y_train)
        return self

    def predict(self, X=None):
        if X is None:
            X = self.X_test
        return self.clf.predict(X)

    def score(self, X=None, y=None):
        if X is None and y is None:
            X, y = self.X_test, self.y_test
        return self.clf.score(X, y)

    def save(self, path="stacking_model.joblib"):
        joblib.dump({
            "model": self.clf,
            "features": self.features,
            "numeric_f": self.numeric_f,
            "cat_f": self.cat_f,
            "target_col": self.target_col,
            "random_state": self.random_state
        }, path)

    @staticmethod
    def load(path="stacking_model.joblib"):
        obj = joblib.load(path)
    
        bm = BuildModel(
            df=pd.DataFrame(),             
            features=obj["features"],
            numeric_f=obj["numeric_f"],
            cat_f=obj["cat_f"],
            target_col=obj["target_col"],
            random_state=obj["random_state"],
            skip_split=True                 
        )
    
        bm.clf = obj["model"]
        return bm
