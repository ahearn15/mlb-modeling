import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import ast


class TrainProdModel:
    def __init__(self):
        pass

    def read_fm(self, fp='data/feature_matrix.csv'):
        fm = pd.read_csv(fp, low_memory=False).set_index('Game_ID')
        X = fm.drop(columns = ['Home_Runs', 'Away_Runs', 'True_Total', 'Home_Win',  'Home_Spread_True',
                           'Home_Cover', 'Away_Cover', 'Away_ML_Hit', 'Over_Hit', 'Under_Hit', 'Home_Spread', 'Away_Spread'])
        y = fm['Home_Win']

        return X, y

    def create_pipeline(self):
        # Create a pipeline that includes scaling, PCA, and the classifier
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('classifier', XGBClassifier())])

        params = {'scaler': StandardScaler(),
                  'pca__n_components': 0.85,
                  'classifier__n_estimators': 500,
                  'classifier__learning_rate': 0.01,
                  'classifier__max_depth': 4,
                  'classifier__colsample_bytree': 0.3,
                  'classifier__min_child_weight': 1,
                  'classifier__gamma': 0.5,
                  'classifier__subsample': 1.0,
                  'classifier__reg_alpha': 0.1,
                  'classifier__reg_lambda': 2}

        self.pipeline.set_params(**params)

    def read_chosen_model(self):
        with open('models/chosen_model.txt', 'r') as file:
            chosen_mod = file.read()

        chosen_mod = chosen_mod.split('\n')
        print(chosen_mod)

    def fit_pipeline(self, X, y):
        self.pipeline.fit(X, y)
        joblib.dump(self.pipeline, 'models/prod_model.pkl')

    def train_mod(self):
        X, y = self.read_fm('data/feature_matrix.csv')
        self.read_chosen_model()
        self.create_pipeline()
        self.fit_pipeline(X, y)
        print('Model trained successfully')


if __name__ == '__main__':
    trainer = TrainProdModel()
    trainer.train_mod()


#%%
