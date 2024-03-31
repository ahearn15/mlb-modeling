
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import itertools
import random
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

roi_grid = {
    'min_odds': [1.10, 1.125, 1.25, 1.333, 1.5],
    'max_odds': [3, 4, 5, 9, 11],
    'min_ev': np.arange(0, 0.1, .01)
}

# Define the parameter grids
param_grids = {
    LogisticRegression: {
        'scaler': [StandardScaler(), MinMaxScaler()],
        'pca__n_components': [round(num, 2) for num in np.arange(0.1, 1, 0.05)] + [None],
        'classifier__C': np.logspace(-4, 4, 20),
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga'],
        'classifier__max_iter': [500, 1000, 5000, 10000],
        'classifier__fit_intercept': [True, False],
        'classifier__tol': [1e-4, 1e-3, 1e-2],
    },
    RandomForestClassifier: {
        'pca__n_components': [round(num, 2) for num in np.arange(0.1, 1, 0.05)] + [None],
        'classifier__n_estimators': [100, 200, 300, 400, 500],
        'classifier__max_depth': [None, 5, 10, 15, 20, 25],
        'classifier__min_samples_split': [2, 5, 10, 15, 20],
        'classifier__min_samples_leaf': [1, 2, 4, 6, 8],
        'classifier__max_features': [.5, .75, 'sqrt', 'log2', None],
        'classifier__bootstrap': [True, False],
        'classifier__criterion': ['gini', 'entropy'],
    },
    MLPClassifier: {
        'pca__n_components': [round(num, 2) for num in np.arange(0.1, 1, 0.05)] + [None],
        'classifier__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,), (100, 100), (30, 50, 30),
                                           (50, 50, 50, 50)],
        'classifier__activation': ['tanh', 'relu', 'logistic', 'identity'],
        'classifier__solver': ['sgd', 'adam', 'lbfgs'],
        'classifier__alpha': [0.0001, 0.001, 0.01, 0.05],
        'classifier__learning_rate': ['constant', 'adaptive', 'invscaling'],
        'classifier__max_iter': [300, 500, 750, 1000, 1500],
        'classifier__learning_rate_init': [0.001, 0.01, 0.1],
        'classifier__beta_1': [0.9, 0.95, 0.99],
        'classifier__beta_2': [0.999, 0.9999, 0.99999],
        'classifier__epsilon': [1e-8, 1e-9, 1e-10],
    },
    SVC: {
        'pca__n_components': [round(num, 2) for num in np.arange(0.1, 1, 0.05)] + [None],
        'classifier__C': [0.01, 0.1, 1, 10, 100, 1000],
        'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'classifier__degree': [2, 3, 4, 5],
        'classifier__coef0': [0.0, 0.5, 1.0],
    },
    xgb.XGBClassifier: {
        'pca__n_components': [round(num, 2) for num in np.arange(0.1, 1, 0.05)] + [None],
        'classifier__n_estimators': [100, 200, 300, 400, 500],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'classifier__max_depth': [3, 4, 5, 6, 7, 8, 10],
        'classifier__colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        'classifier__min_child_weight': [1, 2, 3, 4, 5],
        'classifier__gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'classifier__reg_alpha': [0, 0.1, 0.5, 1],
        'classifier__reg_lambda': [1, 1.5, 2, 3],
    }
}
fm_orig = pd.read_csv('data/feature_matrix.csv', low_memory=False).set_index('Game_ID')


def calculate_ev(prob_win, decimal_odds):
    return (prob_win * (decimal_odds - 1) - (1 - prob_win))


def calculate_payout(fm):
    fm['payout'] = np.where(fm['bet_home'] == 1, fm['win_bet'] * fm['Home_ML'], np.nan)
    fm['payout'] = np.where(fm['bet_away'] == 1, fm['win_bet'] * fm['Away_ML'], fm['payout'])
    return fm


def calculate_roi(fm):
    return (fm['payout'].sum() - fm['bet'].sum()) / fm['bet'].sum()


def calculate_kelly_pct(fm):
    fm['kelly_pct_home'] = (((fm['Home_ML'] - 1) * fm['prob']) - (fm['inv_prob'])) / (fm['Home_ML'] - 1)
    fm['kelly_pct_away'] = (((fm['Away_ML'] - 1) * fm['inv_prob']) - (fm["prob"])) / (fm['Away_ML'] - 1)
    fm['kelly_pct'] = np.where(fm['bet_home'] == 1, fm['kelly_pct_home'], np.nan)
    fm['kelly_pct'] = np.where(fm['bet_away'] == 1, fm['kelly_pct_away'], fm['kelly_pct'])
    return fm


def calculate_payout_kelly(fm):
    fm['payout_kelly'] = np.where(fm['bet_home'] == 1, fm['kelly_pct_home'] * fm['Home_ML'] * fm['win_bet'], np.nan)
    fm['payout_kelly'] = np.where(fm['bet_away'] == 1, fm['kelly_pct_away'] * fm['Away_ML'] * fm['win_bet'],
                                  fm['payout_kelly'])
    return fm


def calculate_roi_kelly(fm):
    return (fm['payout_kelly'].sum() - fm['kelly_pct'].sum()) / fm['kelly_pct'].sum()


def calculate_roi_tstat(fm, kelly):
    if kelly:
        observed_roi = calculate_roi_kelly(fm)
    else:
        observed_roi = calculate_roi(fm)
    n_bets = fm['bet'].sum()
    se = np.sqrt((1 - observed_roi) ** 2 / n_bets)
    tstat = observed_roi / se
    return tstat


def clean_fm(prob_dataframe, roi_params=None):
    fm = fm_orig.copy()
    fm = pd.concat([prob_dataframe, fm], axis=1)
    prob_var = fm.columns[0]
    fm['prob'] = fm[prob_var]
    fm['inv_prob'] = 1 - fm['prob']
    fm = fm.dropna(subset=['prob'])

    fm['prob_ev'] = calculate_ev(fm['prob'], fm['Home_ML'])
    fm['inv_prob_ev'] = calculate_ev(1 - fm['prob'], fm['Away_ML'])
    if roi_params is None:
        fm['bet_home'] = np.where((fm['prob_ev'] > 0) & (fm['prob_ev'] > fm['inv_prob_ev']), 1, 0)
        fm['bet_away'] = np.where((fm['inv_prob_ev'] > 0) & (fm['inv_prob_ev'] > fm['prob_ev']), 1, 0)
    else:
        fm['bet_home'] = np.where(((fm['prob_ev'] > roi_params[2]) &
                                   (fm['Home_ML'] > roi_params[0]) &
                                   (fm['Home_ML'] < roi_params[1]) &
                                   (fm['prob_ev'] > fm['inv_prob_ev'])), 1, 0)
        fm['bet_away'] = np.where(((fm['inv_prob_ev'] > roi_params[2]) &
                                   (fm['Away_ML'] > roi_params[0]) &
                                   (fm['Home_ML'] < roi_params[1]) &
                                   (fm['inv_prob_ev'] > fm['prob_ev'])), 1, 0)
    fm['bet'] = fm['bet_home'] + fm['bet_away']
    fm['win_bet_home'] = np.where((fm['bet_home'] == 1) & (fm['Home_Win'] == 1), 1, 0)
    fm['win_bet_away'] = np.where((fm['bet_away'] == 1) & (fm['Home_Win'] == 0), 1, 0)

    fm['win_bet'] = fm['win_bet_away'] + fm['win_bet_home']
    fm['win_bet'] = np.where(fm['bet_home'] + fm['bet_away'] == 0, np.nan, fm['win_bet'])
    fm = calculate_payout(fm)
    return fm


def get_metrics(prob_dataframe):
    fm = clean_fm(prob_dataframe)
    prob_accuracy = fm['win_bet'].mean()
    prob_brier = brier_score_loss(fm['Home_Win'], fm['prob'])
    prob_logloss = log_loss(fm['Home_Win'], fm['prob'])
    dic = {'prob_accuracy': prob_accuracy, 'prob_brier': prob_brier, 'prob_logloss': prob_logloss}
    return dic


def get_roi_metrics(prob_dataframe, params_copy):
    roi_fold_scores = pd.DataFrame()
    for roi_params in list(itertools.product(*roi_grid.values())):
        fm = clean_fm(prob_dataframe, roi_params)
        fm = calculate_kelly_pct(fm)
        fm['bet_kelly'] = fm['bet']
        fm = calculate_payout_kelly(fm)
        roi = calculate_roi(fm)
        roi_kelly = calculate_roi_kelly(fm)
        tstat_kelly = calculate_roi_tstat(fm, kelly=True)
        tstat = calculate_roi_tstat(fm, kelly=False)
        dic = {'roi': roi, 'tstat': tstat, 'roi_kelly': roi_kelly,
               'tstat_kelly': tstat_kelly, 'roi_params': str(roi_params),
               'pct_bets': fm['bet'].mean()}
        dic.update(dict(zip(roi_grid.keys(), roi_params)))
        roi_fold_scores = pd.concat([roi_fold_scores, pd.DataFrame(dic, index=[0])])
    roi_fold_scores = roi_fold_scores.drop(columns=['min_odds', 'max_odds', 'min_ev'])
    roi_fold_scores['params'] = params_copy
    return roi_fold_scores


def evaluate_params(params, pipeline, param_grid, X, y, kf):
    pipeline.set_params(**dict(zip(param_grid.keys(), params)))
    fold_scores = pd.DataFrame()
    roi_fold_scores = pd.DataFrame()
    params_copy = str(dict(zip(param_grid.keys(), params)))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        pipeline.fit(X_train, y_train)
        y_pred = pd.DataFrame(pipeline.predict_proba(X_test)[:, 1])
        test_proba = y_pred.set_index(X_test.index)
        fold_score = get_metrics(test_proba)
        roi_fold_score = get_roi_metrics(test_proba, params_copy)
        fold_scores = pd.concat([fold_scores, pd.DataFrame(fold_score, index=[0])])
        roi_fold_scores = pd.concat([roi_fold_scores, roi_fold_score])

    fold_scores['params'] = params_copy
    roi_fold_score = roi_fold_scores.groupby(['params', 'roi_params']).mean().reset_index().set_index('params')
    fold_score = fold_scores.groupby('params').mean()
    mast_fold_score = fold_score.merge(roi_fold_score, on='params', how='left')
    # print(mast_fold_score)
    return mast_fold_score


def custom_random_search(pipeline, param_grid, X, y, n_iter_cv, n_iter_search=1000):
    all_params = list(itertools.product(*param_grid.values()))
    random_params = random.sample(all_params, n_iter_search)
    kf = KFold(n_splits=n_iter_cv, shuffle=True, random_state=44)
    print(
        f'Fitting {n_iter_cv} folds for each of {len(random_params)} candidates, totalling {n_iter_cv * len(random_params)} fits.')
    scores = Parallel(n_jobs=-1)(
        delayed(evaluate_params)(params, pipeline, param_grid, X, y, kf) for params in tqdm(random_params))
    scores = pd.concat(scores)
    return scores


def fit_kfold(model, n_splits):
    fm = pd.read_csv('data/feature_matrix.csv', low_memory=False).set_index('Game_ID')

    X = fm.drop(columns=['Home_Runs', 'Away_Runs', 'True_Total', 'Home_Win', 'Home_Spread_True',
                         'Home_Cover', 'Away_Cover', 'Away_ML_Hit', 'Over_Hit', 'Under_Hit', 'Home_Spread',
                         'Away_Spread'])

    y = fm['Home_Win']

    # Create a pipeline that includes scaling, PCA, and the classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('classifier', model)])

    param_grid = param_grids[type(model)]

    scores = custom_random_search(pipeline, param_grid, X, y, n_iter_cv=n_splits, n_iter_search=1000)

    scores['model'] = str(model).split('(')[0] + '()'
    scores['kelly'] = np.where(scores['roi_kelly'] > scores['roi'], True, False)
    scores = scores.reset_index()
    scores = scores.set_index(['model', 'params', 'roi_params']).reset_index()

    scores['max_roi'] = scores[['roi', 'roi_kelly']].max(axis=1)
    scores = scores.sort_values(by='max_roi', ascending=False)
    scores = scores.drop(columns='max_roi')
    classifier = str(pipeline['classifier']).split('(')[0] + '()'
    scores.to_csv(f'results/{classifier}_scores.csv')
    return scores


def model_evaluation(model):
    scores = fit_kfold(model, 10)
    return scores


def main():
    logistic_model_results = model_evaluation(LogisticRegression(n_jobs=-1))
    rf_model_results = model_evaluation(RandomForestClassifier(n_jobs=-1, random_state=44))
    xgb_model_results = model_evaluation(xgb.XGBClassifier(nthread=-1))
    return logistic_model_results, rf_model_results, xgb_model_results


if __name__ == '__main__':
    main()

#%%
