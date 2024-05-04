import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, precision_score, recall_score, f1_score
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
    'min_odds': [1.10, 1.25, 1.333, 1.5],
    'max_odds': [3, 5, 11],
    'min_ev': np.arange(0, 0.16, .025),
    'kelly_mult' : [.25, .5, .75, 1],
    'max_kelly_bet' : [.05, 0.075, .10, .25]}

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
        'scaler': [StandardScaler(), MinMaxScaler()],
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
        'scaler': [StandardScaler(), MinMaxScaler()],
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
        'scaler': [StandardScaler(), MinMaxScaler()],
        'pca__n_components': [round(num, 2) for num in np.arange(0.1, 1, 0.05)] + [None],
        'classifier__C': [0.01, 0.1, 1, 10, 100, 1000],
        'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'classifier__degree': [2, 3, 4, 5],
        'classifier__coef0': [0.0, 0.5, 1.0],
    },
    xgb.XGBClassifier: {
        'scaler': [StandardScaler(), MinMaxScaler()],
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

fm_orig = pd.read_feather('data/feature_matrix.feather')
curr_yr_fm = pd.read_csv('data/curr_year_feature_matrix.csv').set_index('Game_ID')
# keep only columns that are in both feature matrices
curr_yr_fm = curr_yr_fm[[col for col in curr_yr_fm.columns if col in fm_orig.columns]]
fm_orig = pd.concat([fm_orig, curr_yr_fm])

def calculate_ev(prob_win, decimal_odds):
    return (prob_win * (decimal_odds - 1) - (1 - prob_win))


def calculate_payout(fm):
    fm['payout'] = np.where(fm['bet_home'] == 1, fm['win_bet'] * fm['Home_ML'], np.nan)
    fm['payout'] = np.where(fm['bet_away'] == 1, fm['win_bet'] * fm['Away_ML'], fm['payout'])
    return fm

def calculate_roi(fm):
    return (fm['payout'].sum() - fm['bet'].sum()) / fm['bet'].sum()

def calculate_kelly_pct(fm, kelly_multiplier, max_kelly_bet_pct):
    fm['kelly_pct_home'] = (((fm['Home_ML'] - 1) * fm['prob']) - (fm['inv_prob'])) / (fm['Home_ML'] - 1)
    fm['kelly_pct_away'] = (((fm['Away_ML'] - 1) * fm['inv_prob']) - (fm["prob"])) / (fm['Away_ML'] - 1)
    fm['kelly_pct'] = np.where(fm['bet_home'] == 1, fm['kelly_pct_home'], np.nan)
    fm['kelly_pct'] = np.where(fm['bet_away'] == 1, fm['kelly_pct_away'], fm['kelly_pct'])
    fm['kelly_pct'] = fm['kelly_pct'] * kelly_multiplier
    fm['kelly_pct'] = np.where(fm['kelly_pct'] > max_kelly_bet_pct, max_kelly_bet_pct, fm['kelly_pct'])
    return fm

def calculate_payout_kelly(fm):
    fm['payout_kelly'] = np.where(fm['bet_home'] == 1, fm['kelly_pct'] * fm['Home_ML'] * fm['win_bet'], np.nan)
    fm['payout_kelly'] = np.where(fm['bet_away'] == 1, fm['kelly_pct'] * fm['Away_ML'] * fm['win_bet'],
                                  fm['payout_kelly'])
    return fm

def calculate_roi_kelly(fm):
    return (fm['payout_kelly'].sum() - fm['kelly_pct'].sum()) / fm['kelly_pct'].sum()

def calculate_roi_tstat(fm, kelly):
    if kelly:
        observed_roi = calculate_roi_kelly(fm)
        bet_amounts = fm['kelly_pct']
    else:
        observed_roi = calculate_roi(fm)
        bet_amounts = fm['bet']

    n_bets = len(bet_amounts)
    total_bet_amount = bet_amounts.sum()

    roi_diff_sq = (fm['payout_kelly'] / bet_amounts - observed_roi) ** 2 if kelly else (fm['payout'] / fm['bet'] - observed_roi) ** 2
    weighted_se = np.sqrt(np.sum(bet_amounts ** 2 * roi_diff_sq) / (total_bet_amount ** 2 * (n_bets - 1)))

    tstat = observed_roi / weighted_se
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
                                   (fm['Away_ML'] < roi_params[1]) &
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
    #fm['prob'] = np.where(np.isinf(fm['prob']), 0, fm['prob'])
    prob_brier = brier_score_loss(fm['Home_Win'], fm['prob'])
    prob_logloss = log_loss(fm['Home_Win'], fm['prob'])
    # get roc_auc score
    roc_auc = roc_auc_score(fm['Home_Win'], fm['prob'])
    precision = precision_score(fm['Home_Win'], fm['prob'] > 0.5)
    recall = recall_score(fm['Home_Win'], fm['prob'] > 0.5)
    f1 = f1_score(fm['Home_Win'], fm['prob'] > 0.5)

    # add to dictionary
    dic = {'prob_accuracy': prob_accuracy,
           'prob_brier': prob_brier,
           'prob_logloss': prob_logloss,
           'roc_auc': roc_auc,
           'precision': precision,
           'recall': recall,
           'f1': f1}

    return dic


def get_roi_metrics(prob_dataframe, params_copy):
    results_list = []  # Collect dictionaries here to avoid frequent DataFrame operations

    for roi_params in itertools.product(*roi_grid.values()):
        fm = clean_fm(prob_dataframe, roi_params)
        fm = calculate_kelly_pct(fm, roi_params[3], roi_params[4])
        bet_amt = fm['bet'].sum()
        kelly_bet_amt = fm['kelly_pct'].sum()  # Assuming 'kelly_pct' serves the purpose of 'bet_amt_kelly
        fm = calculate_payout_kelly(fm)
        roi = calculate_roi(fm)
        roi_kelly = calculate_roi_kelly(fm)
        tstat_kelly = calculate_roi_tstat(fm, kelly=True)
        winnings_kelly = kelly_bet_amt * roi_kelly
        winnings = bet_amt * roi
        tstat = calculate_roi_tstat(fm, kelly=False)
        results_list.append({'bet_amt': bet_amt, 'winnings': winnings, 'kelly_bet_amt': kelly_bet_amt,
                             'kelly_winnings': winnings_kelly, 'roi': roi, 'tstat': tstat, 'roi_kelly': roi_kelly,
                             'tstat_kelly': tstat_kelly, 'roi_params': str(roi_params),
                             'pct_bets': fm['bet'].mean(), **dict(zip(roi_grid.keys(), roi_params))})

    # Convert the list of dictionaries to a DataFrame in one go
    roi_fold_scores = pd.DataFrame(results_list)
    roi_fold_scores['params'] = params_copy  # Add this column after creating the DataFrame
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
        #roi_fold_score = get_roi_metrics(test_proba, params_copy)
        fold_scores = pd.concat([fold_scores, pd.DataFrame(fold_score, index=[0])])
        #roi_fold_scores = pd.concat([roi_fold_scores, roi_fold_score])

    fold_scores['params'] = params_copy
    #roi_fold_score = roi_fold_scores.groupby(['params', 'roi_params']).mean().reset_index().set_index('params')
    fold_score = fold_scores.groupby('params').mean()
    #mast_fold_score = fold_score.merge(roi_fold_score, on='params', how='left')
    mast_fold_score = fold_score.copy()
    return mast_fold_score


def custom_random_search(pipeline, param_grid, X, y, n_iter_cv, n_iter_search=1000):
    def random_combinations(iterables, r):
        # Calculate the total number of combinations possible
        pools = list(map(tuple, iterables))
        n = 1
        for pool in pools:
            n *= len(pool)
        # Generate random indices for selecting combinations
        indices = sorted(random.sample(range(n), r))

        def combination_from_index(index):
            result = []
            for pool in pools[::-1]:
                index, i = divmod(index, len(pool))
                result.append(pool[i])
            result.reverse()
            return tuple(result)

        # Yield combinations using the indices
        for index in indices:
            yield combination_from_index(index)
    param_grid_values = list(param_grid.values())
    random_params = list(random_combinations(param_grid_values, n_iter_search))
    kf = KFold(n_splits=n_iter_cv, shuffle=True, random_state=44)
    print(
        f'Fitting {n_iter_cv} folds for each of {len(random_params)} candidates, totalling {n_iter_cv * len(random_params)} fits.')
    scores = Parallel(n_jobs=-1)(
        delayed(evaluate_params)(params, pipeline, param_grid, X, y, kf) for params in random_params)
    scores = pd.concat(scores)
    return scores


def fit_kfold(model, n_splits, n_search):
    fm = pd.read_feather('data/feature_matrix.feather')
    curr_yr_fm = pd.read_csv('data/curr_year_feature_matrix.csv').set_index('Game_ID')
    # keep only columns that are in both feature matrices
    curr_yr_fm = curr_yr_fm[[col for col in curr_yr_fm.columns if col in fm.columns]]
    curr_yr_fm['season'] = '2024'
    fm['season'] = '2023'

    fm = pd.concat([fm, curr_yr_fm])
    # convert to dummies
    fm = pd.get_dummies(fm, columns=['season'])

    fm = fm.dropna(subset = 'Home_Win')

    X = fm.drop(columns=['Home_Runs', 'Away_Runs', 'True_Total', 'Home_Win', 'Home_Spread_True',
                         'Home_Cover', 'Away_Cover', 'Away_ML_Hit', 'Over_Hit', 'Under_Hit', 'Home_Spread',
                         'Away_Spread']).fillna(0)
    y = fm['Home_Win']
    # Create a pipeline that includes scaling, PCA, and the classifier
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('classifier', model)])

    param_grid = param_grids[type(model)]
    classifier = str(pipeline['classifier']).split('(')[0] + '()'
    print(f"Searching {classifier}")
    scores = custom_random_search(pipeline, param_grid, X, y, n_iter_cv=n_splits, n_iter_search=n_search)
    print('Search completed')
    scores['model'] = str(model).split('(')[0] + '()'
    #scores['kelly'] = np.where(scores['roi_kelly'] > scores['roi'], True, False)
    scores = scores.reset_index()
    #scores = scores.set_index(['model', 'params', 'roi_params']).reset_index()

    #scores['max_roi'] = scores[['roi', 'roi_kelly']].max(axis=1)
    #scores = scores.sort_values(by='max_roi', ascending=False)
    #scores = scores.drop(columns='max_roi')
    scores.to_feather(f'results/{classifier}_scores.feather')
    return scores


def model_evaluation(model):
    scores = fit_kfold(model, n_splits=2, n_search=5)
    return scores


def main():
    xgb_model_results = model_evaluation(xgb.XGBClassifier(nthread=-1))
    #svc_results = model_evaluation(SVC(probability=True))
    #mlp_results = model_evaluation(MLPClassifier())
    #rf_model_results = model_evaluation(RandomForestClassifier(n_jobs=-1, random_state=44))
    #logistic_model_results = model_evaluation(LogisticRegression(n_jobs=-1))

if __name__ == '__main__':
    main()
#%%
