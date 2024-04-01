import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from create_fm import CreateFeatureMatrix

class PredictGames:
    def __init__(self):
        self.today_date = datetime.now().strftime("%-m-%-d-%Y")
        fp = self.today_date.split('-')
        fp = '-'.join([x.zfill(2) for x in fp])
        year = self.today_date.split('-')[2]
        self.fp = f'data/monster_data/{year}/{fp}/'

    def load_data(self):
        self.trained_fm = pd.read_csv('data/feature_matrix.csv', low_memory=False).set_index('Game_ID')
        self.today_fm = pd.read_csv(self.fp + f'{self.today_date}_fm.csv', low_memory=False).set_index('Game_ID')

    def load_model(self):
        self.pipeline = joblib.load('models/prod_model.pkl')

    def predict(self):
        trained_fm_x = self.trained_fm.drop(columns = ['Home_Runs', 'Away_Runs', 'True_Total', 'Home_Win',  'Home_Spread_True','Home_Cover', 'Away_Cover', 'Away_ML_Hit', 'Over_Hit', 'Under_Hit', 'Home_Spread', 'Away_Spread'])
        trained_fm_x['training_set'] = 1
        today_fm_x = pd.concat([trained_fm_x, self.today_fm])
        today_fm_x = today_fm_x[today_fm_x['training_set'] != 1].fillna(0)
        today_fm_x = today_fm_x[trained_fm_x.columns].drop(columns = 'training_set')

        today_fm_x['Home_Win_Prob'] = self.pipeline.predict_proba(today_fm_x)[:, 1]
        self.results = today_fm_x[['Home_Win_Prob']]

    @staticmethod
    def calculate_ev(prob_win, decimal_odds):
        return (prob_win * (decimal_odds - 1) - (1 - prob_win))

    @staticmethod
    def decimal_to_american(decimal_odds):
        if decimal_odds > 2:
            return (decimal_odds - 1) * 100
        else:
            return -100 / (decimal_odds - 1)

    def prepare_data(self):
        roi_params = (1.125, 10, -5)
        self.today = pd.concat([self.today_fm[['Away', 'Home', 'Time_EST', 'Game_Number', 'Home_ML', 'Away_ML']], self.results], axis=1)
        self.today['Home_EV'] = self.calculate_ev(self.today['Home_Win_Prob'], self.today['Home_ML'])
        self.today['Away_EV'] = self.calculate_ev(1 - self.today['Home_Win_Prob'], self.today['Away_ML'])
        self.today['Bet_Home'] = np.where((self.today['Home_EV'] > self.today['Away_EV']) &
                                          (self.today['Home_EV'] > roi_params[2]) &
                                          (self.today['Home_ML'] > roi_params[0]) &
                                          (self.today['Home_ML'] < roi_params[1]), 1, 0)

        self.today['Bet_Away'] = np.where((self.today['Away_EV'] > self.today['Home_EV']) &
                                          (self.today['Away_EV'] > roi_params[2]) &
                                          (self.today['Away_ML'] > roi_params[0]) &
                                          (self.today['Away_ML'] < roi_params[1]), 1, 0)

        self.today['Home_ML'] = self.today['Home_ML'].apply(self.decimal_to_american)
        self.today['Home_ML'] = np.where(self.today['Home_ML'] > 0, '+' + self.today['Home_ML'].astype(str).str[:3], self.today['Home_ML'].astype(str).str[:4])
        self.today['Away_ML'] = self.today['Away_ML'].apply(self.decimal_to_american)
        self.today['Away_ML'] = np.where(self.today['Away_ML'] > 0, '+' + self.today['Away_ML'].astype(str).str[:3], self.today['Away_ML'].astype(str).str[:4])
        self.today['Away_Win_Prob'] = 1 - self.today['Home_Win_Prob']
        uncnf = self.get_other_games()
        self.today = pd.concat([self.today.reset_index(), uncnf])
        self.today = self.today.drop_duplicates(subset = ['Game_ID'], keep = 'first')
        self.today['As Of'] = datetime.now().strftime("%-m/%-d/%Y %H:%M:%S")
        self.today = self.today.set_index('Game_ID')
        self.confirm_lineups()

    def confirm_lineups(self):
        lineups = pd.read_csv(self.fp + f'{self.today_date}_l.csv')
        lineups = lineups[['team code', ' game_number', ' confirmed']]
        lineups.columns = ['Team', 'Game_Number', 'Confirmed']
        lineups = lineups.drop_duplicates()
        home = lineups.copy()
        away = lineups.copy()
        self.today = self.today.reset_index().merge(home, left_on = ['Home', 'Game_Number'], right_on = ['Team', 'Game_Number'], how = 'left')
        self.today = self.today.drop(columns = ['Team']).rename(columns = {'Confirmed': 'Home_Confirmed'})
        self.today = self.today.merge(away, left_on = ['Away', 'Game_Number'], right_on = ['Team', 'Game_Number'], how = 'left')
        self.today = self.today.drop(columns = ['Team']).rename(columns = {'Confirmed': 'Away_Confirmed'})
        self.today['Confirmed'] = np.where((self.today['Home_Confirmed'] == 'Y') & (self.today['Away_Confirmed'] == 'Y'), 'Y', 'N')
        self.today = self.today.drop(columns = ['Home_Confirmed', 'Away_Confirmed']).set_index('Game_ID')

    def get_other_games(self):
        today_h = pd.read_excel(self.fp + f'{self.today_date}_h.xls')
        today_h['Game_ID'] = CreateFeatureMatrix().get_game_key(today_h)
        today_h[['drop', 'Spread', 'drop', 'Total', 'ML']] = today_h['Odds'].str.split(' ', expand=True)
        today_h = today_h[['Game_ID', 'g', 'Time', 'Team', 'Home', 'Away', 'Spread', 'Total', 'ML']].drop_duplicates()
        today_home = today_h[today_h['Home'] == today_h['Team']]
        today_away = today_h[today_h['Away'] == today_h['Team']]
        today_home = today_home.rename(columns = {'Spread' : 'Home_Spread', 'Total' : 'Home_Total', 'ML' : 'Home_ML'})
        today_away = today_away.rename(columns = {'Spread' : 'Away_Spread', 'Total' : 'Away_Total', 'ML' : 'Away_ML'})
        today = pd.merge(today_home, today_away, on = ['Game_ID', 'g', 'Home', 'Away', 'Time'])
        today = today[['Game_ID', 'Home', 'Away', 'g', 'Time', 'Home_ML', 'Away_ML']]
        today = today.rename(columns = {'g' : 'Game_Number', 'Time' : 'Time_EST'})
        today['Home_ML'] = today['Home_ML'].str.replace('(', '')
        today['Away_ML'] = today['Away_ML'].str.replace('(', '')
        today['Home_ML'] = today['Home_ML'].str.replace(')', '')
        today['Away_ML'] = today['Away_ML'].str.replace(')', '')
        return today

    def save_predictions(self):
        # convert time est to datetime
        #print(self.today)

        self.today['Time_EST_dt'] = pd.to_datetime(self.today['Time_EST'], format = 'mixed')
        self.today = self.today.sort_values(by = ['Time_EST_dt'])
        self.today = self.today.drop(columns = ['Time_EST_dt'])
        self.today.to_csv(self.fp + f'{self.today_date}_picks.csv')

    def predict_games(self):
        self.load_data()
        self.load_model()
        self.predict()
        self.prepare_data()
        self.save_predictions()

if __name__ == '__main__':
    predictor = PredictGames()
    predictor.predict_games()

#%%
