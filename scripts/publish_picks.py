import pandas as pd
import numpy as np
import gspread
from datetime import datetime
from scipy import stats
import os

class PublishPicks:
    def __init__(self):
        self.today_date = datetime.now()
        self.today_date = self.today_date.strftime("%-m-%-d-%Y")
        self.fp = self.today_date.split('-')
        self.fp = '-'.join([x.zfill(2) for x in self.fp])
        year = self.today_date.split('-')[2]
        self.fp = f'data/monster_data/{year}/{self.fp}/'
        self.today_picks = pd.read_csv(self.fp + f'{self.today_date}_picks.csv', dtype = {'Home_ML' : str, 'Away_ML':str})


    def tidy_predictions(self):
        today = self.today_picks.copy()
        # sort by time
        today['Home_ML'] = today['Home_ML'].astype(str)
        today['Away_ML'] = today['Away_ML'].astype(str)

        today['Game'] = today['Home'] + ' (' + today['Home_ML'] + ') vs ' + today['Away'] + ' (' + today['Away_ML'] + ')'
        today['Time'] = 'Game ' + today['Game_Number'].astype(str) + ' - ' + today['Time_EST']
        today['Official Pick'] = np.where(today['Bet_Home'] == 1, today['Home'] + ' ' + today['Home_ML'],
                                          np.where(today['Bet_Away'] == 1, today['Away'] + ' ' + today['Away_ML'], 'No bet'))
        today['Pr(Win)'] = np.where(today['Bet_Home'] == 1, today['Home_Win_Prob'],
                                    np.where(today['Bet_Away'] == 1, today['Away_Win_Prob'], np.nan))
        today['EV'] = np.where(today['Bet_Home'] == 1, today['Home_EV'],
                               np.where(today['Bet_Away'] == 1, today['Away_EV'], np.nan))
        today['Kelly Pct.'] = np.where(today['Bet_Home'] == 1, today['Home_Kelly'],
                                        np.where(today['Bet_Away'] == 1, today['Away_Kelly'], np.nan))
        today['Kelly Pct.'] = np.where(today['Kelly Pct.'] > 0.25, 0.25, today['Kelly Pct.'])
        today['Kelly Units'] = today['Kelly Pct.'] * 100 / 5
        today['Kelly Units'] = today['Kelly Units'].fillna(0)
        today['Kelly Units'] = today['Kelly Units'].apply(lambda x: round(x * 2) / 2)

        # remove trailing zeros
        today['Kelly Units'] = today['Kelly Units'].apply(lambda x: str(x).rstrip('0').rstrip('.') if '.' in str(x) else x)
        today['Kelly Units'] = np.where(today['Official Pick'] == 'No bet', '', today['Kelly Units'] + 'u')

        today_tidy = today.reset_index()[['Game', 'Time', 'Confirmed', 'Official Pick', 'Pr(Win)', 'EV', 'As Of',
                                          'Kelly Pct.', 'Kelly Units']]
        today_tidy = today_tidy.rename(columns = {'Confirmed' : 'Lineups Confirmed', 'Kelly Units' : 'Units'})
        today_tidy['Pr(Win)'] = today_tidy['Pr(Win)'].apply(lambda x: format(x, ".2%"))
        today_tidy['EV'] = today_tidy['EV'].apply(lambda x: format(x, ".2%"))
        today_tidy['Kelly Pct.'] = today_tidy['Kelly Pct.'].apply(lambda x: format(x, ".2%"))
        today_tidy['Pr(Win)'] = np.where(today_tidy['Pr(Win)'] == 'nan%', '', today_tidy['Pr(Win)'])
        today_tidy['EV'] = np.where(today_tidy['EV'] == 'nan%', '', today_tidy['EV'])
        today_tidy['Kelly Pct.'] = np.where(today_tidy['Kelly Pct.'] == 'nan%', '', today_tidy['Kelly Pct.'])

        return today_tidy

    def publish_picks_gsheets(self):
        today_tidy = self.tidy_predictions()
        #print("Today's picks:")
        #print(today_tidy[today_tidy['Official Pick'] != 'No bet'].drop(columns = ['As Of']))
        as_of = today_tidy['As Of'][0]
        today_tidy = today_tidy.drop(columns = 'As Of')
        cols = ['MLB Picks', '', '', '', '', '', '', '']
        second_row = pd.DataFrame([f'As of: {as_of}', '', '', '', '', '', '','']).T
        second_row.columns = cols
        third_row = pd.DataFrame(cols).T
        third_row.columns = cols
        third_row.iloc[0] = today_tidy.columns

        today_tidy.columns = cols
        today_tidy = pd.concat([second_row, third_row, today_tidy])
        blank_df = pd.DataFrame(index=range(15), columns = cols)
        blank_df = blank_df.fillna('')
        today_tidy = pd.concat([today_tidy, blank_df])
        gc = gspread.service_account(filename='misc/mlb-modeling-a9139a680fef.json')
        gc = gc.open('MLB Model Picks')
        sh = gc.worksheet("Today's Picks")
        sh.update([today_tidy.columns.values.tolist()] + today_tidy.values.tolist())


    def publish_results_gsheets(self):
        results = pd.read_csv('data/picks_results.csv')
        results = results.drop(columns = ['Unnamed: 0'])
        results['Date'] = results['Game_ID'].str.split('-').str[2]
        results['Date'] = results['Date'].str[:4] + '-' + results['Date'].str[4:6] + '-' + results['Date'].str[6:]
        results['Game_ID'] = results['Date']
        results = results.drop(columns = ['Date']).rename(columns = {'Game_ID' : 'Date'})
        roi = results['Return'].sum() / results['Return'].count()
        units = results['Return'].sum() - results['Return'].count()
        current_record = str(results['Win_Bet'].sum()) + '-' + str(results['Win_Bet'].count() - results['Win_Bet'].sum())
        n_bets = results['Win_Bet'].count()
        se = np.sqrt((1 - roi) ** 2 / n_bets)
        tstat = roi / se
        pval = 1 - stats.t.cdf(tstat, n_bets - 1)

        roi = "{:.2%}".format(roi)
        tstat = "{:.4f}".format(tstat)
        pval = "{:.4f}".format(pval)
        if units > 0:
            units = "+" + "{:.2f}".format(units)
        else:
            units = "{:.2f}".format(units)

        cols = ['Season results'] + ([''] * 16)

        second_row = pd.DataFrame([f'Record: {current_record}', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '' ,'']).T
        third_row = pd.DataFrame([f'ROI: {roi}', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '' ,'']).T
        fourth_row = pd.DataFrame([f'{units}u', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '' ,'']).T
        fifth_row = pd.DataFrame([f'T-stat: {tstat} (p = {pval})', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '' ,'']).T
        second_row.columns = cols
        third_row.columns = cols
        fourth_row.columns = cols
        fifth_row.columns = cols
        header = pd.concat([second_row, third_row, fourth_row, fifth_row])
        blank_row = pd.DataFrame({col: np.nan for col in results.columns}, index=[0])
        results = pd.concat([blank_row, results], axis = 0)
        results.iloc[0] = results.columns
        results.columns = cols
        results = pd.concat([header, results], axis = 0)

        gc = gspread.service_account(filename='misc/mlb-modeling-a9139a680fef.json')
        gc = gc.open('MLB Model Picks')
        sh = gc.worksheet("Results")
        sh.update([results.columns.values.tolist()] + results.values.tolist())

if __name__ == '__main__':
    publisher = PublishPicks()
    publisher.publish_picks_gsheets()
    #publisher.publish_results_gsheets()
#%%

#%%

#%%
