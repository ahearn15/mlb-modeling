import pandas as pd
import numpy as np
import gspread
from datetime import datetime
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
        today_tidy = today.reset_index()[['Game', 'Time', 'Confirmed', 'Official Pick', 'Pr(Win)', 'EV', 'As Of']]
        today_tidy = today_tidy.rename(columns = {'Confirmed' : 'Lineups Confirmed'})
        today_tidy['Pr(Win)'] = today_tidy['Pr(Win)'].apply(lambda x: format(x, ".2%"))
        today_tidy['EV'] = today_tidy['EV'].apply(lambda x: format(x, ".2%"))
        today_tidy['Pr(Win)'] = np.where(today_tidy['Pr(Win)'] == 'nan%', '', today_tidy['Pr(Win)'])
        today_tidy['EV'] = np.where(today_tidy['EV'] == 'nan%', '', today_tidy['EV'])
        return today_tidy

    def publish_picks_gsheets(self):
        today_tidy = self.tidy_predictions()
        print("Today's picks:")
        print(today_tidy[today_tidy['Official Pick'] != 'No bet'].drop(columns = ['As Of']))
        as_of = today_tidy['As Of'][0]
        today_tidy = today_tidy.drop(columns = 'As Of')
        cols = ['MLB Picks', '', '', '', '', '']
        second_row = pd.DataFrame([f'As of: {as_of}', '', '', '', '', '']).T
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
        sh = gc.worksheet('Games')
        sh.update([today_tidy.columns.values.tolist()] + today_tidy.values.tolist())


if __name__ == '__main__':
    publisher = PublishPicks()
    publisher.publish_picks_gsheets()
#%%

#%%

#%%
