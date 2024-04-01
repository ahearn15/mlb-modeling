import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import os
import re
import requests
from scipy import stats

class RetrieveResults:
    def __init__(self):
        self.team_to_abbreviation = {
            "Arizona D'Backs": 'ARI',
            'Atlanta Braves': 'ATL',
            'Baltimore Orioles': 'BAL',
            'Boston Red Sox': 'BOS',
            'Chicago Cubs': 'CHC',
            'Chicago White Sox': 'CHW',
            'Cincinnati Reds': 'CIN',
            'Cleveland Guardians': 'CLE',
            'Colorado Rockies': 'COL',
            'Detroit Tigers': 'DET',
            'Houston Astros': 'HOU',
            'Kansas City Royals': 'KC',
            'Los Angeles Angels': 'LAA',
            'Los Angeles Dodgers': 'LAD',
            'Miami Marlins': 'MIA',
            'Milwaukee Brewers': 'MIL',
            'Minnesota Twins': 'MIN',
            'New York Yankees': 'NYY',
            'New York Mets': 'NYM',
            'Oakland Athletics': 'OAK',
            'Philadelphia Phillies': 'PHI',
            'Pittsburgh Pirates': 'PIT',
            'San Diego Padres': 'SD',
            'San Francisco Giants': 'SF',
            'Seattle Mariners': 'SEA',
            'St. Louis Cardinals': 'STL',
            'Tampa Bay Rays': 'TB',
            'Texas Rangers': 'TEX',
            'Toronto Blue Jays': 'TOR',
            'Washington Nationals': 'WAS'
        }

    def scrape_data(self):
        url = 'https://www.baseball-reference.com/leagues/majors/2024-schedule.shtml'
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'html.parser')

        games = soup.find_all('p', class_='game')
        # get all games
        mast_games_df = pd.DataFrame()
        for game in games:
            hrefs = game.find_all('a', href=True)
            if 'Boxscore' in str(hrefs):
                ref = str(hrefs[2]['href']).replace('.shtml', '')
                date = ref[-9:-1]

            home = game.find_all('a')[0].text
            away = game.find_all('a')[1].text

            string = game.text
            numbers_in_parentheses = re.findall(r'\((\d+)\)', string)
            extracted_numbers = [int(number) for number in numbers_in_parentheses]
            if extracted_numbers == []:
                continue
            else:
                away_runs = extracted_numbers[0]
                home_runs = extracted_numbers[1]

                # put in dataframe
                game = pd.DataFrame({
                    'Date': date,
                    'Home': home,
                    'Away': away,
                    'Home_Runs': home_runs,
                    'Away_Runs': away_runs
                }, index = [0])
                mast_games_df = pd.concat([mast_games_df, game], axis = 0)
        return mast_games_df


    def get_results(self):
        mast_games_df = self.scrape_data()
        # create game number flag
        mast_games_df['Game_Number'] = mast_games_df.groupby(['Date', 'Home', 'Away']).cumcount() + 1

        # map to abbreviations
        mast_games_df['Home'] = mast_games_df['Home'].map(self.team_to_abbreviation)
        mast_games_df['Away'] = mast_games_df['Away'].map(self.team_to_abbreviation)
        mast_games_df['Date'] = pd.to_datetime(mast_games_df['Date'])
        mast_games_df['Date'] = mast_games_df['Date'].dt.strftime('%Y-%m-%d')

        mast_games_df['Game_ID'] = mast_games_df['Home'] + '-' + mast_games_df['Away'] + '-' + mast_games_df['Date'].astype(str).replace('-', '', regex = True) + '-' + mast_games_df['Game_Number'].astype(str)
        mast_games_df = mast_games_df.set_index('Game_ID')
        mast_games_df.to_csv('data/game_results.csv')


    def merge_game_results(self):
        # get subdirectories
        subdirs = [x[0] for x in os.walk('data/')][3:]
        mast_picks = pd.DataFrame()
        for dir in subdirs:
            # check if file with '_r.csv' exists
            if not os.path.exists(dir + '/_r.csv'):
                # get all files in directory
                files = os.listdir(dir)
                # get all files that end with '_picks.csv'
                for file in files:
                    if '_picks.csv' in file:
                        daily_picks = pd.read_csv(dir + '/' + file)
                        mast_picks = pd.concat([mast_picks, daily_picks], axis = 0)

        # only get games that were bet on
        mast_picks['Bet'] = mast_picks['Bet_Home'] + mast_picks['Bet_Away']
        mast_picks = mast_picks[mast_picks['Bet'] == 1]
        mast_picks = mast_picks.drop(columns = ['As Of', 'Confirmed'])

        results = pd.read_csv('data/game_results.csv')
        mast_picks = mast_picks.merge(results[['Game_ID', 'Home_Runs', 'Away_Runs']], on = ['Game_ID'], how = 'inner')
        mast_picks['Win_Bet'] = np.where(
            ((mast_picks['Bet_Home'] == 1) & (mast_picks['Home_Runs'] > mast_picks['Away_Runs'])) |
            ((mast_picks['Bet_Away'] == 1) & (mast_picks['Home_Runs'] < mast_picks['Away_Runs'])), 1, 0)

        # convert ml to decimal
        mast_picks['Home_ML'] = np.where(mast_picks['Home_ML'] > 0, (mast_picks['Home_ML'] / 100) + 1, (100 / abs(mast_picks['Home_ML'])) + 1)
        mast_picks['Away_ML'] = np.where(mast_picks['Away_ML'] > 0, (mast_picks['Away_ML'] / 100) + 1, (100 / abs(mast_picks['Away_ML'])) + 1)

        mast_picks['Return'] = np.where((mast_picks['Bet_Home'] == 1) & (mast_picks['Win_Bet'] == 1), mast_picks['Home_ML'], 0)
        mast_picks['Return'] = np.where((mast_picks['Bet_Away'] == 1) & (mast_picks['Win_Bet'] == 1), mast_picks['Away_ML'], mast_picks['Return'])
        mast_picks['Result'] = mast_picks['Return'] - 1
        mast_picks = mast_picks[['Game_ID', 'Home', 'Away', 'Game_Number', 'Home_ML', 'Away_ML', 'Home_Win_Prob', 'Away_Win_Prob', 'Home_EV', 'Away_EV', 'Home_Runs', 'Away_Runs', 'Bet_Home', 'Bet_Away', 'Win_Bet', 'Return', "Result"]]

        roi = mast_picks['Return'].sum() / mast_picks['Return'].count()
        units = mast_picks['Return'].sum() - mast_picks['Return'].count()

        current_record = str(mast_picks['Win_Bet'].sum()) + '-' + str(mast_picks['Win_Bet'].count() - mast_picks['Win_Bet'].sum())
        # format roi as %

        n_bets = mast_picks['Win_Bet'].count()
        se = np.sqrt((1 - roi) ** 2 / n_bets)
        tstat = roi / se
        pval = 1 - stats.t.cdf(tstat, n_bets - 1)

        roi = "{:.2%}".format(roi)
        tstat = "{:.4f}".format(tstat)
        pval = "{:.4f}".format(pval)
        # format units as 2 decimal places with +/- sign
        if units > 0:
            units = "+" + "{:.2f}".format(units)
        else:
            units = "{:.2f}".format(units)
        mast_picks.to_csv('data/picks_results.csv')

    def eval_results(self):
        self.get_results()
        self.merge_game_results()

if __name__ == '__main__':
    results = RetrieveResults()
    results.eval_results()

#%%
