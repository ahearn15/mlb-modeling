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
from matplotlib import pyplot as plt
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
                read_locked = False
                for file in files:
                    # first check if _locked_picks.csv exists
                    if '_locked_picks.csv' in file:
                        try:
                            read_locked = True
                            daily_picks = pd.read_csv(dir + '/' + file)
                            # need to convert CHC (-113) vs MIL (+103) to Home: CHC, Away: MIL, Home_ML: -113, Away_ML: +103
                            daily_picks['Away'] = daily_picks['Game_ID'].str.split('-').str[0]
                            daily_picks['Home'] = daily_picks['Game_ID'].str.split('-').str[1]
                            daily_picks['Home_ML'] = pd.to_numeric(daily_picks['Game'].str.split(' ').str[1].str.replace('(', '').str.replace(')', '').str.replace('+', ''))
                            daily_picks['Away_ML'] = pd.to_numeric(daily_picks['Game'].str.split(' ').str[-1].str.replace('(', '').str.replace(')', '').str.replace('+', ''))

                            # get last char in game_id
                            daily_picks['Game_Number'] = daily_picks['Game_ID'].str[-1]
                            daily_picks['Time_EST'] = daily_picks['Time'].str.split('-').str[-1]
                            daily_picks['Bet_Home'] = np.where(daily_picks['Official Pick'].str.split(' ').str[0] == daily_picks['Home'], 1, 0)
                            daily_picks['Bet_Away'] = np.where(daily_picks['Official Pick'].str.split(' ').str[0] == daily_picks['Away'], 1, 0)

                            daily_picks['Home_Win_Prob'] = np.where(daily_picks['Bet_Home'] == 1,
                                                                    pd.to_numeric(daily_picks['Pr(Win)'].str.replace('%', ''))/100,
                                                                    1 - pd.to_numeric(daily_picks['Pr(Win)'].str.replace('%', ''))/100)
                            daily_picks['Away_Win_Prob'] = 1 - daily_picks['Home_Win_Prob']

                            daily_picks['Home_ML_Dec'] = np.where(daily_picks['Home_ML'] > 0, (daily_picks['Home_ML'] / 100) + 1, (100 / abs(daily_picks['Home_ML'])) + 1)
                            daily_picks['Away_ML_Dec'] = np.where(daily_picks['Away_ML'] > 0, (daily_picks['Away_ML'] / 100) + 1, (100 / abs(daily_picks['Away_ML'])) + 1)

                            daily_picks['Home_EV'] = daily_picks['Home_Win_Prob'] * (daily_picks['Home_ML_Dec'] - 1) - (1 - daily_picks['Home_Win_Prob'])
                            daily_picks['Away_EV'] = daily_picks['Away_Win_Prob'] * (daily_picks['Away_ML_Dec'] - 1) - (1 - daily_picks['Away_Win_Prob'])

                            daily_picks['Home_Kelly'] = ((daily_picks['Home_ML_Dec'] - 1) * daily_picks['Home_Win_Prob'] - (1 - daily_picks['Home_Win_Prob'])) / (daily_picks['Home_ML_Dec'] - 1)
                            daily_picks['Away_Kelly'] = ((daily_picks['Away_ML_Dec'] - 1) * daily_picks['Away_Win_Prob'] - (1 - daily_picks['Away_Win_Prob'])) / (daily_picks['Away_ML_Dec'] - 1)
                            daily_picks['Priority'] = 1
                            mast_picks = pd.concat([mast_picks, daily_picks], axis = 0)

                        except pd.errors.EmptyDataError:
                            pass
                    # use _picks.csv if _locked_picks.csv does not exist
                    elif '_picks.csv' in file and read_locked == False:
                        daily_picks = pd.read_csv(dir + '/' + file)
                        daily_picks['Priority'] = 0

                        mast_picks = pd.concat([mast_picks, daily_picks], axis = 0)

        mast_picks = mast_picks.sort_values(by = ['Game_ID', 'Priority'], ascending=False)
        mast_picks = mast_picks.drop_duplicates(subset = 'Game_ID', keep = 'first')
        # only get games that were bet on
        mast_picks['Bet'] = mast_picks['Bet_Home'] + mast_picks['Bet_Away']
        mast_picks = mast_picks[mast_picks['Bet'] == 1]
        mast_picks = mast_picks.drop(columns = ['As Of', 'Confirmed'])

        results = pd.read_csv('data/game_results.csv')
        mast_picks = mast_picks.merge(results[['Game_ID', 'Home_Runs', 'Away_Runs', 'Date']], on = ['Game_ID'], how = 'inner')
        mast_picks['Win_Bet'] = np.where(
            ((mast_picks['Bet_Home'] == 1) & (mast_picks['Home_Runs'] > mast_picks['Away_Runs'])) |
            ((mast_picks['Bet_Away'] == 1) & (mast_picks['Home_Runs'] < mast_picks['Away_Runs'])), 1, 0)

        # convert ml to decimal
        mast_picks['Home_ML'] = np.where(mast_picks['Home_ML'] > 0, (mast_picks['Home_ML'] / 100) + 1, (100 / abs(mast_picks['Home_ML'])) + 1)
        mast_picks['Away_ML'] = np.where(mast_picks['Away_ML'] > 0, (mast_picks['Away_ML'] / 100) + 1, (100 / abs(mast_picks['Away_ML'])) + 1)

        mast_picks['Return'] = np.where((mast_picks['Bet_Home'] == 1) & (mast_picks['Win_Bet'] == 1), mast_picks['Home_ML'], 0)
        mast_picks['Return'] = np.where((mast_picks['Bet_Away'] == 1) & (mast_picks['Win_Bet'] == 1), mast_picks['Away_ML'], mast_picks['Return'])
        mast_picks['Result'] = mast_picks['Return'] - 1
        mast_picks = mast_picks[['Game_ID', 'Date', 'Home', 'Away', 'Game_Number', 'Home_ML', 'Away_ML', 'Home_Win_Prob', 'Away_Win_Prob', 'Home_EV', 'Away_EV', 'Home_Runs', 'Away_Runs', 'Bet_Home', 'Bet_Away', 'Win_Bet', 'Return', "Result"]]
        mast_picks['Home_Kelly'] = (((mast_picks['Home_ML'] - 1) * mast_picks['Home_Win_Prob']) - (mast_picks['Away_Win_Prob'])) / (mast_picks['Home_ML'] - 1)
        mast_picks['Away_Kelly'] = (((mast_picks['Away_ML'] - 1) * mast_picks['Away_Win_Prob']) - (mast_picks['Home_Win_Prob'])) / (mast_picks['Away_ML'] - 1)

        # capping home and away kelly at .25
        mast_picks['Home_Kelly'] = np.where(mast_picks['Home_Kelly'] > .25, .25, mast_picks['Home_Kelly'])
        mast_picks['Away_Kelly'] = np.where(mast_picks['Away_Kelly'] > .25, .25, mast_picks['Away_Kelly'])

        mast_picks['Bet_Kelly'] = np.where((mast_picks['Bet_Home'] == 1), mast_picks['Home_Kelly'], mast_picks['Away_Kelly'])

        mast_picks['Return_Kelly'] = np.where((mast_picks['Bet_Home'] == 1) & (mast_picks['Win_Bet'] == 1), mast_picks['Home_Kelly'] * mast_picks['Home_ML'], 0)
        mast_picks['Return_Kelly'] = np.where((mast_picks['Bet_Away'] == 1) & (mast_picks['Win_Bet'] == 1), mast_picks['Away_Kelly'] * mast_picks['Away_ML'], mast_picks['Return_Kelly'])
        mast_picks['Result_Kelly'] = np.where(mast_picks['Bet_Home'] == 1, mast_picks['Return_Kelly'] - mast_picks['Home_Kelly'],mast_picks['Return_Kelly'] - mast_picks['Away_Kelly'])


        # standardize results to Units where 1 unit = .05 kelly
        mast_picks['Bet_Kelly_Units'] = mast_picks['Bet_Kelly'] / .05
        mast_picks['Result_Kelly_Units'] = mast_picks['Result_Kelly'] / .05

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
        mast_picks = mast_picks.sort_values(by = 'Date')

        daily = mast_picks.groupby('Date')[['Bet_Kelly_Units', 'Result_Kelly_Units']].sum()
        daily.to_csv('data/daily_results.csv')

        mast_picks.to_csv('data/picks_results.csv')

    def create_fig(self):
        res = pd.read_csv('data/picks_results.csv')
        res = res.drop(columns=['Unnamed: 0'])
        res['Date'] = res['Game_ID'].str.split('-').str[2]
        res['Date'] = res['Date'].str[:4] + '-' + res['Date'].str[4:6] + '-' + res['Date'].str[6:]
        units = round(res['Result_Kelly_Units'].sum(),1)

        res = res.groupby('Date')['Result_Kelly_Units'].sum().reset_index()
        res['Cumulative_Units'] = res['Result_Kelly_Units'].cumsum()
        # add a zero as first row, indicating start of season
        res = pd.concat([pd.DataFrame({'Date': '2024-03-27', 'Result_Kelly_Units': [0], 'Cumulative_Units': [0]}), res]).reset_index(drop=True)
        res['col'] = np.where(res['Result_Kelly_Units'] < 0, '#b81500', '#028241')

        # make barplot of daily results, color by positive/negative
        plt.figure(figsize=(12, 9))
        plt.bar(res['Date'], res['Result_Kelly_Units'], color=res['col'], alpha=0.8)
        plt.xticks(rotation=90)
        # add cumulative line
        plt.plot(res['Date'], res['Cumulative_Units'], color='#006e9f')
        # label last point in line
        plt.text(res['Date'].iloc[-1], res['Cumulative_Units'].iloc[-1], f'+{units}u', fontsize=12)

        # remove right spine
        plt.gca().spines['right'].set_visible(False)
        # remove top spine
        plt.gca().spines['top'].set_visible(False)

        # add horizontal line at 0
        plt.axhline(0, color='black', linestyle='--')

        # label y axis "Units"
        plt.ylabel('Units')
        # label x axis "Date"
        plt.xlabel('Date')

        plt.title('Season results')
        # save plot
        plt.savefig('data/season_results.png')


    def eval_results(self):
            self.get_results()
            self.merge_game_results()
            self.create_fig()


if __name__ == '__main__':
    results = RetrieveResults()
    results.eval_results()
