import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


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
                home_runs = extracted_numbers[0]
                away_runs = extracted_numbers[1]

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

        mast_games_df['Game_ID'] = mast_games_df['Away'] + '-' + mast_games_df['Home'] + '-' + mast_games_df['Date'].astype(str).replace('-', '', regex = True) + '-' + mast_games_df['Game_Number'].astype(str)
        mast_games_df = mast_games_df.set_index('Game_ID')
        mast_games_df.to_csv('data/game_results.csv')

if __name__ == '__main__':
    results = RetrieveResults()
    results.get_results()