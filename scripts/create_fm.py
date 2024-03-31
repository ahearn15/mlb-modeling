import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import warnings
import time
import os
import pickle
warnings.filterwarnings('ignore')


class CreateFeatureMatrix:
    def __init__(self):
        #self.today_date = time.strftime("%m-%d-%Y")
        self.today_date = datetime.now()
        self.today_date = self.today_date.strftime("%-m-%-d-%Y")
        self.fp = self.today_date.split('-')
        self.fp = '-'.join([x.zfill(2) for x in self.fp])
        year = self.today_date.split('-')[2]
        self.fp = f'data/monster_data/{year}/{self.fp}/'

    @staticmethod
    def merge_game_data(player_data, game_data):
        game_df = player_data.merge(game_data, on='Game_ID', how='inner')
        return game_df

    @staticmethod
    def group_clean_hit(hit_df):
        def wavg(group, avg_name, weight_name):
            d = group[avg_name]
            w = group[weight_name]
            try:
                return (d * w).sum() / w.sum()
            except ZeroDivisionError:
                return d.mean()

        hit_df['date'] = hit_df['date'].astype(str)
        grouped = hit_df.groupby(['Team', 'Game_ID'])
        numeric_cols = hit_df.select_dtypes(include=[np.number]).columns.tolist()

        sum_cols = ['AB', 'R', 'HR', 'RBI', 'SB']
        for col in sum_cols:
            numeric_cols.remove(col)

        numeric_cols.remove('g')

        w_hit1 = pd.DataFrame()

        for col in numeric_cols:
            w_hit1[col] = grouped.apply(wavg, col, 'AB')

        new_sum_cols = ['proj_' + col for col in sum_cols]
        w_hit1[new_sum_cols] = grouped.sum()[sum_cols]
        w_hit1['count'] = grouped.count()['Name']
        w_hit1.columns = ['hitter_' + col for col in w_hit1.columns]
        w_hit1 = w_hit1[w_hit1['hitter_count'] >= 9]
        w_hit1 = w_hit1.groupby('Game_ID').filter(lambda x: len(x) == 2)

        return w_hit1.reset_index()

    def group_clean_pit(self, pit_df):
        def wavg(group, avg_name, weight_name):
            d = group[avg_name]
            w = group[weight_name]
            try:
                return (d * w).sum() / w.sum()
            except ZeroDivisionError:
                return d.mean()

        pit_df['date'] = pit_df['date'].astype(str)

        pit_df['R'] = (pit_df['ERA'] / 9) * pit_df['IP']
        pit_df['WH'] = pit_df['WHIP'] * pit_df['IP']

        grouped = pit_df.groupby(['Team', 'Game_ID'])

        numeric_cols = pit_df.select_dtypes(include=[np.number]).columns.tolist()

        sum_cols = ['R', 'WH']
        for col in sum_cols:
            numeric_cols.remove(col)

        numeric_cols.remove('g')

        w_pit = pd.DataFrame()

        for col in numeric_cols:
            w_pit[col] = grouped.apply(wavg, col, 'IP')
        new_sum_cols = ['proj_' + col for col in sum_cols]
        w_pit[new_sum_cols] = grouped.sum()[sum_cols]
        w_pit['count'] = grouped.count()['Name']
        w_pit.columns = ['pitcher_' + col for col in w_pit.columns]
        w_pit = w_pit[w_pit['pitcher_count'] >= 6]
        w_pit = w_pit.groupby('Game_ID').filter(lambda x: len(x) == 2)
        return w_pit.reset_index()

    def read_hitters(self):
        year = self.today_date.split('-')[2]
        hit = pd.read_excel(
            self.fp + f'{self.today_date}_h.xls')

        hit['date'] = self.today_date

        lineups = pd.read_csv(self.fp + f'{self.today_date}_l.csv')

        lineups = lineups.rename(columns={'team code': 'Team',
                                          ' game_date': 'date',
                                          ' game_number': 'g',
                                          ' player name': 'Name'})
        lineups = lineups[['Team', 'date', 'g', 'Name']]

        lineups['date'] = lineups['date'].str.replace('/', '-')
        # take month and day from date
        lineups['date'] = lineups['date'].str.split('-').str[0] + '-' + lineups['date'].str.split('-').str[1] + '-' + year

        hit = hit.merge(lineups, on=['date', 'g', 'Name'], how='inner')
        hit['Team'] = hit['Team_y']
        hit = hit.drop(columns=['Team_x', 'Team_y'])

        #hit['Matchup'] = hit['Matchup'].apply(lambda x: ' '.join(x.split()[:3]))
        # group by team, game, and date and replace Matchup with mode of group
        #hit['Matchup'] = hit.groupby(['Team', 'g', 'date'])['Matchup'].transform(lambda x: x.mode()[0])
        #hit.to_csv('inspect.csv')
        hit['Game_ID'] = self.get_game_key(hit)
        hit = hit.drop(columns=['Own', 'Inj', 'Pos', 'vs. Pitcher',
                                'Odds', 'pR', 'opR', 'Time', 'Weather'])
        w_hit = self.group_clean_hit(hit)
        return w_hit

    def read_pitchers(self):
        pit = pd.read_excel(self.fp + f'{self.today_date}_p.xls')
        pit['date'] = self.today_date
        pit['Game_ID'] = self.get_game_key(pit)
        pit = pit.drop(columns=['Own', 'Inj', 'Pos', 'vs. Pitcher',
                                'Odds', 'pR', 'opR', 'Time', 'Weather'])
        w_pit = self.group_clean_pit(pit)
        return w_pit

    def get_game_key(self, df):
        # check if hitter or pitcher dataframe
        if (df['Matchup'].str[:2].isin(['R ', 'L ']).mean() == 1):
            df['Matchup'] = df['Matchup'].str[2:]


        df['date'] = self.today_date
        df['Opp'] = df['Matchup'].str.extract(r'^(\S+)')
        df['Home'] = np.where(df['Opp'].str.contains('@'), 0, 1)
        df['Home'] = df.apply(lambda x: x['Opp'][1:] if x['Opp'].startswith('@') else x['Team'], axis=1)
        df['Away'] = df.apply(lambda x: x['Team'] if x['Opp'].startswith('@') else x['Opp'], axis=1)
        df['date'] = df['date'].astype('datetime64[ns]')

        # Sort the team names alphabetically and create a unique identifier
        df['Game_ID'] = df['Away'] + '-' + df['Home'] + '-' + df['date'].dt.strftime('%Y%m%d') + '-' + df['g'].astype(
            str)
        # Sort the team names alphabetically and create a unique identifier
        game_id = df['Away'] + '-' + df['Home'] + '-' + df['date'].dt.strftime('%Y%m%d') + '-' + df['g'].astype(str)
        return game_id

    def read_game_df(self):
        year = self.today_date.split('-')[2]
        df = pd.read_excel(self.fp + f'{self.today_date}_h.xls')
        df['date'] = self.today_date

        df['date'] = df['date'].astype('datetime64[ns]')
        # Assuming df is your DataFrame
        df['Home'] = df.apply(lambda x: x['Matchup'][1:] if x['Matchup'].startswith('@') else x['Team'], axis=1)
        df['Away'] = df.apply(lambda x: x['Team'] if x['Matchup'].startswith('@') else x['Matchup'], axis=1)

        df['Home'] = df['Home'].str.split(' ').str[0]
        df['Away'] = df['Away'].str.split(' ').str[0]

        df['Game_ID'] = df['Away'] + '-' + df['Home'] + '-' + df['date'].dt.strftime('%Y%m%d') + '-' + df['g'].astype(
            str)
        df['Game_Number'] = df['g']
        df = df.drop(columns='g')

        df = df[['date', 'Team', 'Home', 'Away', 'Game_ID', 'Game_Number', 'Odds', 'Time', 'Weather']]
        df[['drop', 'Spread', 'drop', 'Total', 'ML']] = df['Odds'].str.split(' ', expand=True)
        df['Total'] = df['Total'].str.replace(')', '')
        df['ML'] = df['ML'].str.replace(')', '')
        df['ML'] = df['ML'].str.replace('(', '')
        df[['Temp', 'Wind_Dir', 'wind_dir2', 'Wind_Speed', 'Rain_Chance', 'Surface', '']] = df['Weather'].str.split(' ',
                                                                                                                    expand=True)
        df['Wind_Dir'] = df['Wind_Dir'] + df['wind_dir2']
        df[['Wind_Speed', 'Wind2']] = df['Wind_Speed'].str.split('-', expand=True)
        df['Wind_Speed'] = (pd.to_numeric(df['Wind_Speed']) + pd.to_numeric(df['Wind2'])) / 2
        df['Rain_Chance'] = pd.to_numeric(df['Rain_Chance'].str.replace('%', '')) / 100
        df['Surface'] = np.where(df['Surface'] == '', df[''], df['Surface'])
        df = df[['date', 'Time', 'Game_ID', 'Team', 'Home', 'Away', 'Game_Number', 'Spread', 'Total', 'ML', 'Temp',
                 'Wind_Dir', 'Wind_Speed', 'Rain_Chance', 'Surface']]

        home = df[df['Team'] == df['Home']].drop_duplicates(subset='Game_ID')
        away = df[df['Team'] == df['Away']].drop_duplicates(subset='Game_ID')

        home = home.rename(columns={'Spread': 'Home_Spread', 'ML': 'Home_ML'}).drop(columns='Team')
        away = away.rename(columns={'Spread': 'Away_Spread', 'ML': 'Away_ML'}).drop(columns='Team')

        df = home.merge(away, on=['Game_ID', 'date', 'Time', 'Home', 'Away', "Game_Number", "Total", "Temp", 'Wind_Dir',
                                  'Wind_Speed', 'Rain_Chance', 'Surface'])

        df['Date'] = df['date'].astype('datetime64[ns]')
        df['Time_EST'] = df['Time']

        df = df[['Game_ID', 'Date', 'Time_EST', 'Home', 'Away', 'Game_Number', 'Temp', 'Wind_Dir', 'Wind_Speed',
                 'Rain_Chance', 'Surface', 'Home_Spread', 'Away_Spread', 'Home_ML', 'Away_ML', 'Total']]

        # Time zone differences from Eastern Time (ET)
        time_differences = {
            'Eastern': 0,  # No change
            'Central': -1,  # 1 hour behind ET
            'Mountain': -2,  # 2 hours behind ET
            'Western': -3  # 3 hours behind ET
        }

        # Team to time zone mapping
        team_time_zones = {
            'ATL': 'Eastern', 'PHI': 'Eastern', 'TOR': 'Eastern', 'BAL': 'Eastern', 'TB': 'Eastern',
            'NYY': 'Eastern', 'CLE': 'Eastern', 'DET': 'Eastern', 'MIA': 'Eastern', 'BOS': 'Eastern',
            'WAS': 'Eastern', 'NYM': 'Eastern', 'PIT': 'Eastern', 'STL': 'Central', 'CIN': 'Central',
            'KC': 'Central', 'CHC': 'Central', 'MIL': 'Central', 'CHW': 'Central', 'MIN': 'Central',
            'HOU': 'Central', 'COL': 'Mountain', 'ARI': 'Mountain', 'LAA': 'Western', 'SEA': 'Western',
            'OAK': 'Western', 'SF': 'Western', 'SD': 'Western', 'LAD': 'Western', 'TEX': 'Western'
        }

        # Function to convert time from ET to local time
        def convert_to_local_time(row):
            home_team = row['Home']
            time_et_str = row['Time_EST']
            time_et = datetime.strptime(time_et_str, '%I:%M%p')

            time_zone = team_time_zones.get(home_team, 'Eastern')
            time_difference = time_differences[time_zone]

            local_time = time_et + timedelta(hours=time_difference)
            return local_time.strftime('%I:%M%p')

        # Applying the conversion
        df['Local_Time'] = df.apply(convert_to_local_time, axis=1)

        def categorize_time(time_str):
            time = datetime.strptime(time_str, '%I:%M%p')
            if time < datetime.strptime('2:45PM', '%I:%M%p'):
                return 'Early Afternoon'
            elif time <= datetime.strptime('6:00PM', '%I:%M%p'):
                return 'Mid-Afternoon'
            else:
                return 'Evening'

        # Applying the categorization
        df['Generalized_Time'] = df['Local_Time'].apply(categorize_time)
        return df

    def get_player_data(self):
        w_hit = self.read_hitters()
        w_pit = self.read_pitchers()
        w = w_hit.merge(w_pit, on=['Team', 'Game_ID'], how='inner')
        # Define additional statistics columns (like "Runs" and "Steals")
        stats_columns = w.drop(columns=['Team', 'Game_ID']).columns

        w_away = w[w['Game_ID'].str.split('-').str[0] == w['Team']]
        w_home = w[w['Game_ID'].str.split('-').str[0] != w['Team']]

        w_away.columns = ['Away', 'Game_ID'] + [col + '_away' for col in stats_columns.tolist()]
        w_home.columns = ['Home', 'Game_ID'] + [col + '_home' for col in stats_columns.tolist()]

        w_away = w_away.drop(columns=['Away'])
        w_home = w_home.drop(columns=['Home'])

        # # Merging the two DataFrames on 'GameKey'
        game_df = pd.merge(w_away, w_home, on='Game_ID', how='inner')

        return game_df

    def get_game_data(self):
        games = self.read_game_df()
        return games

    def get_win_prob(self, game_df):
        exp_hit_brier, exp_pit_brier = (1.3134641935977553, -0.08986798523145466)
        exp_hit_ll, exp_pit_ll = (1.3107707140044293, -0.08960026212210591)
        for score in ['log_loss', 'brier']:
            if score == 'log_loss':
                exp_hit = exp_hit_ll
                exp_pit = exp_pit_ll
            else:
                exp_hit = exp_hit_brier
                exp_pit = exp_pit_brier
            game_df[f'pr_win_home_hit_{score}'] = (game_df['hitter_proj_R_home'] ** exp_hit) / (
                    game_df['hitter_proj_R_home'] ** exp_hit + game_df['hitter_proj_R_away'] ** exp_hit)

            game_df[f'pr_win_away_hit_{score}'] = 1 - game_df[f'pr_win_home_hit_{score}']

            game_df[f'pr_win_home_pit_{score}'] = (game_df['pitcher_proj_R_home'] ** exp_pit) / (
                    game_df['pitcher_proj_R_home'] ** exp_pit + game_df['pitcher_proj_R_away'] ** exp_pit)

            game_df[f'pr_win_away_pit_{score}'] = 1 - game_df[f'pr_win_home_pit_{score}']

        return game_df

    def create_interactions(self, game_df):
        hitter_away = [col for col in game_df.columns if 'hitter_' in col
                       and 'proj' not in col
                       and '_away' in col]
        hitter_home = [col for col in game_df.columns if 'hitter_' in col
                       and 'proj' not in col
                       and '_home' in col]

        pitcher_away = [col for col in game_df.columns if 'pitcher_' in col
                        and 'proj' not in col
                        and '_away' in col]

        pitcher_home = [col for col in game_df.columns if 'pitcher_' in col
                        and 'proj' not in col
                        and '_home' in col]

        # Scaling the variables
        with open('models/fm_scaler.pkl', 'rb') as f:
           scaler = pickle.load(f)

        scaled_columns = hitter_away + hitter_home + pitcher_away + pitcher_home
        game_df[scaled_columns] = scaler.transform(game_df[scaled_columns])

        for hit_away in hitter_away:
            for pit_home in pitcher_home:
                game_df[f'{hit_away}_div_{pit_home}'] = game_df[hit_away] / game_df[pit_home]
                game_df[f'{hit_away}_x_{pit_home}'] = game_df[hit_away] * game_df[pit_home]

        for hit_home in hitter_home:
            for pit_away in pitcher_away:
                game_df[f'{hit_home}_div_{pit_away}'] = game_df[hit_home] / game_df[pit_away]
                game_df[f'{hit_home}_c_{pit_away}'] = game_df[hit_home] * game_df[pit_away]

        return game_df

    def create_dummies(self, game_df):
        dummy_cols = pd.get_dummies(game_df[['Wind_Dir', 'Surface', 'Generalized_Time']])
        game_df = pd.concat([game_df, dummy_cols], axis=1).drop(columns=['Wind_Dir', 'Surface', 'Generalized_Time'])
        game_df['Temp'] = pd.to_numeric(game_df['Temp'], errors='coerce')
        game_df['Temp'] = np.where(pd.isna(game_df['Temp']), 70, game_df['Temp'])
        game_df['Wind_Speed'] = np.where(pd.isna(game_df['Wind_Speed']), 0, game_df['Wind_Speed'])
        game_df['Rain_Chance'] = np.where(pd.isna(game_df['Rain_Chance']), 0, game_df['Rain_Chance'])

        return game_df

    def clean_odds(self, game_df):
        # convert american odds to decimal odds
        convert_to_decimal = lambda american_odds: (american_odds / 100) + 1 if american_odds > 0 else 1 - (
                100 / american_odds)

        game_df['Home_ML'] = pd.to_numeric(game_df['Home_ML'].str.replace('+', ''), errors='coerce')
        game_df['Away_ML'] = pd.to_numeric(game_df['Away_ML'].str.replace('+', ''), errors='coerce')
        game_df['Home_ML'] = game_df['Home_ML'].apply(convert_to_decimal)
        game_df['Away_ML'] = game_df['Away_ML'].apply(convert_to_decimal)

        game_df['Home_Spread'] = pd.to_numeric(game_df['Home_Spread'].str.replace('+', ''), errors='coerce')
        game_df['Away_Spread'] = pd.to_numeric(game_df['Away_Spread'].str.replace('+', ''), errors='coerce')
        game_df['Home_Spread'] = game_df['Home_Spread'].apply(convert_to_decimal)
        game_df['Away_Spread'] = game_df['Away_Spread'].apply(convert_to_decimal)

        game_df['Home_Underdog'] = np.where(game_df['Home_ML'] > game_df['Away_ML'], 1, 0)

        def calculate_implied_odds(df):
            # Calculate implied probabilities without considering vig
            df['implied_odds_home_without_vig'] = 1 / df['Home_ML']
            df['implied_odds_away_without_vig'] = 1 / df['Away_ML']

            # Calculate the total implied probability (sum of both probabilities without considering vig)
            total_implied_prob = df['implied_odds_home_without_vig'] + df['implied_odds_away_without_vig']
            df['pr_odds_home'] = df['implied_odds_home_without_vig'] / total_implied_prob
            df['pr_odds_away'] = df['implied_odds_away_without_vig'] / total_implied_prob

            # Drop the intermediate columns
            df.drop(['implied_odds_home_without_vig', 'implied_odds_away_without_vig'], axis=1, inplace=True)

            return df

        game_df = calculate_implied_odds(game_df)
        return game_df

    def create_feature_matrix(self):
        player_data = self.get_player_data()
        game_data = self.get_game_data()
        game_df = self.merge_game_data(player_data, game_data)
        game_df = self.create_interactions(game_df)
        game_df = self.get_win_prob(game_df)

        game_df = self.create_dummies(game_df)
        game_df = self.clean_odds(game_df)

        fm = game_df.set_index('Game_ID')
        year = self.today_date.split('-')[2]
        fm.to_csv(self.fp + f'{self.today_date}_fm.csv')

if __name__ == '__main__':
   fm = CreateFeatureMatrix()
   fm.create_feature_matrix()
