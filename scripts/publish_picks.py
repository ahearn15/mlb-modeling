import pandas as pd
import numpy as np
import gspread
from datetime import datetime, timedelta
from scipy import stats
import os
import discord
import asyncio


class PublishPicks:
    def __init__(self):
        self.today_date = datetime.now()
        self.today_date = self.today_date.strftime("%-m-%-d-%Y")
        self.fp = self.today_date.split('-')
        self.fp = '-'.join([x.zfill(2) for x in self.fp])
        year = self.today_date.split('-')[2]
        self.fp = f'data/monster_data/{year}/{self.fp}/'
        self.today_picks = pd.read_csv(self.fp + f'{self.today_date}_picks.csv', dtype={'Home_ML': str, 'Away_ML': str})
        locked_picks_file = self.fp + f'{self.today_date}_locked_picks.csv'
        if os.path.exists(locked_picks_file):
            try:
                self.locked_picks = pd.read_csv(locked_picks_file, dtype=str)
            except pd.errors.EmptyDataError:
                self.locked_picks = pd.DataFrame()
        else:
            self.locked_picks = pd.DataFrame()
        if os.path.exists(self.fp + 'results_sent.txt'):
            with open(self.fp + 'results_sent.txt', 'r') as f:
                if f.read().strip() == 'Y':
                    self.results_sent = True
        else:
            self.results_sent = False

    def tidy_predictions(self):
        today = self.today_picks.copy()
        # sort by time
        today['Home_ML'] = today['Home_ML'].astype(str)
        today['Away_ML'] = today['Away_ML'].astype(str)

        today['Game'] = today['Home'] + ' (' + today['Home_ML'] + ') vs ' + today['Away'] + ' (' + today[
            'Away_ML'] + ')'
        today['Time'] = 'Game ' + today['Game_Number'].astype(str) + ' - ' + today['Time_EST']
        today['Official Pick'] = np.where(today['Bet_Home'] == 1, today['Home'] + ' ' + today['Home_ML'],
                                          np.where(today['Bet_Away'] == 1, today['Away'] + ' ' + today['Away_ML'],
                                                   'No bet'))
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
        today['Kelly Units'] = today['Kelly Units'].apply(
            lambda x: str(x).rstrip('0').rstrip('.') if '.' in str(x) else x)
        today['Kelly Units'] = np.where(today['Official Pick'] == 'No bet', '', today['Kelly Units'] + 'u')

        today_tidy = today.reset_index()[
            ['Game_ID', 'Game', 'Time', 'Confirmed', 'Official Pick', 'Pr(Win)', 'EV', 'As Of',
             'Kelly Pct.', 'Kelly Units']]
        today_tidy = today_tidy.rename(columns={'Confirmed': 'Lineups Confirmed', 'Kelly Units': 'Units'})
        today_tidy['Pr(Win)'] = today_tidy['Pr(Win)'].apply(lambda x: format(x, ".2%"))
        today_tidy['EV'] = today_tidy['EV'].apply(lambda x: format(x, ".2%"))
        today_tidy['Kelly Pct.'] = today_tidy['Kelly Pct.'].apply(lambda x: format(x, ".2%"))
        today_tidy['Pr(Win)'] = np.where(today_tidy['Pr(Win)'] == 'nan%', '', today_tidy['Pr(Win)'])
        today_tidy['EV'] = np.where(today_tidy['EV'] == 'nan%', '', today_tidy['EV'])
        today_tidy['Kelly Pct.'] = np.where(today_tidy['Kelly Pct.'] == 'nan%', '', today_tidy['Kelly Pct.'])

        # convert time to datetime
        today_tidy['Time2'] = today_tidy['Time'].str.split(' - ').str[1].str.replace('am', 'pm')
        today_tidy['Time2'] = pd.to_datetime(today_tidy['Time2'], format='%I:%M%p')

        today_tidy['Time2'] = today_tidy['Time2'].apply(lambda x: x.replace(year=datetime.now().year,
                                                                            month=datetime.now().month,
                                                                            day=datetime.now().day))

        # check if game start time is within 15 minutes
        now = datetime.now()
        for index, row in today_tidy.iterrows():
            game_time = row['Time2']  # assuming Time_EST is in '%H:%M' format
            if (game_time - now) <= timedelta(minutes=15):
                # check if game is already locked
                if (('Game_ID' in self.locked_picks.columns) and not
                self.locked_picks[self.locked_picks['Game_ID'] == row['Game_ID']].empty):
                    continue  # skip this iteration if game is already locked
                else:
                    self.locked_picks = pd.concat([self.locked_picks, today_tidy.loc[[index]]])
                    # check if Official Pick is not 'No bet'
                    if row['Official Pick'] != 'No bet':
                        self.send_game_to_discord_bot(today_tidy.loc[[index]])

        today_tidy = today_tidy.drop(columns='Time2')
        if 'Time2' in self.locked_picks.columns:
            self.locked_picks = self.locked_picks.drop(columns='Time2')
        # save locked picks to a file
        self.locked_picks.to_csv(self.fp + f'{self.today_date}_locked_picks.csv', index=False)
        return today_tidy

    def publish_picks_gsheets(self):
        today_tidy = self.tidy_predictions()
        today_tidy['Locked'] = 'N'
        # load locked picks and append to today_tidy
        try:
            locked_picks = pd.read_csv(self.fp + f'{self.today_date}_locked_picks.csv', dtype=str)
            locked_picks['Locked'] = 'Y'
        except pd.errors.EmptyDataError:
            locked_picks = pd.DataFrame()
        today_tidy = pd.concat([locked_picks, today_tidy])

        for col in today_tidy:
            today_tidy[col] = np.where(pd.isna(today_tidy[col]), '', today_tidy[col])
        today_tidy = today_tidy.drop_duplicates(subset='Game_ID').drop(columns='Game_ID')
        # print("Today's picks:")
        # print(today_tidy[today_tidy['Official Pick'] != 'No bet'].drop(columns = ['As Of']))
        # get last value of today_tidy['As Of']
        as_of = today_tidy['As Of'].iloc[-1]
        today_tidy = today_tidy.drop(columns='As Of')
        cols = ['MLB Picks', '', '', '', '', '', '', '', '']
        second_row = pd.DataFrame([f'As of: {as_of}', '', '', '', '', '', '', '', '']).T
        second_row.columns = cols
        third_row = pd.DataFrame(cols).T
        third_row.columns = cols
        third_row.iloc[0] = today_tidy.columns

        today_tidy.columns = cols
        today_tidy = pd.concat([second_row, third_row, today_tidy])
        blank_df = pd.DataFrame(index=range(15), columns=cols)
        blank_df = blank_df.fillna('')
        today_tidy = pd.concat([today_tidy, blank_df])
        gc = gspread.service_account(filename='misc/mlb-modeling-a9139a680fef.json')
        gc = gc.open('MLB Model Picks')
        sh = gc.worksheet("Today's Picks")
        sh.update([today_tidy.columns.values.tolist()] + today_tidy.values.tolist())

    def send_game_to_discord_bot(self, game):
        # read token from discord_token.txt
        with open('misc/discord_token.txt', 'r') as f:
            token = f.read()
        loop = asyncio.get_event_loop()
        client = discord.Client(intents=discord.Intents.default())
        guild_name = 'Algorhythm Bets'
        channel_name = 'mlb-official-picks'

        # send message to discord bot
        @client.event
        async def on_ready():
            guild = discord.utils.get(client.guilds,
                                      name=guild_name)
            channel = discord.utils.get(guild.channels,
                                        name=channel_name)

            message = (f"🚨OFFICIAL PICK🚨\n"
                       f"{game['Game'].values[0]}\n"
                       f"{game['Time'].values[0]}\n"
                       f"{game['Official Pick'].values[0]} "
                       f"({game['EV'].values[0]} EV)\n"
                       f"{game['Units'].values[0]} ({game['Kelly Pct.'].values[0]}) \n")
            await channel.send(message)
            # close client after all messages sent
            await client.close()

        try:
            loop.run_until_complete(client.start(token))
        except Exception as e:
            print(f'Error: {e}')
            loop.run_until_complete(client.close())

    def send_results_to_discord_bot(self, results_df):
        current_time = datetime.now().strftime("%-m/%-d/%Y %H:%M:%S")
        # check if time is between 9:00 am and 12:00 pm
        current_time = datetime.strptime(current_time, "%m/%d/%Y %H:%M:%S")
        if self.results_sent:
            return
        if not (9 <= current_time.hour < 12):
            return

        def calculate_record_and_units(df):
            win_bets = list(df[df['Win_Bet'] == 1].count())[0]
            lose_bets = list(df[df['Win_Bet'] == 0].count())[0]
            record = f"{win_bets}-{lose_bets}"

            units = round(df['Result_Kelly_Units'].sum() * 2) / 2
            units = f"+{units}" if units > 0 else str(units)

            return record, units

        yesterday = results_df[results_df['Date'] == results_df['Date'].max()]
        yesterday_record, yesterday_units = calculate_record_and_units(yesterday)

        season_record, season_units = calculate_record_and_units(results_df)

        roi = (results_df['Result_Kelly_Units'].sum() + results_df['Bet_Kelly_Units'].sum()) / results_df[
            'Bet_Kelly_Units'].sum() - 1

        n_bets = results_df['Win_Bet'].count()
        se = np.sqrt((1 - roi) ** 2 / n_bets)
        tstat = roi / se
        pval = 1 - stats.t.cdf(tstat, n_bets - 1)

        roi = "{:.2%}".format(roi)
        pval = "{:.3}".format(pval)

        message = (f"Yesterday: {yesterday_record} ({yesterday_units}u)\n"
                   f"Season record: {season_record} ({season_units}u)\n"
                   f"Season ROI: {roi} (p={pval})")

        with open('misc/discord_token.txt', 'r') as f:
            token = f.read()
        loop = asyncio.get_event_loop()
        client = discord.Client(intents=discord.Intents.default())
        guild_name = 'Algorhythm Bets'
        channel_name = 'mlb-daily-results'

        # send message to discord bot
        @client.event
        async def on_ready():
            guild = discord.utils.get(client.guilds,
                                      name=guild_name)
            channel = discord.utils.get(guild.channels,
                                        name=channel_name)

            await channel.send(message)
            # close client after all messages sent
            await client.close()

        try:
            loop.run_until_complete(client.start(token))
        except Exception as e:
            print(f'Error: {e}')
            loop.run_until_complete(client.close())
        with open(self.fp + 'results_sent.txt', 'w') as f:
            f.write('Y')

    def publish_results_gsheets(self):
        results = pd.read_csv('data/picks_results.csv')
        results = results.drop(columns=['Unnamed: 0'])
        results['Date'] = results['Game_ID'].str.split('-').str[2]
        results['Date'] = results['Date'].str[:4] + '-' + results['Date'].str[4:6] + '-' + results['Date'].str[6:]
        results['Game_ID'] = results['Date']
        results = results.drop(columns=['Date']).rename(columns={'Game_ID': 'Date'})
        self.send_results_to_discord_bot(results)
        roi = (results['Result_Kelly_Units'].sum() + results['Bet_Kelly_Units'].sum()) / results[
            'Bet_Kelly_Units'].sum() - 1
        units = results['Result_Kelly_Units'].sum()
        current_record = str(results['Win_Bet'].sum()) + '-' + str(
            results['Win_Bet'].count() - results['Win_Bet'].sum())
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

        cols = ['Season results'] + ([''] * 23)

        second_row = pd.DataFrame(
            [f'Record: {current_record}', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
             '', '', '', '']).T
        third_row = pd.DataFrame(
            [f'ROI: {roi}', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
             '']).T
        fourth_row = pd.DataFrame(
            [f'{units}u', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']).T
        fifth_row = pd.DataFrame(
            [f'T-stat: {tstat} (p = {pval})', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
             '', '', '', '', '']).T
        second_row.columns = cols
        third_row.columns = cols
        fourth_row.columns = cols
        fifth_row.columns = cols
        header = pd.concat([second_row, third_row, fourth_row, fifth_row])
        blank_row = pd.DataFrame({col: "" for col in results.columns}, index=[0])
        results = pd.concat([blank_row, results], axis=0)
        results.iloc[0] = results.columns
        results.columns = cols
        results = pd.concat([header, results], axis=0)

        gc = gspread.service_account(filename='misc/mlb-modeling-a9139a680fef.json')
        gc = gc.open('MLB Model Picks')
        sh = gc.worksheet("Results")
        sh.update([results.columns.values.tolist()] + results.values.tolist())


if __name__ == '__main__':
    publisher = PublishPicks()
    publisher.publish_picks_gsheets()
    publisher.publish_results_gsheets()
# %%
