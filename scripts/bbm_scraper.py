from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import os
import pandas as pd
import json
from bs4 import BeautifulSoup
from selenium.webdriver.support.select import Select
import numpy as np
from datetime import datetime

class BaseballMonsterScraper:
    def __init__(self):
        # make directory of today's date if it doesn't exist
        today_date = time.strftime("%m-%d-%Y")
        download_path = f'data/monster_data/2024/{today_date}'
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        self.orig_wd = os.getcwd()
        cookie_file = 'misc/cookies.txt'
        with open(cookie_file, 'r') as f:
            cookies_ = f.read()
        self.cookies_ = json.loads(cookies_)
        os.chdir(download_path)
        # remove all files in download path that don't begin with today's date
        today_date_no_zero = datetime.now().strftime("%-m-%-d-%Y")
        files = os.listdir()
        for file in files:
            if not file.startswith(today_date_no_zero):
                os.remove(file)
        self.download_path = os.getcwd()
        self.driver = self.create_driver()

    def create_driver(self):
        s = Service(ChromeDriverManager().install())
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--headless=new")
        prefs = {"download.default_directory": self.download_path}
        chrome_options.add_experimental_option("prefs", prefs)
        driver_ = webdriver.Chrome(options=chrome_options)
        driver_.implicitly_wait(5)
        url = "https://baseballmonster.com/lineups.aspx"
        driver_.get(url)
        driver_.delete_all_cookies()
        for cookie in self.cookies_:
            if 'sameSite' in cookie:
                if cookie['sameSite'] == None:
                    cookie['sameSite'] = 'None'
                if cookie['sameSite'] == 'lax':
                    cookie['sameSite'] = 'Lax'
                if cookie['sameSite'] == 'strict':
                    cookie['sameSite'] = 'Strict'
                if cookie['sameSite'] == 'no_restriction':
                    cookie['sameSite'] = 'None'
            try:
                driver_.add_cookie(cookie)
            except:
                pass
        driver_.get('https://baseballmonster.com/lineups.aspx')
        return driver_

    def scrape_pitchers(self):
        self.driver.get('https://baseballmonster.com/dailyprojections.aspx')
        lrpa = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_ShowLRPACheckBox"]"""
        woba = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_wOBACheckBox"]"""
        iso = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_isoCheckBox"]"""
        k = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_kCheckBox"]"""
        swstr = '''//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_SwStrCheckBox"]'''
        opp_woba = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_oppWOBACheckBox"]"""
        opp_iso = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_oppISOCheckBox"]"""
        opp_k = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_oppKCheckBox"]"""
        opp_swstr = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_oppSwStrCheckBox"]"""
        adv_stats = [lrpa, woba, iso, k, swstr, opp_woba, opp_iso, opp_k, opp_swstr]
        for xpath in adv_stats:
            box = self.driver.find_element(By.XPATH, xpath)
            box.click()
        dropdown = Select(self.driver.find_element(By.XPATH, value='//*[@id="PlayerTypeFilterControl"]'))
        dropdown.select_by_value('P')
        download_button = self.driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_ExcelButton"]')
        download_button.click()
        source = self.driver.page_source
        soup = BeautifulSoup(source, 'html.parser')
        date = soup.find('input', {'name': 'ctl00$ContentPlaceHolder1$StartDateTextBox'})
        date = date['value'].replace("/", "-")
        # rename most recently downloaded file
        # Get a list of all the files in the current directory.
        files = os.listdir()
        # Find the file with the most recent modification time.
        most_recent_file = max(files, key=os.path.getmtime)
        # Rename the file.
        os.rename(most_recent_file, f"{date}_p.xls")

    def scrape_hitters(self):
        self.driver.get('https://baseballmonster.com/dailyprojections.aspx')
        lrpa = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_ShowLRPACheckBox"]"""
        woba = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_wOBACheckBox"]"""
        iso = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_isoCheckBox"]"""
        k = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_kCheckBox"]"""
        swstr = '''//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_SwStrCheckBox"]'''
        opp_woba = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_oppWOBACheckBox"]"""
        opp_iso = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_oppISOCheckBox"]"""
        opp_k = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_oppKCheckBox"]"""
        opp_swstr = """//*[@id="ContentPlaceHolder1_AdvancedStatsUserControl1_oppSwStrCheckBox"]"""
        adv_stats = [lrpa, woba, iso, k, swstr, opp_woba, opp_iso, opp_k, opp_swstr]
        for xpath in adv_stats:
            box = self.driver.find_element(By.XPATH, xpath)
            box.click()
        dropdown = Select(self.driver.find_element(By.XPATH, value='//*[@id="PlayerTypeFilterControl"]'))
        dropdown.select_by_value('H')
        download_button = self.driver.find_element(By.XPATH, '//*[@id="ContentPlaceHolder1_ExcelButton"]')
        download_button.click()
        source = self.driver.page_source
        soup = BeautifulSoup(source, 'html.parser')
        date = soup.find('input', {'name': 'ctl00$ContentPlaceHolder1$StartDateTextBox'})
        date = date['value'].replace("/", "-")
        # rename most recently downloaded file
        # Get a list of all the files in the current directory.
        files = os.listdir()
        # Find the file with the most recent modification time.
        most_recent_file = max(files, key=os.path.getmtime)
        # Rename the file.
        os.rename(most_recent_file, f"{date}_h.xls")

    def scrape_games(self):
        source = self.driver.page_source
        soup = BeautifulSoup(source, 'html.parser')
        date = soup.find('input', {'name': 'ctl00$ContentPlaceHolder1$StartDateTextBox'})
        date = date['value'].replace("/", "-")
        games = soup.find('table', {'class': 'table table-bordered table-hover table-sm base-td-small datatable ml-0'})
        table_body = games.find('tbody')
        if games is not None:
            data = []
            rows = table_body.find_all('tr')

            for row in rows:
                cols = row.find_all('td')
                cols = [ele.text.strip() for ele in cols]
                data.append([ele for ele in cols]) # Get rid of empty values

            games = pd.DataFrame(data)
            games.columns = ['drop', 'away', 'drop', 'home', 'drop', 'time', 'drop', 'weather']
            games = games.drop(columns = ['drop'])

        games.to_csv(self.download_path + f'/{date}_g.csv')
    @staticmethod
    def scrape_lineups():
        today_date = time.strftime("%m-%d-%Y")

        url = f'https://baseballmonster.com/lineups.aspx?csv=1&d={today_date}'
        lineups = pd.read_csv(url)
        month, day, year = today_date.split('-')
        formatted_date = f"{int(month)}-{int(day)}-{year}"
        lineups.to_csv(f'{formatted_date}_l.csv')

    def scrape_data(self):
        self.scrape_pitchers()
        self.scrape_hitters()
        self.scrape_games()
        self.driver.quit()
        self.scrape_lineups()
        os.chdir(self.orig_wd)

if __name__ == '__main__':
   scraper = BaseballMonsterScraper()
   scraper.scrape_data()


#%%
