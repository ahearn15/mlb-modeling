from bbm_scraper import BaseballMonsterScraper
from create_fm import CreateFeatureMatrix
from predict_games import PredictGames
from publish_picks import PublishPicks
from get_results import RetrieveResults

def main():
    # Scrape Data
    #print('Scraping data')
    scraper = BaseballMonsterScraper()
    scraper.scrape_data()
    #print('Data scraped successfully')

    # Generate Feature Matrix
    #print('Creating feature matrix')
    fm = CreateFeatureMatrix()
    fm.create_feature_matrix()
    #print('Feature matrix created successfully')

    # Make predictions
    #print('Making predictions')
    predictor = PredictGames()
    predictor.predict_games()
    #print('Predictions made successfully')

    #print('Publishing picks')
    publisher = PublishPicks()
    publisher.publish_picks_gsheets()
    #print('Picks published successfully')

    #print('Getting results')
    results = RetrieveResults()
    results.eval_results()

if __name__ == '__main__':
    main()
