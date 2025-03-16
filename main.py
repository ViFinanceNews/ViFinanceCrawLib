from QualAna.QualAna import QualAnaIns
from QuantAna.QuantAna import QuantAnaIns
from QualAna.ArticleFactCheckUtility import ArticleFactCheckUtility
from QualAna.ScrapeAndTagArticles import ScrapeAndTagArticles
import time
import logging
import pprint

start_time = time.time()
app = ScrapeAndTagArticles()

articles_tag = app.search_and_scrape("Vietnam Economy in the first-fews month of 2025")
for article in articles_tag:
    print(article)


end_time = time.time()
elapsed_time = end_time - start_time  # Calculate elapsed time 

print(f"Elapsed time: {elapsed_time:.4f} seconds")

