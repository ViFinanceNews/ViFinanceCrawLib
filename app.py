from article_database.ArticleScraper import ArticleScraper
from article_database.Database import Database
from Summarizer.newsProcessor import NewsProcessor

class app:
    def __init__(self, url):
        self.db = Database()
        self.news_processor = NewsProcessor(ArticleScraper(url))
    
    def store_article(self):
        self.db.insert_record("article",self.news_processor.transform_article())
    
    def store_tags(self):
        tags = self.news_processor.get_tags()
        self.db.insert_record("tag", tags)

    def store_article_tags(self):
        self.db.insert_record("article_tag", self.news_processor.get_article_tags())

if __name__=="__main__":
    app = app("https://www.reuters.com/business/autos-transportation/tesla-aims-halve-cost-batteries-raise-production-20-million-vehicles-2023-2021-09-22/")
    app.store_article()
    app.store_tags()
    app.store_article_tags()
