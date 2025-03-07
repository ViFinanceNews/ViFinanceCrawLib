from newsplease import NewsPlease
import pprint
"""
# Sample NewsArticle object (as a dictionary)
news_article = {
    "authors": ["VnExpress"],
    "date_download": "2025-02-28 21:18:28",
    "date_modify": None,
    "date_publish": "2024-10-25",
    "description": "Bé gái Tilly Smith, học lớp 6, người vùng Oxshott (Surrey, Anh), đã cứu sống hàng trăm người...",
    "filename": "https%3A%2F%2Fvnexpress.net%2Fhoc-de-song-hay-hoc-de-thi-4808158.html.json",
    "image_url": "https://i2-vnexpress.vnecdn.net/2024/10/24/bya-1729787393-8100-1729787407.jpg?w=1200&h=0&q=100&dpr=1&fit=crop&s=F9PVaZSrsVfCaMSiE_-26g",
    "language": "vi",
    "localpath": None,
    "title": "Học để sống hay học để thi?",
    "title_page": None,
    "title_rss": None,
    "source_domain": "vnexpress.net",
    "maintext": "Hôm đó, đang chơi trên bãi biển cùng mẹ và em gái 7 tuổi, Tilly chợt thấy màu nước biển...",
    "url": "https://vnexpress.net/hoc-de-song-hay-hoc-de-thi-4808158.html"
}
"""
class ArticleScraper:

    def __init__(self):
        return
    
    @staticmethod
    def scrape_article(url):
        """
        Scrapes an article from the given URL.

        Parameters:
        - url (str): The article URL.

        Returns:
        - Dict with article details (title, text, date, authors, etc.).
        """
        try:
            article = NewsPlease.from_url(url)
            return {
                "author": " ".join(article.authors) if article.authors else "Unknown",  # Default to ["Unknown"] if missing
                "title": article.title or "No title available",
                "date_publish": article.date_publish.strftime("%Y-%m-%d") if article.date_publish is not None else "Unknown",
                "description": article.description or "No description available",
                "main_text": article.maintext or "No content available",  # Fix attribute: `maintext`, not `main_text`
                "image_url": article.image_url or "No image available",
                "url": article.url or "No URL available",
                "date_download": article.date_download.strftime("%Y-%m-%d") if article.date_download is not None else "Unknown",
            }
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {e}")
            return None
    
    @staticmethod
    def scrape_multiple_articles(urls):
        """
        Scrapes multiple articles from a list of URLs concurrently.

        Parameters:
        - urls (list of str): List of article URLs.

        Returns:
        - List of dictionaries containing article details.
        """
        results = list()
        try:
            articles = NewsPlease.from_urls(urls, timeout=6)
            for url, article in articles.items():
                curr_item = dict()
                curr_item[url] = article
                results.append(curr_item)
            return results
        except Exception as e:
            print(f"❌ Failed to scrape {urls}: {e}")
            return None

    @staticmethod
    def decode_article_obj(self, article):
        """
        Decode the NewsArticle.NewsArticle object to data-point
        return: author, title, date_publish, description, main_text, image_url, url, date_download
        """
        if not article:
            return None  # Return None if article is missing
        return {
            "author": " ".join(article.authors) if article.authors else "Unknown",  # Default to ["Unknown"] if missing
            "title": article.title or "No title available",
            "date_publish": article.date_publish.strftime("%Y-%m-%d") if article.date_publish is not None else "Unknown",
            "description": article.description or "No description available",
            "main_text": article.maintext or "No content available",  # Fix attribute: `maintext`, not `main_text`
            "image_url": article.image_url or "No image available",
            "url": article.url or "No URL available",
            "date_download": article.date_download.strftime("%Y-%m-%d") if article.date_download is not None else "Unknown",
        }

# if __name__ == '__main__':
#    urls = ["https://vnexpress.net/dieu-tri-hoc-them-4851548.html", "https://vnexpress.net/hoc-de-song-hay-hoc-de-thi-4808158.html"]
#    scraper = ArticleScraper()
#    data = scraper.scrape_article("https://vnexpress.net/dieu-tri-hoc-them-4851548.html")
#    print(pprint.pprint(data))
#    datum = scraper.scrape_multiple_articles(urls=urls)
#    for element in datum:
#         for article in element.values():
#             data = dict(scraper.decode_article_obj(article))
#             # print(data['title'])  # ✅ Correct: Pass a single article
#             # print(data['title'])
#             for key, value in data.items():
#                 print("Key: " + key + " Value:  " + str(value))