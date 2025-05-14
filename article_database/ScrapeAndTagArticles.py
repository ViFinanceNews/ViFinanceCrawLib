from ViFinanceCrawLib.article_database.DataBaseCockroach import Database
from ViFinanceCrawLib.QualAna.ArticleFactCheckUtility import ArticleFactCheckUtility
from dotenv import load_dotenv
import os
import redis
import json
import warnings
NEUTRAL = 0
class ScrapeAndTagArticles:
    
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=os.getenv("REDIS_PORT"),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=True
        )
        self.utility = ArticleFactCheckUtility()
    
    def search_and_scrape(self, query):
        # Step 1: Scrape articles
        articles = self.utility.search_web_fast(query, num_results=10)
        tag_batches = self.utility.generate_tags_batch(articles=articles)
        brief_des_batches = self.utility.generate_brief_descriptions_batch(articles=articles)

        response_result = list()

        # Step 2: Assign tags and store in Redis
        for article, tags, brief_des in zip(articles, tag_batches, brief_des_batches):
            article_data = {
                "author": article.get("author", "Unknown"),
                "title": article.get("title", "No Title"),
                "url": article.get("url"),
                "image_url": article.get("image_url"),
                "date_publish": article.get("date_publish"),
                "main_text": article.get("main_text"),
                "tags": tags,  # Assign generated tags
                "upvotes": article.get("upvotes", 0),
                "vote_type": article.get("vote_type", NEUTRAL),  # Default to NEUTRAL if not present
                "brief_des_batches": brief_des
            }

            # Store in Redis
            self.redis_client.set(article["url"], json.dumps(article_data), ex=3600) # Store in 1 hours
            instance_res = article_data.copy()
            instance_res.pop("main_text", None)
            response_result.append(instance_res)
            
        return response_result      
 
    def get_multiple_article(self, urls):
        """
        Retrieves multiple articles from Redis. If an article is not found, it returns None for that URL.

        Args:
            urls (list): List of article URLs.

        Returns:
            list: A list of article metadata dictionaries (or None if an article is missing).
        """

        # Retrieve multiple articles at once from Redis
        redis_articles = self.redis_client.mget(urls)

        # Decode and parse JSON for each article, handling None cases
        articles = [
            json.loads(article.decode("utf-8")) if article else None
            for article in redis_articles
        ]

        return articles

    def get_an_article(self, url):
        """
        Retrieves an article's main text and metadata from Redis or scrapes it if not found.

        Args:
            url (str): The URL of the article.

        Returns:
            dict: Article metadata and main text.
        """

        # 🔍 Check if the article is already cached in Redis
        redis_article = self.redis_client.get(url)
        
        if redis_article is not None:
            # ✅ Decode JSON from Redis
            redis_article = json.loads(redis_article.decode("utf-8"))
            print(f"✅ Found article in Redis: {redis_article['title']}")
        else:
            print(f"⚠️ Article not found in Redis. Scraping from URL: {url}")

            # 🔄 Scrape the article if not in Redis
            scraped_article = self.utility.scrape_articles_parallel([{"url": url}])  

            if not scraped_article or not scraped_article[0].get("main_text"):
                return {"error": "Failed to retrieve article content."}

            # ✅ Extract required fields
            redis_article = {
                "author": scraped_article[0].get("author", "Unknown"),
                "title": scraped_article[0].get("title", "No Title"),
                "url": url,
                "image_url": scraped_article[0].get("image_url"),
                "date_publish": scraped_article[0].get("date_publish"),
                "main_text": scraped_article[0].get("main_text"),
                "upvotes": scraped_article[0].get("upvotes", 0),
                "vote_type": NEUTRAL
            }
            

            # 🔄 Store in Redis for future use (TTL: 1 hour)
            self.redis_client.set(url, json.dumps(redis_article), ex=3600)

            print(f"✅ Scraped and cached: {redis_article['title']}")

        return redis_article  # ✅ Returns full article data
    
    