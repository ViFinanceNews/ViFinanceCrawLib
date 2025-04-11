from ViFinanceCrawLib.article_database.Database import Database
from ViFinanceCrawLib.QualAna.ArticleFactCheckUtility import ArticleFactCheckUtility
from dotenv import load_dotenv
import os
import pprint
import redis
import json
import time
import logging
NEUTRAL = 0
class ScrapeAndTagArticles:
    

    def __init__(self):
        load_dotenv()
        connection_str = os.getenv("CONNECTION_STR")
        
        self.db = Database(connection_string=connection_str)
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

        url_list = []

        # Step 2: Assign tags and store in Redis
        for article, tags in zip(articles, tag_batches):
            article_data = {
                "author": article.get("author", "Unknown"),
                "title": article.get("title", "No Title"),
                "url": article.get("url"),
                "image_url": article.get("image_url"),
                "date_publish": article.get("date_publish"),
                "main_text": article.get("main_text"),
                "tags": tags,  # Assign generated tags
                "upvotes": article.get("upvotes", 0),
                "vote_type": article.get("vote_type", NEUTRAL)  # Default to NEUTRAL if not present
            }

            # Store in Redis
            self.redis_client.set(article["url"], json.dumps(article_data), ex=3600)

            url_list.append(article["url"])

        print(url_list)
        return url_list      
    
    #move from redis to database
    def move_to_database(self,url):
        self.db.connect()
        # Step 1: Retrieve article from Redis
        redis_article = self.redis_client.get(url)
        if redis_article is not None:
            redis_article = redis_article.decode("utf-8")  # 🔥 Decode bytes to string
            redis_article = json.loads(redis_article)
            print(redis_article)
        else:
            return "Cannot retrieve None article"
        # Step 2: Insert Article into SQL Database
        article_data = {
            "author": redis_article["author"],
            "title": redis_article["title"],
            "url": redis_article["url"],
            "image_url": redis_article["image_url"],
            "date_publish": redis_article["date_publish"],
            "upvotes": redis_article["upvotes"],
            "vote_type": redis_article["vote_type"]
        }
        insert_query = "INSERT INTO article (author, title, url, image_url, date_publish) OUTPUT INSERTED.article_id VALUES (?, ?, ?, ?, ?)"
        article_id_row = self.db.execute_query(insert_query, params=(article_data["author"], article_data["title"], article_data["url"], article_data["image_url"], article_data["date_publish"]), 
                                               fetch_one=True, commit=True)
        
        if article_id_row:
            sql_article_id = article_id_row[0]
            print(f"📰 Moved Article to SQL with new ID {sql_article_id}")
            
            # Step 3: Insert Tags (if not exist) + retrieve tag_ids
            tag_ids = []
            for tag in redis_article["tags"]:
                tag_exist_query = "SELECT tag_id FROM tag WHERE tag_name = ?"
                existing_tag = self.db.execute_query(tag_exist_query, params=(tag), fetch_one=True, commit=True)
                if existing_tag:
                    tag_id = existing_tag[0]
                    print(f"🏷️ Existing Tag ID: {tag_id}")
                else:
                    # Insert tag
                    insert_tag_query = "INSERT INTO tag (tag_name) OUTPUT INSERTED.tag_id VALUES (?)"
                    tag_id_row = self.db.execute_query(insert_tag_query, params=(tag,), fetch_one=True, commit=True)
                    tag_id = tag_id_row[0]
                    print(f"🏷️ Inserted Tag: {tag} with ID {tag_id}")
                tag_ids.append(tag_id)

            # Step 4: Insert article_tag
            for tag_id in tag_ids:
                map_query = "INSERT INTO article_tag (article_id, tag_id) VALUES (?, ?)"
                self.db.execute_query(map_query, params=(sql_article_id, tag_id), commit=True)
                print(f"🔗 Mapped Article {sql_article_id} with Tag {tag_id}")

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
    
    def move_query(self, user_id,query):
        """
        Move the query to the database.

        Args:
            query (str): The query to be moved.
        """
        # Connect to the database
        self.db.connect()

        insert_query = "INSERT INTO user_history (user_id,user_history,user_history_time) VALUES (?,?,?)"
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.db.execute_query(insert_query, params=(user_id, query, current_time), commit=True)

# if __name__=="__main__":
#     processor = ScrapeAndTagArticles()
    
#     processor.search_and_scrape("Vàng")
