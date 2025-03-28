from ViFinanceCrawLib.article_database.Database import Database
from ViFinanceCrawLib.QualAna.ArticleFactCheckUtility import ArticleFactCheckUtility
from dotenv import load_dotenv
import os
import pprint
import redis
import json

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
        articles = self.utility.search_web_fast(query, num_results=5)
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
                "tags": tags  # Assign generated tags
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
        
        # Step 2: Insert Article into SQL Database
        article_data = {
            "author": redis_article["author"],
            "title": redis_article["title"],
            "url": redis_article["url"],
            "image_url": redis_article["image_url"],
            "date_publish": redis_article["date_publish"]
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


if __name__=="__main__":
    processor = ScrapeAndTagArticles()
    
    processor.search_and_scrape("Vàng")
