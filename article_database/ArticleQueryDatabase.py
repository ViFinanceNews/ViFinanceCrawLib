from ViFinanceCrawLib.article_database.DataBaseCockroach import Database
from dotenv import load_dotenv
import os
import redis
import json
import time


class AQD:
    # To Interact & Manage the article object inside the Database
    def __init__(self):
        load_dotenv()
        database_url = os.getenv("DATABASE_URL")
        self.db = Database(database_url==database_url)
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=os.getenv("REDIS_PORT"),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=True
        )
    
    def move_to_database(self, url):
        try:
            self.db.connect()
            print(f"📥 Attempting to move article from Redis to DB for URL: {url}")

            # Step 1: Retrieve article from Redis
            redis_article = self.redis_client.get(url)
            if not redis_article:
                print("❌ Error: Cannot retrieve article — not found in Redis.")
                return "Cannot retrieve None article"

            try:
                redis_article = json.loads(redis_article.decode("utf-8"))
            except Exception as e:
                print(f"❌ Error decoding Redis article JSON: {e}")
                return "Failed to decode article"
            
            # Step 2: Insert article into PostgreSQL
            article_data = {
                "author": redis_article.get("author"),
                "title": redis_article.get("title"),
                "url": redis_article.get("url"),
                "image_url": redis_article.get("image_url"),
                "date_publish": redis_article.get("date_publish"),
            }

            insert_article_query = """
                INSERT INTO article (author, title, url, image_url, date_publish)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING article_id
            """
            article_id_row = self.db.execute_query(
                insert_article_query,
                params=tuple(article_data.values()),
                fetch_one=True,
                commit=True
            )

            if not article_id_row:
                print("❌ Failed to insert article into the database.")
                return "Insert failed"

            sql_article_id = article_id_row[0]
            print(f"📰 Inserted article with ID: {sql_article_id}")

            # Step 3: Handle tags
            tag_ids = []
            tags = redis_article.get("tags", [])
            if not isinstance(tags, list):
                print("⚠️ Warning: 'tags' should be a list.")
                tags = []

            for tag in tags:
                try:
                    select_tag_query = "SELECT tag_id FROM tag WHERE tag_name = $1 LIMIT 1"
                    existing_tag = self.db.execute_query(select_tag_query, params=(tag,), fetch_one=True)

                    if existing_tag:
                        tag_id = existing_tag[0]
                        print(f"🏷️ Found existing tag '{tag}' with ID {tag_id}")
                    else:
                        insert_tag_query = "INSERT INTO tag (tag_name) VALUES ($1) RETURNING tag_id"
                        tag_id_row = self.db.execute_query(insert_tag_query, params=(tag,), fetch_one=True, commit=True)
                        tag_id = tag_id_row[0]
                        print(f"🏷️ Inserted new tag '{tag}' with ID {tag_id}")

                    tag_ids.append(tag_id)

                except Exception as e:
                    print(f"❌ Error processing tag '{tag}': {e}")

            # Step 4: Link article with its tags
            map_query = "INSERT INTO article_tag (article_id, tag_id) VALUES ($1, $2)"
            for tag_id in tag_ids:
                try:
                    self.db.execute_query(map_query, params=(sql_article_id, tag_id), commit=True)
                    print(f"🔗 Linked Article {sql_article_id} with Tag {tag_id}")
                except Exception as e:
                    print(f"❌ Failed to link article {sql_article_id} to tag {tag_id}: {e}")

            print("✅ Article moved to database successfully.")
            return "Success"

        except Exception as e:
            print(f"🚨 Critical error during move_to_database: {e}")
            return "Unexpected failure"
        
    def move_query(self, user_id, query):
        """
        Move the user's search query to the user_history table in the database.

        Args:
            user_id (str): The ID of the user making the query.
            query (str): The query string to be stored.
        """
        try:
            self.db.connect()
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")

            # PostgreSQL uses $1, $2, $3 for placeholders
            insert_query = """
                INSERT INTO user_history (user_id, user_history, user_history_time)
                VALUES ($1, $2, $3)
            """

            self.db.execute_query(insert_query, params=(user_id, query, current_time), commit=True)
            print(f"📝 Logged query for user {user_id} at {current_time}")

        except Exception as e:
            print(f"❌ Failed to log query: {e}")