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
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")

        cockroach_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=require;Connection Timeout=120"
        self.db = Database(cockroach_url)
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=os.getenv("REDIS_PORT"),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=True
        )
        
    
    

    def _get_article_from_redis(self, url):
        """
        Retrieve an article from the Redis cache by its URL.

        Args:
            url (str): The unique key (usually the article URL) used to look up the article in Redis.

        Returns:
            dict or None: Returns the article data as a Python dictionary if found and decoded successfully.
                        Returns None if the article is not found or decoding fails.
        """
        redis_data = self.redis_client.get(url)
        if not redis_data:
            print("❌ Error: Cannot retrieve article — not found in Redis.")
            return None
        try:
            return json.loads(redis_data.decode("utf-8"))
        except Exception as e:
            print(f"❌ Error decoding Redis article JSON: {e}")
            return None

    def _insert_article(self, article_data):
        """
        Insert a new article into the database and return its assigned article ID.

        Args:
            article_data (dict): A dictionary containing article information with keys like
                                'author', 'title', 'url', 'image_url', and 'date_publish'.

        Returns:
            int or None: The newly inserted article's ID if successful, otherwise None.
        """
        structured_data = {
            "author": article_data.get("author"),
            "title": article_data.get("title"),
            "url": article_data.get("url"),
            "image_url": article_data.get("image_url"),
            "date_publish": article_data.get("date_publish"),
        }
        print(f"📝 Article data to be inserted: {structured_data}")

        query = """
            INSERT INTO article (author, title, url, image_url, date_publish)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING article_id
        """
        row = self.db.execute_query(query, params=tuple(structured_data.values()), fetch_one=True, commit=True)
        if row:
            print(f"📰 Inserted article with ID: {row[0]}")
            return row[0]
        print("❌ Failed to insert article into the database.")
        return None

    def _process_tags(self, tags):
        """
        Ensure all tags exist in the database and return their corresponding tag IDs.

        This function checks if each tag already exists in the `tag` table.
        - If it exists: retrieves its `tag_id`.
        - If it doesn't exist: inserts it into the table and retrieves the new `tag_id`.

        Args:
            tags (list): A list of tag strings.

        Returns:
            list: A list of tag IDs (integers) that correspond to the input tags.
        """
        tag_ids = []
        if not isinstance(tags, list):
            print("⚠️ Warning: 'tags' should be a list.")
            return tag_ids

        for tag in tags:
            try:
                select_query = "SELECT tag_id FROM tag WHERE tag_name = %s LIMIT 1"
                row = self.db.execute_query(select_query, params=(tag,), fetch_one=True)
                if row:
                    tag_id = row[0]
                    print(f"🏷️ Found existing tag '{tag}' with ID {tag_id}")
                else:
                    insert_query = "INSERT INTO tag (tag_name) VALUES (%s) RETURNING tag_id"
                    tag_id_row = self.db.execute_query(insert_query, params=(tag,), fetch_one=True, commit=True)
                    tag_id = tag_id_row[0]
                    print(f"🏷️ Inserted new tag '{tag}' with ID {tag_id}")
                tag_ids.append(tag_id)
            except Exception as e:
                print(f"❌ Error processing tag '{tag}': {e}")
        return tag_ids

    def _link_article_tags(self, article_id, tag_ids):
        """
        Link an article to multiple tags by inserting records into the article_tag mapping table.

        This function takes the article ID and a list of tag IDs, then creates entries in the
        `article_tag` table to establish many-to-many relationships between the article and tags.

        Args:
            article_id (int): The ID of the article to be linked.
            tag_ids (list): A list of tag IDs to be associated with the article.
        """
        map_query = "INSERT INTO article_tag (article_id, tag_id) VALUES (%s, %s)"
        for tag_id in tag_ids:
            try:
                self.db.execute_query(map_query, params=(article_id, tag_id), commit=True)
                print(f"🔗 Linked Article {article_id} with Tag {tag_id}")
            except Exception as e:
                print(f"❌ Failed to link article {article_id} to tag {tag_id}: {e}")

    def _link_user_article(self, user_id, article_id, vote_type):
        """
            Link a user to an article with their personal vote.

            This method creates a record in the `account_article` table to associate a user
            with an article. It also stores the user's vote on the article.

            Args:
                user_id (str): The ID of the user interacting with the article.
                article_id (int): The ID of the article.
                vote_type (int): The user's vote on the article. Expected values are:
                                -1 (downvote), 0 (neutral), or 1 (upvote).
                                If the value is invalid, defaults to 0.
        """
        try:
            vote = vote_type if vote_type in [-1, 0, 1] else 0
            if vote_type not in [-1, 0, 1]:
                print("⚠️ Invalid or missing vote_type. Defaulting to 0.")
            query = """
                INSERT INTO account_article (user_id, article_id, personal_vote)
                VALUES (%s, %s, %s)
            """
            self.db.execute_query(query, params=(user_id, article_id, vote), commit=True)
            print(f"👤 Linked user {user_id} to article {article_id} with vote {vote}")
        except Exception as e:
            print(f"❌ Failed to link user to article: {e}")


    # moving the Article from Redis to Postgres SQL
    def move_article_to_database(self, url, user_id):
        """
        Move an article from Redis to the database and link it with the user.

        This method retrieves an article from the Redis cache, inserts it into the 
        database, processes any associated tags, and links the article to the user 
        who interacted with it. It also handles inserting and associating vote data 
        with the article.

        Args:
            url (str): The URL of the article to be moved.
            user_id (str): The ID of the user interacting with the article.

        Returns:
            str: A message indicating the result of the operation.
                - "Success" if the article was successfully moved and linked.
                - "Cannot retrieve article" if the article wasn't found in Redis.
                - "Insert failed" if inserting the article into the database fails.
                - "Unexpected failure" if any unexpected error occurs during the process.
        """
        try:
            self.db.connect()
            print(f"📥 Attempting to move article from Redis to DB for URL: {url}")

            redis_article = self._get_article_from_redis(url)
            print(type(redis_article))
            if not redis_article:
                return "Cannot retrieve article"

            article_id = self._insert_article(redis_article)
            if not article_id:
                return "Insert failed"

            tag_ids = self._process_tags(redis_article.get("tags", []))
            self._link_article_tags(article_id, tag_ids)

            self._link_user_article(user_id, article_id, redis_article.get("vote_type"))
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