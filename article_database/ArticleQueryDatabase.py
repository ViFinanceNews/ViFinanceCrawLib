from ViFinanceCrawLib.article_database.DataBaseCockroach import Database
from dotenv import load_dotenv
import os
import redis
import json
import time


class AQD:
    
    def __init__(self):
        """
        Initialize the AQD object, setting up connections to both CockroachDB and Redis.

        Loads environment variables and constructs database and Redis client objects.
        """
        load_dotenv()
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")

        cockroach_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=require"
        self.db = Database(database_url=cockroach_url)
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=os.getenv("REDIS_PORT"),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=True
        )
        self.redis_usr = redis.Redis(
            host=os.getenv("REDIS_HOST_USR_DATA"),
            port=os.getenv("REDIS_PORT"),
            password=os.getenv("REDIS_PASSWORD_USR_DATA"),
            ssl=True
        )

    def move_article_to_database(self, url, user_id):
        """
        Move an article from Redis to the SQL database.

        Args:
            url (str): The Redis key associated with the article.
            user_id (str): The ID of the user who saved the article.

        Returns:
            str: Status message indicating the outcome of the operation.
        """
        try:
            self.db.connect()
            print(f"📥 Attempting to move article from Redis to DB for URL: {url}")

            redis_article = self._get_article_from_redis(url)
            if not redis_article:
                return "Cannot retrieve article"

            article_id = self._insert_or_update_article(redis_article)
            if not article_id:
                return "Insert failed"

            tag_ids = self._process_tags(redis_article.get("tags", []))
            self._link_article_tags(article_id, tag_ids)
            vote_type = self.get_user_vote_for_url(user_id, url)
            self._link_user_article(user_id, article_id, vote_type)

            print("✅ Article moved to database successfully.")
            return "Success"

        except Exception as e:
            print(f"🚨 Critical error during move_to_database: {e}")
            return "Unexpected failure"

    def _get_article_from_redis(self, url):
        """
        Retrieve and decode article data from Redis using its URL.

        Args:
            url (str): The Redis key.

        Returns:
            dict | None: Decoded article data or None if retrieval fails.
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

    def _insert_or_update_article(self, article_data):
        """
        Insert article if title does not exist; otherwise, update existing article.

        Args:
            article_data (dict): The article fields.

        Returns:
            int | None: The inserted/updated article ID, or None on failure.
        """
        structured_data = {
            "author": article_data.get("author"),
            "title": article_data.get("title"),
            "url": article_data.get("url"),
            "image_url": article_data.get("image_url"),
            "date_publish": article_data.get("date_publish"),
            "up_vote": article_data.get("upvotes"),
        }
        title = structured_data["title"]

        # 1. Check if title exists and get article_id
        check_query = "SELECT article_id FROM article WHERE title = %s LIMIT 1;"
        row = self.db.execute_query(check_query, params=(title,), fetch_one=True)

        if row:
            # 2. If exists, update the article with new data
            article_id = row[0]
            update_query = """
                UPDATE article SET
                    author = %s,
                    url = %s,
                    image_url = %s,
                    date_publish = %s,
                    up_vote = up_vote + %s,
                    saved_at = CURRENT_TIMESTAMP
                WHERE article_id = %s
                RETURNING article_id;
            """
            params = (
                structured_data["author"],
                structured_data["url"],
                structured_data["image_url"],
                structured_data["date_publish"],
                structured_data["up_vote"],
                article_id,
            )
            updated_row = self.db.execute_query(update_query, params=params, fetch_one=True, commit=True)
            if updated_row:
                print(f"🔄 Updated article with ID: {updated_row[0]}")
                return updated_row[0]
            else:
                print("❌ Failed to update the article.")
                return None

        else:
            # 3. If not exists, insert new article
            insert_query = """
                INSERT INTO article (author, title, url, image_url, date_publish, up_vote)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING article_id;
            """
            params = tuple(structured_data.values())
            inserted_row = self.db.execute_query(insert_query, params=params, fetch_one=True, commit=True)
            if inserted_row:
                print(f"📰 Inserted article with ID: {inserted_row[0]}")
                return inserted_row[0]
            else:
                print("❌ Failed to insert the article.")
                return None

    def _process_tags(self, tags):
        """
        Ensure all tags exist in the 'tag' table and collect their IDs.

        Args:
            tags (list): A list of tag names.

        Returns:
            list[int]: A list of tag IDs.
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
        Link an article to its tags using the article_tag join table.

        Args:
            article_id (int): The article ID.
            tag_ids (list): List of tag IDs to associate.
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
        Link an article to a user in the account_article table or update their vote.

        Args:
            user_id (str): The user’s ID.
            article_id (int): The article’s ID.
            vote_type (int | None): The user’s vote on the article, if any.
        """
        try:
            vote = vote_type if vote_type in [-1, 0, 1] else 0
            if vote_type not in [-1, 0, 1]:
                print("⚠️ Invalid or missing vote_type. Defaulting to 0.")

            query = """
                INSERT INTO account_article (user_id, article_id, personal_vote)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, article_id)
                DO UPDATE SET personal_vote = EXCLUDED.personal_vote
            """
            self.db.execute_query(query, params=(user_id, article_id, vote), commit=True)
            print(f"🔄 Linked or updated user {user_id} for article {article_id} with vote {vote}")

        except Exception as e:
            print(f"❌ Failed to link or update user-article relation: {e}")

    def get_userID_from_session(self, SESSION_ID: str):
        """
        Retrieve user ID associated with a session ID from Redis.

        Args:
            SESSION_ID (str): The session identifier.

        Returns:
            str | None: The user ID if found, otherwise None.
        """
        session_key = f"session:{SESSION_ID}"
        session_data = self.redis_usr.get(session_key)
        
        if session_data is None:
            print("Session not found or expired.")
            return None

        try:
            session_data_dict = json.loads(session_data.decode("utf-8"))
            user_id = session_data_dict.get('userId')
            if user_id:
                return user_id
            else:
                print("Unauthorized – No userId in session")
                return None
        except json.JSONDecodeError:
            print("Session data is not in JSON format, returning raw data.")
            return session_data.decode()
    
    def move_query_to_history(self, user_id, query):
        """
        Move the user's search query to the user_history table in the database.

        Args:
            user_id (str): The ID of the user making the query.
            query (str): The query string to be stored.
        """
        try:
            self.db.connect()
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")

            # PostgreSQL uses %s, %s, %s for placeholders
            insert_query = """
                INSERT INTO user_history (user_id, user_history, user_history_time)
                VALUES (%s, %s, %s)
            """

            self.db.execute_query(insert_query, params=(user_id, query, current_time), commit=True)
            print(f"📝 Logged query for user {user_id} at {current_time}")

        except Exception as e:
            print(f"❌ Failed to log query: {e}")
    
    def get_user_vote_for_url(self, user_id: str, url: str) -> int:
        """
        Get the vote type (-1, 0, 1) a user has given to a specific article by URL.

        Args:
            user_id (str): The ID of the user.
            url (str): The article URL.

        Returns:
            int: Vote type (-1 = downvote, 0 = neutral/no vote, 1 = upvote).
        """
        try:
            user_votes_key = f"user:{user_id}:personal_vote"
            vote_raw = self.redis_usr.hget(user_votes_key, url)

            if vote_raw is None:
                return 0  # Default to NEUTRAL_VOTE if not found

            return int(vote_raw.decode("utf-8"))

        except Exception as e:
            print(f"❌ Error retrieving vote for user {user_id}, url {url}: {e}")
            return 0  # Fallback to neutral on error

    # Doing the upsert after the Session done
    def upsert_articles_from_user_hash(self, user_id: str, session_id: str):
        """
        Main method to:
        - Connect DB
        - Get filtered article keys from Redis
        - Upsert each article using _insert_or_update_article()
        - Delete the Redis key if upsert successful and TTL criteria met
        - Upsert will not include the vote data from user point of view (like the save mode with 0,1,-1) - but all the vote data will be recorded
        """

        session_key = f"session:{session_id}"  # this was map to user_id - the TTL = 1 hours
        user_key = f"user:{user_id}:personal_vote" # map to a dict with {url : personal_vote} - the TTL = Forever

        try:
            self.db.connect()
            print("🔄 Starting bulk upsert operation for articles from Redis.")

            article_map = self.redis_client.hgetall(user_key) # A dictionary
            
            urls_list = [key.decode('utf-8') for key in article_map.keys()]

            upserted_count = 0

            redis_data_list = self.redis_client.mget(urls_list)

            
            for key, redis_data in zip(urls_list, redis_data_list):
                if not redis_data:
                    continue
                
                if not redis_data:
                    print(f"⚠️ Empty data for key {key}")
                    continue

                try:
                    article_data = json.loads(redis_data.decode("utf-8"))
                except Exception as e:
                    print(f"❌ JSON decode error for key {key}: {e}")
                    continue

                title = article_data.get("title")
                date_publish = article_data.get("date_publish")

                if not title or not date_publish:
                    print(f"⚠️ Missing required field(s) title/date_publish for key: {key}")
                    continue

                # Use the reusable insert/update method here
                inserted_id = self._insert_or_update_article(article_data)

                if inserted_id: # article_id - Only delete Redis key if DB operation successful
                    session_exists = self.redis_client.exists(session_key)
                    ttl = self.redis_client.ttl(key)
                    if not session_exists: 
                        if (ttl == -1 or ttl > 3600): # if the TTL > 1 hour
                            self.redis_client.delete(key)
                            print(f"✅ Upserted article ID {inserted_id} and deleted Redis key: {key}")
                        print(f"✅ Upserted article ID {inserted_id} and not Redis key because self-expire: {key}")
                    upserted_count += 1
                else:
                    print(f"❌ Failed to upsert article from Redis key: {key}")

            # Check if session still exists 
            session_exists = self.redis_client.exists(session_key)
            if not session_exists: 
                self.redis_client.delete(user_key)
                print(f"🗑️ Session expired. Deleted user Redis hash for user for personal_vote {user_id}.")
            else:
                print(f"🟢 Session active. Kept Redis hash for user working {user_id}.")


            print(f"🔄 Upsert operation completed. Total upserted: {upserted_count}")
            return {"upserted": upserted_count}

        except Exception as e:
            print(f"❌ Critical error during bulk upsert operation: {e}")
            return {"error": str(e)}
    

