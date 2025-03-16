from article_database.Database import Database
from QualAna.ArticleFactCheckUtility import ArticleFactCheckUtility
from dotenv import load_dotenv
import os
import pprint

class ScrapeAndTagArticles:

    def __init__(self):
        load_dotenv(".devcontainer/devcontainer.env")
        connection_str = os.getenv("CONNECTION_STR")
        
        self.db = Database(connection_string=connection_str)
        self.utility = ArticleFactCheckUtility()
    
    def search_and_scrape(self, query):
        
        pp = pprint.PrettyPrinter(indent=4)
        # Step 1: Scrape articles
        articles = self.utility.search_web(query, num_results=10)
        self.db.connect()
        for article in articles:
            content = article["main_text"]
            tags = self.utility.generate_tags(content)
            article["tags"] = tags

            # Step 2: Insert Article -> Retrieve article_id
            article_data = {
                "author": article["author"],
                "title": article["title"],
                "url": article["url"],
                "image_url": article["image_url"],
                "date_publish": article["date_publish"]
            }
            insert_query = "INSERT INTO article (author, title, url, image_url, date_publish) OUTPUT INSERTED.article_id VALUES (?, ?, ?, ?, ?)"
            article_id_row = self.db.execute_query(insert_query, params=(article_data["author"], article_data["title"], article_data["url"], article_data["image_url"], article_data["date_publish"]), 
                                                   fetch_one=True, commit=True)
            
            if article_id_row:
                article_id = article_id_row[0]
                print(f"📰 Inserted Article ID: {article_id}")
                
                # Step 3: Insert Tags (if not exist) + retrieve tag_ids
                tag_ids = []
                for tag in tags:
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
                    self.db.execute_query(map_query, params=(article_id, tag_id), commit=True)
                    print(f"🔗 Mapped Article {article_id} with Tag {tag_id}")
        
        self.db.close()
        print("✅ Done processing articles and tags.")
        return articles 

 
