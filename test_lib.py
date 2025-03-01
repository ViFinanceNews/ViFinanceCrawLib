from article_database.ArticleScraper import ArticleScraper
from article_database.SearchEngine import SearchEngine
from article_database.Database import Database
from article_database.TitleCompleter import TitleCompleter
import pprint
def main(query = "Giá xăng dầu", num_results=10):
    # 🔹 API Keys & Database Config
    API_KEY =  'AIzaSyB3zrWrFVnUJC9HwOTv9neDr_9RmsIX7Ec' # "YOUR_GOOGLE_API_KEY"
    SEARCH_ENGINE_ID = '623ed3882c61b4b8e' # YOUR_SEARCH_ENGINE_ID
    DB_CONNECTION_STR = "Driver={ODBC Driver 18 for SQL Server};Server=tcp:vifinancenews.database.windows.net,1433;Database=Vietnam_Finance_News;Uid=ViFinanceNews;Pwd={ViFinanceNew#2025};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
                     # "Driver={ODBC Driver 18 for SQL Server};Server=tcp:yourserver.database.windows.net,1433;Database=yourdb;Uid=username;Pwd=password;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"

    if not API_KEY or not SEARCH_ENGINE_ID or not DB_CONNECTION_STR:
        print("❌ Missing API keys or database connection string. Set them as environment variables.")
        return
    
    # 🔹 Initialize Components
    try:
        search_engine = SearchEngine(API_KEY, SEARCH_ENGINE_ID)
        database = Database(DB_CONNECTION_STR)
        database.connect()
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # 🔹 Search for news articles
    try:
        articles = search_engine.search(query, num=num_results)
    except Exception as e:
        print(f"❌ Search failed: {e}")
        database.close()
        return

    table_name = "articles"
    schema_name = "dbo"
    tc = TitleCompleter()
    # 🔹 Scrape & Store in Database

    # ✅ Check table existence ONCE
    if not database.check_table_exists(schema_name=schema_name, table_name=table_name):
        schema = {
            "id": "INT IDENTITY(1,1) PRIMARY KEY",
            "author": "NVARCHAR(500) DEFAULT 'Unknown'",
            "title": "NVARCHAR(500) DEFAULT 'No title available'",
            "date_publish": "DATETIME NULL",  
            "description": "NVARCHAR(MAX) DEFAULT 'No description available'",
            "main_text": "NVARCHAR(MAX) DEFAULT 'No content available'",
            "image_url": "NVARCHAR(1000) DEFAULT 'No image available'",
            "url": "NVARCHAR(1000) UNIQUE NOT NULL",  
            "date_download": "DATETIME NULL"  
        }
        database.create_table(table_name=table_name, schema=schema)

    # ✅ Filter out articles BEFORE processing
    valid_articles = []
    for article in articles:
        try:
            if not article.get("link"):
                continue  # Skip articles with no links
            
            original_title = article["title"]
            # Scrape article content
            article_data = ArticleScraper.scrape_article(article["link"])
            
            if (
                not article_data["main_text"]
                or article_data["main_text"] == "No content available"
                or article_data["author"] == "Unknown"
            ):
                continue  # Skip invalid articles
            
            # ✅ Complete title AFTER filtering
            article_data["title"] = tc.complete_title(original_title=original_title, article=article)
            
            pprint.pprint(article_data)
            valid_articles.append(article_data)

        except Exception as e:
            print(f"❌ Error processing article {article['link']}: {e}")

    # ✅ Batch insert all valid articles
    if valid_articles:
        database.insert_records_bulk(table_name=table_name, records=valid_articles)  # Bulk insert
    
    # 🔹 Close the database connection
    database.close()
    return


if __name__ == "__main__":
    main()