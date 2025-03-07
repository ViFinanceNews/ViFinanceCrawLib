"""
 This module include a list of method & object of Utility for
 supporting the Article Fact Check
"""

import google.generativeai as genai
import os
import requests
from bs4 import BeautifulSoup  
from article_database.ArticleScraper import ArticleScraper
from article_database.SearchEngine import SearchEngine
from article_database.TitleCompleter import TitleCompleter

class ArticleFactCheckUtility():

    def __init__(self, model_name='gemini-2.0-pro-exp-02-05'):
        genai.configure(api_key=os.getenv("API_KEY"))
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.tc = TitleCompleter()
        self.search_engine  =  SearchEngine(os.getenv("SEARCH_API_KEY"), os.getenv("SEARCH_ENGINE_ID"))
        self.article_scraper = ArticleScraper()
        return
        
    def generate_search_queries(self, statement):
        prompt = f"""Identify keywords and concepts from the following statement. Using this claim, generate a neutral, thought-provoking question that encourages discussion without assuming a particular stance. The question must be between 10 and 50 words long. Return only the generated question with no other surrounding text, and you MUST write it in Vietnamese.

        Statement: {statement}
        """
        response = self.model.generate_content(prompt)
        claims = response.text.split('\n')
        claims = [claim.strip() for claim in claims if claim.strip()]  # Clean up
        return claims

    def search_web(self, query, num_results=5):
        tc = self.tc
        searchEngine = self.search_engine
        articles = searchEngine.search(query, num=num_results)
        article_scraper = self.article_scraper
        valid_articles = []
        for article in articles:
            try:
                if not article.get("link"):
                    continue  # Skip articles with no links

                original_title = article["title"]
                # Scrape article content
                article_data = article_scraper.scrape_article(article["link"])

                if (
                    not article_data["main_text"]
                    or article_data["main_text"] == "No content available"
                    or article_data["author"] == "Unknown"
                ):
                    continue  # Skip invalid articles

                # ✅ Complete title AFTER filtering
                article_data["title"] = tc.complete_title(original_title=original_title, article=article)
                valid_articles.append(article_data)

            except Exception as e:
                print(f"❌ Error processing article {article['link']}: {e}")
        return valid_articles
    
    def analyze_evidence(self, statement, evidence):
        """Analyzes evidence to determine if it supports, contradicts, or is neutral to the statement."""
        prompt = f"""Bạn là một trợ lý kiểm tra thông tin. Dưới đây là một thông tin và một số bằng chứng. Hãy xác định xem bằng chứng này **hỗ trợ, mâu thuẫn hay trung lập** đối với thông tin đó. Đưa ra điểm tin cậy (0-100) cho đánh giá của bạn.

        Thông tin: {statement}

        Bằng chứng:
        {evidence}

        Hãy trả lời theo định dạng sau:

        Kết luận: [Hỗ trợ/Mâu thuẫn/Trung lập]
        Mức độ tin cậy: [0-100]
        Giải thích: [Một lời giải thích ngắn gọn về lý do của bạn, có đề cập đến nguồn bằng chứng. Nếu có URL trong bằng chứng, hãy chèn nó vào trong lời giải thích dưới dạng liên kết.]
        Lời khuyên cho người dùng (nếu có): [Một lời khuyên ngắn gọn]

        Ví dụ cách chèn liên kết:
        - "Bằng chứng từ [nguồn này](URL) cho thấy rằng..."
        - "Theo thông tin từ bài viết này ([link](URL)), ..."
        """

        response = self.model.generate_content(prompt)
        return response.text

    
