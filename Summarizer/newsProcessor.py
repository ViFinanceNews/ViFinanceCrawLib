import os
from dotenv import load_dotenv
import google.generativeai as genai
from newsplease import NewsPlease
import hashlib
import uuid

# Load environment variables from .env file
load_dotenv()

# Configure your key
api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("No API_KEY found in environment variables")

genai.configure(api_key=api_key)

class NewsProcessor:
    def __init__(self, article):
        self.article = article
        self.model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')
        self.id = str(uuid.UUID(hashlib.md5(article.url.encode()).hexdigest()))
        self.tags_id = []

    def generate_tags(self):
        """
        Generates tags for the abstract using the Gemini LLM.

        Returns:
        - List of dicts with tag IDs and names for the Tags table.
        """
        abstract = self.article.description
        if not abstract:
            abstract = self.summarize()

        prompt = f"""
        Generate 1-4 tags for the following news article abstract. The tags should be relevant to the content. 
        Generate only tags and nothing else, expected format: tag1, tag2, tag3, tag4.
        
        Abstract:
        {abstract}

        Tags:
        """

        response = self.model.generate_content(prompt)
        tags = response.text.strip().split(',')

        for tag in tags:
            tag_id = str(uuid.UUID(hashlib.md5(tag.encode()).hexdigest()))
            self.tags_id.append({"tag_id": tag_id, "tag_name": tag.strip()})
        return self.tags_id

    def summarize(self):
        """
        Generates a summarized article using the Gemini LLM.

        Returns:
        - An abstract of the article.
        """
        main_text = self.article.maintext
        if not main_text:
            return "No main text available."

        prompt = f"""
        Generate a 1-3 sentence summary for the following article. The summary should be concise and capture the main points of the article.
        Expected format: abstract.
        
        Main Text:
        {main_text}
        """

        response = self.model.generate_content(prompt)
        return response.text.strip()

    def transform_article(self):
        """
        Prepares data for the Articles table.
        """
        # Handle authors as a single string (assuming one author or joining multiple)
        authors = ", ".join(self.article.get("authors", [])) if self.article.get("authors") else "Unknown Author"
        
        transformed = {
            "article_id": self.id,
            "author": authors[:255],  # Truncate to fit nvarchar(255)
            "title": self.article.get("title", "Unknown Title")[:255],  # Truncate to fit nvarchar(255)
            "article_link": self.article.get("url", "Unknown URL")[:500],  # Truncate to fit nvarchar(500)
            "image_url": self.article.get("image_url", "")[:500],  # Truncate to fit nvarchar(500), default to empty if not present
            "published_at": self.article.get("date_publish", None)  # None if not available; ensure it's a datetime
        }
        return transformed

    def get_tags(self):
        """
        Prepares data for the Tags table.
        Returns a list of dicts with tag_id and tag_name.
        """
        return self.tags_id

    def get_article_tags(self):
        """
        Prepares data for the ArticleTags junction table.
        Returns a list of dicts with article_id and tag_id pairs.
        """
        return [
            {"article_id": self.id, "tag_id": tag["tag_id"]}
            for tag in self.tags_id
        ]

if __name__ == "__main__":
    # Example usage
    article = NewsPlease.from_url('https://vietnamnet.vn/my-bieu-tinh-phan-doi-elon-musk-len-cao-dai-ly-xe-dien-tesla-bi-dap-pha-2379517.html')
    processor = NewsProcessor(article)

    # Generate tags
    tags = processor.generate_tags()
    print("Tags for Tags Table:", tags)

    # Get article data
    article_data = processor.transform_article()
    print("Article Data for Articles Table:", article_data)

    # Get article-tag relationships
    article_tags = processor.get_article_tags()
    print("Article-Tag Relationships for ArticleTags Table:", article_tags)
