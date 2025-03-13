import os
from dotenv import load_dotenv
import google.generativeai as genai
from newsplease import NewsPlease

# Load environment variables from .env file
load_dotenv("news.env")

api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("No API_KEY found in environment variables")

genai.configure(api_key=api_key)

class NewsProcessor:
    def __init__(self, article):
        self.article = article
        self.model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')

    def generate_tags(self):
        """
        Generates tags for the abstract using the Gemini LLM.

        Returns:
        - List of tags.
        """
        abstract = self.article.description
        if not abstract:
            return "No abstract available."

        prompt = f"""
        Generate 1-4 tags for the following news article abstract. The tags should be relevant to the content. 
        Generate only tags and nothing else, expected format: tag1, tag2, tag3, tag4.
        
        Abstract:
        {abstract}

        Tags:
        """

        response = self.model.generate_content(prompt)
        return response.text.strip().split(',')

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

if __name__ == "__main__":
    # Example usage
    article = NewsPlease.from_url('https://vietnamnet.vn/my-bieu-tinh-phan-doi-elon-musk-len-cao-dai-ly-xe-dien-tesla-bi-dap-pha-2379517.html')
    processor = NewsProcessor(article)

    tags = processor.generate_tags()
    print("Tags:", tags)

    summary = processor.summarize()
    print("Summary:", summary)
