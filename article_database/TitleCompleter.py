import re
import requests
from bs4 import BeautifulSoup

class TitleCompleter:
        
    def complete_title(self, original_title, article):
        """
        Attempts to reconstruct a cut-off title using context clues.
        
        Parameters:
        - original_title (str): The possibly truncated title.
        - article (dict): A dictionary containing 'title' and 'link'.

        Returns:
        - str: The best possible full title.
        """
        # Step 1: Check if the title is truncated
        if "..." not in original_title:
            return original_title  # No need to fix it
        
        # Step 2: Try to extract from URL
        url_title = article["link"].split("/")[-1].replace("-", " ")
        url_title = re.sub(r'\.htm.*$', '', url_title)  # Remove file extensions
        url_title = url_title.strip().capitalize()

        # Step 3: Scrape the webpage for a possible full title
        try:
            response = requests.get(article["link"], timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            scraped_title = soup.title.string if soup.title else ""

            if scraped_title:
                return scraped_title  # Use the full title from the page
        except Exception as e:
            print(f"Warning: Could not scrape the article. Error: {e}")

        # Step 4: If all else fails, combine the best guess
        reconstructed_title = f"{original_title[:-3]} {url_title}"
        return reconstructed_title

# if __name__ == '__main__':
    # tc = TitleCompleter()
    # # # Example Usage
    # # original_title = "Tiếp tục sử dụng các công cụ bình ổn giá xăng dầu - Nhịp sống kinh ..."
    # # article = {
    # #     "title": "Tiếp tục sử dụng các công cụ bình ổn giá xăng dầu - Nhịp sống kinh ...",
    # #     "link": "https://vneconomy.vn/tiep-tuc-su-dung-cac-cong-cu-binh-on-gia-xang-dau.htm"
    # # }

    # # fixed_title = tc.complete_title(original_title, article)
    # # print(f"🔹 Fixed Title: {fixed_title}")