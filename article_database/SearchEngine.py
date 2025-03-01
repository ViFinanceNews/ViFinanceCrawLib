import httpx

# HOW TO GET THE FREE-SEARCH API from GOOGLE: https://www.youtube.com/watch?v=4YhxXRPKI0c&ab_channel=JieJenn
class SearchEngine:
    def __init__(self, api_key, search_engine_id, base_url="https://www.googleapis.com/customsearch/v1"):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = base_url
   
    def search(self, query, language="nil", num=10, exact_match=False, **extra_params):
        """
        Searches for news articles using Google's Custom Search API.

        Parameters:
        - query (str): Search keyword.
        - language (str, optional): Language filter (default: English).
        - num (int, optional): Number of results (1-10).
        - exact_match (bool, optional): If True, searches for the exact phrase.
        - extra_params (dict): Additional API parameters.

        Returns:
        - List of article links.
        """
        if exact_match:
            query = f'"{query}"'  

        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': num,
            'lr': language,
            **extra_params  # Merge additional parameters dynamically
        }

        try:
            response = httpx.get(self.base_url, params=params)
            response.raise_for_status()
            items = response.json().get("items", [])

            return [{"title": item["title"], "link": item["link"]} for item in items]

        except httpx.HTTPStatusError as e:
            print(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            print(f"Error: {e}")
        
        return []