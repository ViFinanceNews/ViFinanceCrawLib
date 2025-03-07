# ViFinanceNews Analysis Tool  

ViFinanceNews Analysis Tool is an advanced financial news analysis system that automates the collection, storage, and evaluation of financial news articles. It consists of three core modules designed for comprehensive qualitative and quantitative analysis.

## Features  

### 1️⃣ Article Database 📊  
- Web-scraping module that collects financial news from various sources.  
- Stores and organizes scraped data into an SQL database for easy retrieval.  

### 2️⃣ QualAna (Qualitative Analysis) 📝  
- Analyzes news articles from a qualitative perspective.  
- Extracts sentiment, tone, key themes, and potential biases.  
- Assesses the credibility and impact of financial news.  

### 3️⃣ QuantAna (Quantitative Analysis) 📈  
- Uses NLP-based quantitative metrics to evaluate financial news.  
- **Semantic Similarity Score**: Compares articles for redundancy and consistency.  
- **Directness**: Measures how explicitly financial insights are conveyed.  
- **Relatability**: Evaluates the relevance of articles to financial topics.  

## Installation  

1. Clone this repository:  
   ```sh
   git clone https://github.com/your-repo/ViFinanceNews.git
   cd ViFinanceNews

	2.	Install dependencies:

        pip install -r requirements.txt


	3.	Using the Library by cd to the project

    4. Getting the key:
        API_KEY (from Google Cloud Console) 
        
        SEARCH_ENGINE_ID (Google Search Engine)
        
        SEARCH_API_KEY (from Google Cloud Console)


Usage

    Running the script
    ```
        python QualAna/main.py
    ```

License

This project is licensed under the MIT License.

🚀 ViFinanceNews Analysis Tool helps investors, analysts, and researchers make data-driven decisions by providing a comprehensive view of financial news especially for Vietnamese reader.

