<<<<<<< HEAD
# ViFinanceCrawLib

ViFinanceCrawLib is a Python-based library designed for financial data extraction, processing, and title completion. It provides tools for scraping financial data, managing database interactions, and completing incomplete titles using predictive techniques.

## Features
- **Web Scraper**: Extracts financial data from various online sources.
- **Database Handler**: Manages data storage and retrieval efficiently.
- **Title Completer**: Enhances financial dataset consistency by completing missing or incomplete titles.

## Project Structure
```
ViFinanceCrawLib/
│-- Articlescraper.py        # Handles data extraction from financial sources
│-- Database.py       # Manages database operations (CRUD)
│-- TitleCompleter.py # Completes missing financial titles using predictive methods
│-- README.md         # Project documentation
│-- test_lib.py       # client program in testing the lib
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/DTJ-Tran/ViFinanceCrawLib.git
   cd ViFinanceCrawLib
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### **1. Scraper**
Run the scraper to fetch financial data:
```sh
python scraper.py
```

### **2. Database Operations**
To store or retrieve data, use:
```sh
python database.py
```

### **3. Title Completion**
Fill missing titles in your dataset:
```sh
python title_completer.py
```

## Contributing
Feel free to fork this repository and submit pull requests to improve functionality!

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For issues and contributions, reach out via [GitHub Issues](https://github.com/DTJ-Tran/ViFinanceCrawLib/issues).

=======
Here’s a README.md file for your ViFinanceNews Analysis Tool:

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


	3.	Set up the database:

        python setup_database.py

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
>>>>>>> 698b2c1 (Initial commit)
