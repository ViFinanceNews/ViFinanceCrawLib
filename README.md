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

