import flask
from ViFinanceCrawLib.article_database.ScrapeAndTagArticles import ScrapeAndTagArticles
from flask import request, jsonify

app = flask.Flask(__name__)
scrapped_url = []

@app.route('/retrive article from websites and display', methods=['POST'])
def get_articles():
    data = request.get_json()
    if not data :
        return jsonify({"error": "Invalid input"}), 400

    search_query = data
    processor = ScrapeAndTagArticles()

    scrapped_url.append(processor.search_and_scrape(search_query))
    return jsonify({"message": "success"}), 200

# If user favorites the article, move it from Redis to the database using its URL as key
@app.route('/move to database', methods=['POST'])
def move_to_database():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Invalid input, 'url' is required"}), 400

    processor = ScrapeAndTagArticles()
    for url in scrapped_url:
        if scrapped_url==url:
            processor.move_to_database(url)

    return jsonify({"message": "success"}), 200

#testrun
if __name__ == '__main__':
    app.run(debug=True)