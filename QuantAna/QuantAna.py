"""
    This is the QuantAna module (Quantitative Analysis) module
    The Problem that QuantAna trying to is to provide the quantitative analysis
    on the article content
    Including: Sentimental Analysis - Toxicity Detection 

"""

from article_database.TextCleaning import TextCleaning as tc
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from detoxify import Detoxify
from sentence_transformers import util
class QuantAna:

    def __init__(self):
        print("QuantAna initialized")
        self.sentence_transformer_model = "keepitreal/vietnamese-sbert"
        self.sentiment_model = "wonrax/phobert-base-vietnamese-sentiment"
        self.sentiment_pipeline = "text-classification"

    def compute_semantic_similarity(self, article1, article2):
        """Calculate Semantic Similarity between 2 articles and Source
            article1 & article2 : str (pure-string)
        """
        emb_query = self.sentence_transformer_model.encode(article1, convert_to_tensor=True)
        emb_source = self.sentence_transformer_model.encode(article2, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(emb_query, emb_source).item()
        return similarity_score
    
    def sentiment_analysis(self, article_text):
        "Detecting the sentiment in the article & measure how strong it's"
        sentiment_result = sentiment_model(article_text)
        sentiment_label = sentiment_result[0]['label']
        sentiment_score = sentiment_result[0]['score']
        return {
            "sentiment_label": sentiment_label,  # NEG: Tiêu cực, POS: Tích cực, NEU: Trung tính
            "sentiment_score": sentiment_score
        }

    def detect_toxicity(self, article_text):
        """Detects toxicity and misinformation in the article."""
        toxicity_score = Detoxify("multilingual").predict(article_text)

        return {
            "Tính Độc Hại": toxicity_score["toxicity"],
            "Tính Xúc Phạm": toxicity_score["insult"],
            "Tính Đe Doạ": toxicity_score["threat"],
            "Công kích danh tính": toxicity_score["identity_attack"],
            "Mức Độ Thô Tục":  toxicity_score["obscene"]
        }
    
if __name__ == "__main__":
    quantAna = QuantAna()
    testStr =  "Người nhập cư đang làm trầm trọng hơn tình hình kinh tế."
    toxicity_res = quantAna.detect_toxicity(testStr)
    print(toxicity_res)
