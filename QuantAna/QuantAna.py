"""
    This is the QuantAna module (Quantitative Analysis) module
    The Problem that QuantAna trying to is to provide the quantitative analysis
    on the article content
    Including: Sentimental Analysis - Toxicity Detection 

"""

from article_database.TextCleaning import TextCleaning as tc
from transformers import pipeline
import torch
from detoxify import Detoxify
from sentence_transformers import SentenceTransformer, util
class QuantAnaIns:

    def __init__(self):
        self.sentence_transformer_name= "keepitreal/vietnamese-sbert"
        self.sentence_transformer_model = SentenceTransformer(self.sentence_transformer_name)
        self.sentiment_model_name = "tabularisai/multilingual-sentiment-analysis"
        self.sentiment_pipeline = "text-classification"
        self.sentimement_model = pipeline(self.sentiment_pipeline, model=self.sentiment_model_name)

    def compute_semantic_similarity(self, article1, article2):
        """Calculate Semantic Similarity between 2 articles and Source
            article1 & article2 : str (pure-string)
        """
        emb_query = self.sentence_transformer_model.encode(article1,convert_to_tensor=True)
        emb_source = self.sentence_transformer_model.encode(article2, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(emb_query, emb_source).item()
        return similarity_score
    
    def sentiment_analysis(self, article_text):
        "Detecting the sentiment in the article & measure how strong it's"
        sentiment_result = self.sentimement_model(article_text)
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
    
