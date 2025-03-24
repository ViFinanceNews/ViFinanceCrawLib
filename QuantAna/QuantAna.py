"""
    This is the QuantAna module (Quantitative Analysis) module
    The Problem that QuantAna trying to is to provide the quantitative analysis
    on the article content
    Including: Sentimental Analysis - Toxicity Detection 

"""
from sagemaker.huggingface import HuggingFaceModel
from ViFinanceCrawLib.article_database.TextCleaning import TextCleaning as tc
from transformers import pipeline
import torch
from detoxify import Detoxify
from sentence_transformers import util
import boto3
import sagemaker
import json
import numpy as np
import pandas as pd
class QuantAnaIns:

    def __init__(self):
        self.endpoint_name = 'sentence-feature-extract'
        self.runtime = boto3.client('sagemaker-runtime')
        self.HF_MODEL_ID = "Fsoft-AIC/videberta-base"
        self.sm_client = boto3.client('sagemaker')
        try:
            self.role = sagemaker.get_execution_role()
        except ValueError:
            iam = boto3.client('iam')
            self.role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
        # Check if endpoint exists
        endpoint_exists = False
        try:
            response = self.sm_client.describe_endpoint(EndpointName=self.endpoint_name)
            status = response['EndpointStatus']
            if status in ['InService', 'Creating']:
                print(f"Endpoint '{self.endpoint_name}' already exists with status: {status}.")
                endpoint_exists = True
            else:
                print(f"Endpoint exists but status is {status}. Recreating...")
        except self.sm_client.exceptions.ClientError:
            print(f"Endpoint '{self.endpoint_name}' does not exist. Deploying new endpoint.")

        # Only deploy if endpoint does not exist
        if not endpoint_exists:
            self.HF_TASK = 'feature-extraction'
            self.hub = {
                'HF_MODEL_ID': self.HF_MODEL_ID,
                'HF_TASK': self.HF_TASK
            }
            self.huggingface_model = HuggingFaceModel(
                transformers_version='4.37.0',
                pytorch_version='2.1.0',
                py_version='py310',
                env=self.hub,
                role=self.role
            )
            self.model = self.huggingface_model.deploy(
                initial_instance_count=1,
                instance_type='ml.t2.medium',
                endpoint_name=self.endpoint_name
            )
            print(f"New endpoint '{self.endpoint_name}' deployed and ready.")
        else:
            print("Skipping deployment, using existing endpoint.")


        self.sentiment_model_name = "tabularisai/multilingual-sentiment-analysis"
        self.sentiment_pipeline = "text-classification"
        self.sentiment_model = pipeline(self.sentiment_pipeline, model=self.sentiment_model_name)
        print("Successful create QuantAna")

    def compute_semantic_similarity(self, article1, article2):
        """Calculate Semantic Similarity between 2 articles
            article1 & article2: str
        """
        # Batch input
        article_list = [article1, article2]
        
        payload = {
            "inputs": article_list
        }

        try:
            # Single API call for both articles
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            result = json.loads(response['Body'].read())  # result = list of token embeddings

            embeddings = []
            for token_embedding in result:
                token_embedding = np.squeeze(np.array(token_embedding), axis=0)  # Remove batch dimension
                sentence_embedding = np.mean(token_embedding, axis=0)  # Mean pooling
                embeddings.append(sentence_embedding)

            # Convert to torch tensors, ensure shapes are [1, embedding_dim]
            emb_tensor1 = torch.tensor(embeddings[0]).unsqueeze(0)
            emb_tensor2 = torch.tensor(embeddings[1]).unsqueeze(0)
            similarity_score = util.pytorch_cos_sim(emb_tensor1, emb_tensor2).item()
            return similarity_score

        except Exception as e:
            print(f"[ERROR] SageMaker Invocation Failed: {str(e)}")
            return None

    def compute_multi_semantic_similarity(self, source_articles, query_article=None):
        """
        Calculate Semantic Similarity:
        - (Optional) Query article vs each source article
        - Pairwise similarity between source articles (intersource)

        Args:
            source_articles (List[str]): List of source article strings
            query_article (str, optional): Query article string. Default is None.

        Returns:
            dict: {
                'query_to_sources': List[float] or None,
                'intersource': List[List[float]]
            }
        """
        try:
            embeddings = []

            # Handle chunking if too many source articles
            chunk_size = 5
            source_chunks = [source_articles[i:i+chunk_size] for i in range(0, len(source_articles), chunk_size)]

            # If query exists, first call includes query
            for idx, chunk in enumerate(source_chunks):
                if idx == 0 and query_article:
                    input_batch = [query_article] + chunk
                else:
                    input_batch = chunk

                payload = {
                    "inputs": input_batch
                }

                # API call
                response = self.runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps(payload)
                )
                result = json.loads(response['Body'].read())  # token embeddings list

                # Pooling embeddings
                for token_embedding in result:
                    token_embedding = np.squeeze(np.array(token_embedding), axis=0)  # Remove batch dim
                    sentence_embedding = np.mean(token_embedding, axis=0)  # Mean pooling
                    embeddings.append(sentence_embedding)

            # Convert to torch tensors
            embedding_tensors = [torch.tensor(e).unsqueeze(0) for e in embeddings]

            # Process similarity scores
            query_to_sources = None
            intersource_start_idx = 0

            if query_article:
                query_tensor = embedding_tensors[0]
                source_tensors = embedding_tensors[1:]
                query_to_sources = [
                    util.pytorch_cos_sim(query_tensor, src).item()
                    for src in source_tensors
                ]
                intersource_start_idx = 1  # Skip query in intersource

            # Pairwise intersource similarity
            source_tensors = embedding_tensors[intersource_start_idx:]
            intersource = []
            for i, src1 in enumerate(source_tensors):
                row = []
                for j, src2 in enumerate(source_tensors):
                    score = util.pytorch_cos_sim(src1, src2).item()
                    row.append(score)
                intersource.append(row)

            if query_article:
                # === 1. Query-to-Source Similarity ===
                query_df = pd.DataFrame({
                    'Source': [f'Source_{i+1}' for i in range(len(query_to_sources))],
                    'Matching_to_Query': query_to_sources
                })

                print("=== Query to Sources Similarity ===")
                print(query_df.round(3))  # Rounded to 3 decimal places
                print("\n")

            # === 2. Intersource Similarity Matrix ===
            labels = [f"Source_{i+1}" for i in range(len(intersource))]
            matrix_df = pd.DataFrame(np.array(intersource), index=labels, columns=labels)

            print("=== Intersource Similarity Matrix ===")
            print(matrix_df.round(3))
            return {
                'query_to_sources': query_to_sources,
                'intersource': intersource
            }

        except Exception as e:
            print(f"[ERROR] SageMaker Invocation Failed: {str(e)}")
            return None

    
    def sentiment_analysis(self, article_text):
        "Detecting the sentiment in the article & measure how strong it's"
        sentiment_result = self.sentiment_model(article_text)
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
    
    def terminate(self):
        # Terminate this - but this would stop the end-point (notice this be-careful)
        try:
            self.sm_client.delete_endpoint(EndpointName='sentence-feature-extract')
            print("Endpoint deleted.")
        except self.sm_client.excteameptions.ClientError as e:
            print("Endpoint deletion skipped (maybe doesn't exist):", e)

        # Delete Endpoint Config
        try:
            self.sm_client.delete_endpoint_config(EndpointConfigName='sentence-feature-extract')
            print("Endpoint config deleted.")
        except self.sm_client.exceptions.ClientError as e:
            print("Endpoint config deletion skipped:", e)
        


    
