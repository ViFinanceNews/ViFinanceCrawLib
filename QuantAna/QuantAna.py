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
from vncorenlp.vncorenlp import VnCoreNLP
from dotenv import load_dotenv
import google.generativeai as genai
import os
from pathlib import Path
import sys
class QuantAnaIns:

    def __init__(self):
        load_dotenv()
        genai.configure(api_key=os.getenv("API_KEY"))
        model_name='gemini-2.0-flash-thinking-exp-01-21'
        self.translator_model = genai.GenerativeModel(model_name)
        self.endpoint_name = 'sentence-feature-extract'
        self.runtime = boto3.client('sagemaker-runtime')
        self.HF_MODEL_ID = "Fsoft-AIC/videberta-base"
        self.sm_client = boto3.client('sagemaker')
        filename = "VnCoreNLP-1.1.1.jar"
        file_path = Path.cwd() /filename
        if not file_path.exists():
            print(f"Error: Required file '{filename}' not found in the current directory.")
            sys.exit(1)  # Stop execution if the file is missing
        self.rdrsegmenter = VnCoreNLP(file_path, annotators="wseg",  max_heap_size='-Xmx500m')
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

        
        # Check if the model exists in SageMaker
        model_exists = False
        try:
            response = self.sm_client.list_models()
            # Extract all model names
            if response["Models"]:
                model_names = [model["ModelName"] for model in response["Models"]]
                if self.endpoint_name in model_names:
                    model_exists = True
            else:
                print(f"Model '{self.endpoint_name}' not found in SageMaker.")
        except self.sm_client.exceptions.ClientError:
                print(f"Failed to retrieve models '{self.endpoint_name}' from SageMaker. - Create new model")
        

        # Only deploy the if endpoint or model not exist
        if (not endpoint_exists):
                if not model_exists:
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
                        role=self.role,
                        name= self.endpoint_name
                    )
                else:
                    print(f"Model '{self.endpoint_name}' exists. Deploying new endpoint...")
                self.model = self.huggingface_model.deploy(
                    initial_instance_count=1,
                    instance_type='ml.t2.medium',
                    endpoint_name=self.endpoint_name
                )
                print(f"New endpoint '{self.endpoint_name}' deployed and ready.")
        else:
            print("Skipping deployment, using existing endpoint.")
    
        self.sentiment_model_name = "tabularisai/multilingual-sentiment-analysis" # Deploy on Hugging_Face using the same-endpoint with the Semantic Comparision
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

    def compute_multi_semantic_similarity(self, source_articles, query_article=None, display_table=False):
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
            if display_table:
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

    def translation_from_Vie_to_Eng(self, text :str):
        """
        Generate the clear and concise translation from Vietnamese text to English
        for Toxicity Analysis
        
        Parameters:
            article (str): The article content to analyze.

        Returns:
            str: The formatted prompt for LLM analysis.
        """
    
        prompt = f"""
            Bạn là một nhà khoa học và nhà văn, dịch giả song ngữ thông thạo cả tiếng Anh và tiếng Việt. Nhiệm vụ của bạn là dịch văn bản từ **tiếng Việt sang tiếng Anh** một cách chính xác, giữ nguyên giọng điệu, bao gồm cả mức độ trang trọng, cảm xúc, sự thô bạo hoặc độc hại nếu có.

            Yêu cầu:
                1. **Chỉ dịch từ tiếng Việt sang tiếng Anh.** Không dịch theo hướng ngược lại.
                2. **Bảo toàn ý nghĩa và giọng điệu gốc:** Nếu văn bản có sự mỉa mai, châm biếm, trang trọng, hoặc bất kỳ cảm xúc nào, hãy đảm bảo dịch sao cho giữ nguyên sắc thái đó.
                3. **Vì kết quả sẽ qua một bộ phân tích cần chính xác - nên các từ ngữ phân biệt, miệt thị cũng không được nói giảm nói tránh - phải nói thẳng dùng đúng từ gốc như trong từ điển
                3. **Dịch chính xác thuật ngữ chuyên ngành:** Khi gặp thuật ngữ khoa học, công nghệ, AI hoặc ngôn ngữ học, hãy dịch đúng và sử dụng từ ngữ phổ biến trong giới chuyên môn.
                4. **Dịch tự nhiên, không dịch theo kiểu máy móc:** Hãy đảm bảo câu văn trôi chảy, tự nhiên, đúng ngữ pháp và dễ hiểu đối với người bản ngữ.
                5. **Không kiểm duyệt nội dung:** Nếu văn bản có nội dung thô bạo, độc hại hoặc chỉ trích, hãy dịch đúng với giọng điệu gốc thay vì làm nhẹ đi hoặc thay đổi ý nghĩa.
                6. **Chỉ trả về bản dịch tiếng Anh.** Không giải thích, không thêm nội dung ngoài bản dịch.

            Ví dụ:
                • **Tiếng Việt:** "Nghiên cứu này mang tính đột phá, nhưng sự kiêu ngạo của tác giả thì không thể chịu nổi."
                • **Tiếng Anh:** "This research is groundbreaking, but the author’s arrogance is unbearable."

            Hãy dịch văn bản sau đây theo đúng các nguyên tắc trên và chỉ trả về bản dịch tiếng Anh:

            {text}
            """
        
        try:
            response = self.translator_model.generate_content(prompt)
            if not hasattr(response, "text") or not response.text:
                print("⚠️ Warning: Empty response from AI model.")
                return []

            analysis = response.text
            return analysis
        
        except Exception as e:
            print(f"❌ Error in generate_search_queries: {e}")
            return []

    def combine_tokens(self, tokens):
        """ 
        Combines a list of segmented tokens into a single string.

        This method is specifically designed to handle tokens generated by VnCoreNLP segmentation (By Dat,et al 2018) , 
        where words are often segmented into smaller units with underscores ("_") to represent morphemes 
        or sub-word units. It joins the list of tokens into a single string with spaces separating each token 
        and removes any underscores from the resulting string to reconstruct the original word form.

        Parameters:
        tokens (list): A list of tokens (strings) obtained from VnCoreNLP segmentation, 
                        which may include underscores between sub-word units.

        Returns:
        str: A single string formed by joining the tokens with spaces, 
            with all underscores removed, effectively merging the segmented words.

        Example:
        >>> obj.combine_tokens(["hello", "world_example"])
        'hello worldexample'
        
        This method is useful for reconstructing the text after segmentation when processing text with VnCoreNLP.

        Original Link to the Segmentation Library:
        https://github.com/vncorenlp/VnCoreNLP.git 
        """
        return " ".join(tokens).replace("_", "")

    def detect_toxicity(self, article_text: str):
        """Detects toxicity and misinformation in the article.
           Parameter: article_text (Vietnamese String need to be pre-process and segmentized)
           Returns:
            A dictionary with the format: (the printed out result would be translate to Vietnamese)
                {"Toxicity: Score", "Insult": Score, "Threat": Score, "Identity Attack" : Score, "Obscene" :Score}
        """
        try:
            tokenized_text = self.rdrsegmenter.tokenize(article_text)
            pre_processed_sentences = self.combine_tokens(tokenized_text)
            translation = self.translation_from_Vie_to_Eng(pre_processed_sentences)
            toxicity_score = Detoxify("multilingual").predict(article_text)

            return {
                "Tính Độc Hại": toxicity_score["toxicity"],
                "Tính Xúc Phạm": toxicity_score["insult"],
                "Tính Đe Doạ": toxicity_score["threat"],
                "Công kích danh tính": toxicity_score["identity_attack"],
                "Mức Độ Thô Tục":  toxicity_score["obscene"]
            }
        except Exception as e:
            # Log the exception with details for debugging
            print(f"An error occurred: {e}")
            # Optionally log more details, such as the input and the stack trace
            import traceback
            traceback.print_exc()

            # Return a response or a default value in case of error
            return {
                "error": "An error occurred while detecting toxicity",
                "details": str(e)
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
        


    
