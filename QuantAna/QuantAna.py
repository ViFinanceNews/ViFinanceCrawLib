"""
    This is the QuantAna module (Quantitative Analysis) module
    The Problem that QuantAna trying to is to provide the quantitative analysis
    on the article content
    Including: Sentimental Analysis - Toxicity Detection 

"""
import sys
from sagemaker.huggingface import HuggingFaceModel
from ViFinanceCrawLib.article_database.TextCleaning import TextCleaning as tc
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
import re

class QuantAnaIns:
    def __init__(self):
        load_dotenv()
        genai.configure(api_key=os.getenv("API_KEY"))

        self.model_name = 'gemini-2.0-flash-lite'
        self.translator_model = genai.GenerativeModel(self.model_name)
        self.endpoint_name = 'sentence-feature-extract'
        self.sentiment_endpoint_name = 'multi-ling-sentiment-analysis'
        self.runtime = boto3.client('sagemaker-runtime')
        self.sm_client = boto3.client('sagemaker')
        self.HF_MODEL_ID_FEATURE_EXTRACT = "Fsoft-AIC/videberta-base"
        self.HF_TASK_FEATURE = "feature-extraction"
        self.HF_MODEL_ID_SENTIMENT = "tabularisai/multilingual-sentiment-analysis"
        self.HF_TASK_SENTIMENT = 'text-classification'
        self._set_up_vncorenlp()
        self._set_up_sagemaker_role()

        # Deploy the models if they don't exist
        self._deploy_endpoint(self.endpoint_name, self.HF_MODEL_ID_FEATURE_EXTRACT, self.HF_TASK_FEATURE)
        self._deploy_endpoint(self.sentiment_endpoint_name, self.HF_MODEL_ID_SENTIMENT, self.HF_TASK_SENTIMENT)

    def _set_up_vncorenlp(self):
        filename = "VnCoreNLP/VnCoreNLP-1.1.1.jar"
        file_path = Path.cwd() / filename
        if not file_path.exists():
            print(f"Error: Required file '{filename}' not found in the current directory.")
            sys.exit(1)
        self.rdrsegmenter = VnCoreNLP(str(file_path), annotators="wseg", max_heap_size='-Xmx500m')

    def _set_up_sagemaker_role(self):
        try:
            self.role = sagemaker.get_execution_role()
        except ValueError:
            iam = boto3.client('iam')
            self.role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

    def _check_endpoint_exists(self, endpoint_name):
        try:
            response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            if status in ['InService', 'Creating']:
                print(f"Endpoint '{endpoint_name}' already exists with status: {status}.")
                return True
            else:
                print(f"Endpoint exists but status is {status}. Recreating...")
                return False
        except self.sm_client.exceptions.ClientError:
            print(f"Endpoint '{endpoint_name}' does not exist. Deploying new endpoint.")
            return False

    def _check_model_exists(self, endpoint_name):
        try:
            response = self.sm_client.list_models()
            model_names = [model["ModelName"] for model in response.get("Models", [])]
            if endpoint_name in model_names:
                return True
            else:
                print(f"Model '{endpoint_name}' not found in SageMaker.")
                return False
        except self.sm_client.exceptions.ClientError:
            print(f"Failed to retrieve models '{endpoint_name}' from SageMaker. Creating new model.")
            return False

    def _deploy_endpoint(self, endpoint_name, model_id, task):
        endpoint_exists = self._check_endpoint_exists(endpoint_name)
        if not endpoint_exists:
            model_exists = self._check_model_exists(endpoint_name)
            if not model_exists:
                hub = {
                    'HF_MODEL_ID': model_id,
                    'HF_TASK': task
                }
                huggingface_model = HuggingFaceModel(
                    transformers_version='4.37.0',
                    pytorch_version='2.1.0',
                    py_version='py310',
                    env=hub,
                    role=self.role,
                    name=endpoint_name
                )
                self.model = huggingface_model.deploy(
                    initial_instance_count=1,
                    instance_type='ml.t2.medium',
                    endpoint_name=endpoint_name
                )
                print(f"New endpoint '{endpoint_name}' deployed and ready.")
            else:
                print(f"Model '{endpoint_name}' exists - deploying new endpoint.")
        else:
            print(f"Skipping deployment, using existing endpoint '{endpoint_name}'.")

        print("QuantAna created successfully.")

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
    
    def generative_extractive(self, article_text):
        prompt = f"""
        Bạn là một chuyên gia tóm tắt trích xuất. Hãy **trích nguyên văn 5 câu quan trọng nhất** từ bài viết sau để nắm bắt nội dung cốt lõi và trình bày dưới dạng **một đoạn văn súc tích**.

        ### **Yêu cầu:**  
        - **Phải trích nguyên văn** từ bài viết, **không viết lại, không diễn giải**.  
        - Các câu phải **đủ ý, quan trọng, và có ý nghĩa độc lập**.  
        - Sắp xếp các câu một cách hợp lý để đảm bảo mạch nội dung tự nhiên.  
        - Nếu bài viết có ít hơn 10 câu, hãy trích xuất toàn bộ các câu quan trọng nhất có thể.  
        - **Không thêm bất kỳ phản hồi hoặc nội dung thừa thãi nào**, chỉ xuất ra đoạn văn chứa các câu trích xuất.

        ### **Bài viết:**  
        {article_text}

        ### **Định dạng đầu ra (một đoạn văn chứa 10 câu nguyên văn):**  
        "[Câu 1] [Câu 2] [Câu 3] … [Câu 10]"
        """

        try:
            response = self.translator_model.generate_content(prompt)
            if not getattr(response, "text", None):
                print("⚠️ Warning: Empty response from AI model.")
                return ""

            return response.text

        except Exception as e:
            print(f"❌ Error in generative_extractive: {e}")
            return ""

    def sentiment_analysis(self, article_text):
        article_text = re.sub(r"\[Câu \d+\]\s*", "", article_text)
        print(article_text)
        payload = {
            "inputs": article_text
        }
        try:
            response = self.runtime.invoke_endpoint(
                EndpointName = self.sentiment_endpoint_name,
                ContentType= 'application/json',
                Body=json.dumps(payload)
            )
            sentiment_result = json.loads(response['Body'].read())[0]  # result = list of token embeddings
            sentiment_label = sentiment_result['label']
            sentiment_score = sentiment_result['score']
            return {
                "sentiment_label": sentiment_label,  # NEG: Tiêu cực, POS: Tích cực, NEU: Trung tính
                "sentiment_score": sentiment_score
                }
        except Exception as e:
            print(f"[ERROR] SageMaker Invocation Failed: {str(e)}")
            return None
       
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

    def normalize_result(self, value):
        # Check if the value is a numpy float type
        if isinstance(value, np.float32) or isinstance(value, np.float64):
            return float(value) * 100  # Return the score
        else:
            return float(value) * 100  # Return the raw value as a string

    def detect_toxicity(self, article_text: str):
        """Detects toxicity and misinformation in the article.
           Parameter: article_text (Vietnamese String need to be pre-process and segmentized)
           Returns:
            A dictionary with the format: (the printed out result would be translate to Vietnamese)
                {"Toxicity: Score", "Insult": Score, "Threat": Score, "Identity Attack" : Score, "Obscene" :Score}
        """
        try:
            tokenized_text = self.rdrsegmenter.tokenize(article_text)
            pre_processed_sentences = self.combine_tokens(tokenized_text[0])
            translation = self.translation_from_Vie_to_Eng(pre_processed_sentences)
            toxicity_score = Detoxify("multilingual").predict(translation)

            return {
                "Tính Độc Hại": self.normalize_result(toxicity_score["toxicity"]),
                "Tính Xúc Phạm": self.normalize_result(toxicity_score["insult"]),
                "Tính Đe Doạ": self.normalize_result(toxicity_score["threat"]),
                "Công kích danh tính": self.normalize_result(toxicity_score["identity_attack"]),
                "Mức Độ Thô Tục": self.normalize_result(toxicity_score["obscene"])
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
            self.sm_client.delete_endpoint(EndpointName=self.endpoint_name)
            self.sm_client.delete_endpoint(EndpointName=self.sentiment_endpoint_name)
            print("Endpoint deleted.")
        except self.sm_client.excteameptions.ClientError as e:
            print("Endpoint deletion skipped (maybe doesn't exist):", e)

        # Delete Endpoint Config
        try:
            self.sm_client.delete_endpoint_config(EndpointConfigName=self.endpoint_name)
            self.sm_client.delete_endpoint(EndpointName=self.sentiment_endpoint_name)
            print("Endpoint config deleted.")
        except self.sm_client.exceptions.ClientError as e:
            print("Endpoint config deletion skipped:", e)
        


    
