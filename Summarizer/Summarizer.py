from dotenv import load_dotenv
import google.generativeai as genai
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
import re
from underthesea import sent_tokenize, word_tokenize, text_normalize
from typing import List
import networkx as nx  # support the the Text-Rank Algorithm
import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel
from transformers import AutoTokenizer
import os
import json
import concurrent.futures
import time
import importlib.resources
from QualAna.ArticleFactCheckUtility import ArticleFactCheckUtility

class Summarizer:
    def __init__(self, stopword_file="vietnamese-stopwords-dash.txt", extractive_model="Fsoft-AIC/videberta-base", 
             task='feature-extraction', instance_type='ml.t2.medium', endpoint_name='sentence-feature-extract', abstractive_model ='gemini-2.0-pro-exp-02-05'):

        # Get the directory of the current file (summarizer.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print("📂 Current dir:", current_dir)
        stopword_path = os.path.join(current_dir, stopword_file)
        # setting up with the extractive model
        self.stopword = set([
            "và", "của", "là", "ở", "trong", "có", "được", "cho", "với", "tại",
            "như", "này", "đó", "một", "các", "những", "để", "vào", "ra", "lên",
            "ngoài_ra", "thường_xuyên", "ngày_càng", "bền_vững", "chiến_lược", "quản_lý",
            "hiệu_quả", "hơn_nữa", "đặc_biệt", "bước", "nhỏ", "không_ngừng", "theo",
            "hay", "hoặc", "từ", "về", "lên", "xuống", "trong", "ngoài", "ra", "vào",
            "được", "đã", "là", "với", "của", "trên", "dưới", "giữa", "sau", "trước",
            "khi", "nếu", "thì", "mà", "nhưng", "vì", "do", "nên", "cũng", "để", "cho"
        ])
        if not os.path.exists(stopword_path):
            print("Not exist the additional_file - use the default set of stopwords")
        else:
            try:
                with open(stopword_path, 'r', encoding='utf-8') as file:
                    file_stopwords = {line.strip() for line in file if line.strip()}
                self.stopword = self.stopword.union(file_stopwords)
            except Exception as e:
                print(f"Error reading stopwords file: {e}. Using default stopwords only.")
        
        self.endpoint_name = endpoint_name
        self.runtime = boto3.client('sagemaker-runtime')
        self.HF_MODEL_ID = extractive_model
        self.sm_client = boto3.client('sagemaker')
        
        # Get Role
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
            self.HF_TASK = task
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
                instance_type=instance_type,
                endpoint_name=self.endpoint_name
            )
            print(f"New endpoint '{self.endpoint_name}' deployed and ready.")
        else:
            print("Skipping deployment, using existing endpoint.")


        # setting up with abstractive model
        load_dotenv(".devcontainer/devcontainer.env")
        genai.configure(api_key=os.getenv("API_KEY"))
        self.abstractive_model = genai.GenerativeModel(abstractive_model)

        self.qa_utility = ArticleFactCheckUtility()
        return

    def pre_processing(self, text):
      text = text_normalize(text)
      text = text.lower()
      # keep the some special token that bring semantic data
      text = re.sub(r'[^\w\s.!?%]', '', text)
      try:
          sentences = sent_tokenize(text)
          sentences = [s.strip() for s in sentences if s.strip()]
      except Exception as e:
          print(f"Error splitting sentences: {e}")
          return []
      processed_sentences = []
      for sentence in sentences:
          try:
              # tokenize
              tokens = word_tokenize(sentence, format="text").split()
              # Combine "number + %""
              new_tokens = []
              i = 0
              while i < len(tokens):
                  # Check if the current_token is number - next is %
                  if i + 1 < len(tokens) and tokens[i].isdigit() and tokens[i + 1] == "%":
                      new_tokens.append(tokens[i] + "%")  
                      i += 2  # Skip the token after the %
                  else:
                      new_tokens.append(tokens[i])
                      i += 1
              # Filter stopwords & normalization
              processed_tokens = []
              for token in new_tokens:
                  if token.endswith("%"):  # if token contain % -> keep still
                      processed_tokens.append(token)
                  elif token not in self.stopword:
                      processed_tokens.append(text_normalize(token)) # Remove stopwords
              # Combined to the %
              processed_sentence = " ".join(processed_tokens)
              processed_sentences.append(processed_sentence)
          except Exception as e:
              print(f"Error processing sentence: {e}")
              continue
      return processed_sentences
    
    def get_embeddings(self, texts: List[str]):
        """Get embeddings for a list of texts using the SageMaker endpoint."""
        embeddings = []

        for text in texts:
           
            payload = {
                "inputs": text
            }
            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            result = json.loads(response['Body'].read())
            token_embeddings = np.array(result[0])
            sentence_embedding = np.mean(token_embeddings, axis=0)  # mean pooling over tokens
            embeddings.append(sentence_embedding)
        
        embeddings_array = np.array(embeddings)
        return torch.tensor(embeddings_array, dtype=torch.float)
    
    def calculate_similarity_matrix(self, embeddings):
        embeddings_tensor = embeddings.clone().detach().float()
        normed = F.normalize(embeddings_tensor, p=2, dim=1)  # Normalize each vector
        similarity_matrix = torch.mm(normed, normed.T).numpy()  # Cosine sim = dot product of normalized vectors
        np.fill_diagonal(similarity_matrix, 0)
        return similarity_matrix

    def extractive_summary(self, sentences, num_sentences=3):
        if len(sentences) < num_sentences:
            print(f"Warning: Number of sentences ({len(sentences)}) is less than requested ({num_sentences}). Returning all sentences.")
            return sentences
        sentence_embeddings = self.get_embeddings(sentences)
        similarity_matrix = self.calculate_similarity_matrix(sentence_embeddings)
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph, max_iter=100)
        selected_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sentences]
        return [sentences[i] for i in sorted(selected_indices)]

    def abstractive_summarize(self, extractive_sentences: list, word_limit: int = 150):
        """
        Perform abstractive summarization based on extractive sentences.

        Args:
            extractive_sentences (list[str]): List of key sentences from extractive summarization.
            word_limit (int): Desired maximum number of words in the final summary.

        Returns:
            str: Abstractive summary generated by Gemini.
            Returns None if there is an error.
        """
        if not extractive_sentences:
            return "No extractive sentences provided for summarization."
    
        prompt = f"""
        Bạn là một trợ lý AI chuyên tóm tắt văn bản. 
        Dưới đây là các câu then chốt được trích xuất từ văn bản gốc:

        ---
        {" ".join(f"- {sentence}" for sentence in extractive_sentences)}
        ---

        **Yêu cầu:**
        1. Tóm tắt nội dung chính một cách tự nhiên, dễ hiểu, có tính liên kết.
        2. Không lặp lại nguyên văn, hãy diễn đạt lại súc tích, giữ nguyên ý nghĩa.
        3. Không cần tiêu đề, dấu gạch đầu dòng, emoji, hoặc bất kỳ tag nào.
        4. Chỉ xuất ra một đoạn văn duy nhất, không kèm thêm phần mở đầu, tiêu đề hay kết luận.
        5. Giới hạn độ dài tối đa khoảng {word_limit} từ.

        **Ví dụ:**

        _Câu then chốt:_
        - Chủ đề A có tác động đáng kể đến lĩnh vực B.
        - Nhiều chuyên gia cho rằng xu hướng C sẽ tiếp tục phát triển.
        - Một số yếu tố bên ngoài như D cũng ảnh hưởng đến tình hình chung.

        _Tóm tắt mẫu:_
        Chủ đề A đang ảnh hưởng mạnh đến lĩnh vực B, đồng thời xu hướng C được kỳ vọng sẽ tiếp tục phát triển trong thời gian tới. Các yếu tố bên ngoài như D cũng góp phần định hình bối cảnh hiện tại.

        ---

        Bây giờ, hãy áp dụng cách làm tương tự với các câu then chốt sau và chỉ trả về đoạn tóm tắt:
        """
        try:
            response = self.abstractive_model.generate_content(prompt)
            summary = response.text.strip()

            return summary

        except Exception as e:
            print(f"Error during abstractive summarization: {e}")
            return None

    def summarize(self, text, num_sentences = 5):
        sentences = self.pre_processing(text)
        if not sentences:
            print("No content after preprocessing.")
            return "", []
        extractive_output = self.extractive_summary(sentences, num_sentences=num_sentences)
        summary = self.abstractive_summarize(extractive_output)
        return summary
    
    def question_and_answer(self, query, evidence=None):
        """
        Answer a question based on either provided evidence or by searching online.
        If evidence is provided, skip the search. Otherwise, perform the search.
        """

        # === Step 1: Understand the question ===
        reasoning = self.qa_utility.understanding_the_question(query)

        # === Step 2: Determine evidence source ===
        all_evidences = []

        if evidence:  # Evidence is provided by user
            for ev in evidence:
                all_evidences.append(
                    f"Source: {ev['title']}\n"
                    f"Author: {ev['author']}\n"
                    f"URL: {ev['url']}\n"
                    f"Snippet: {ev['main_text']}\n"
                )
        else:  # No evidence → Perform search
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self.qa_utility.search_web_fast, query=query)
                while not future.done():
                    time.sleep(0.1)
                search_results = future.result()
                for result in search_results:
                    all_evidences.append(
                        f"Source: {result['title']}\n"
                        f"Author: {result['author']}\n"
                        f"URL: {result['url']}\n"
                        f"Snippet: {result['main_text']}\n"
                    )

        # === Step 3: Synthesize answer ===
        answer = self.qa_utility.synthesize_and_summarize(query=query, reasoning=reasoning, evidence=all_evidences)

        return answer, all_evidences, query

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
        
