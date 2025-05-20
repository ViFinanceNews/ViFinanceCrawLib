from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_sequence
import re
import nltk
from typing import List
import networkx as nx  # support the the Text-Rank Algorithm
import os
# import json
import concurrent.futures
import time
from ViFinanceCrawLib.QualAna.ArticleFactCheckUtility import ArticleFactCheckUtility
import queue
from tqdm import tqdm
# import subprocess

class SummarizerAlbert:
    def __init__(self, stopword_file="vietnamese-stopwords-dash.txt",
                extractive_model="cservan/multilingual-albert-base-cased-32k",  
                task='feature-extraction',  
                abstractive_model ='gemini-2.0-flash-lite'):

        # setting up with abstractive model
        load_dotenv()
        genai.configure(api_key=os.getenv("API_KEY"))
        self.abstractive_model = genai.GenerativeModel(abstractive_model)
        self.qa_utility = ArticleFactCheckUtility()

        # Get the directory of the current file (summarizer.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print("📂 Current dir:", current_dir)
        stopword_path = os.path.join(current_dir, stopword_file)
        resource = "punkt_tab"
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print(f"{resource} is already installed.")
        except LookupError:
            print(f"{resource} not found. Downloading...")
            nltk.download(resource)
            print(f"{resource} downloaded successfully.")
        
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
        
        self._tokenizer = None
        self._model = None
        self.model_root="/app/models/hub"
        self.extractive_model = extractive_model
        
        print("DONE")
        return

    @property
    def tokenizer(self):
        if self._tokenizer is None or self._model is None:
            self._tokenizer, self._model = self._load_extractive_model()
        return self._tokenizer

    @property
    def model(self):
        if self._model is None or self._tokenizer is None:
            self._tokenizer, self._model = self._load_extractive_model()
        return self._model

    def _load_extractive_model(self):
        print("📦 Importing transformers...")
        import time
        start = time.time()
        from transformers import AutoModel, AutoTokenizer  # do not move this out
        end = time.time()
        print(f"🕒 transformers import time: {end - start:.2f} seconds")

        using_volume = os.path.isdir(self.model_root) and len(os.listdir(self.model_root)) > 0
        print(f"[INFO] using_volume = {using_volume}")

        start = time.time()
        if not using_volume:
            print("🔍 Downloading model from Hugging Face...")
            tokenizer = AutoTokenizer.from_pretrained(self.extractive_model)
            model = AutoModel.from_pretrained(self.extractive_model)
        else:
            print("🔍 Loading model from local volume...")
            albert_folder = self.find_model_folder("models--cservan--multilingual-albert-base-cased-32k", self.model_root)
            if albert_folder is None:
                raise Exception("❌ Model folder not found in volume.")
            tokenizer = AutoTokenizer.from_pretrained(albert_folder)
            model = AutoModel.from_pretrained(albert_folder)

        end = time.time()
        print(f"🕒 Model + Tokenizer loaded in {end - start:.2f} seconds")

        model.eval()
        return tokenizer, model

    def find_model_folder(self, model_name, root_model_dir):
        for root, dirs, files in os.walk(root_model_dir):
            # Check if 'config.json' exists (which indicates a valid Hugging Face model folder)
            if "config.json" in files and model_name in root:
                return root
        return None
    
    def text_normalize(self, text):
        return text.strip().lower()  # Basic normalization (can be extended)

    def pre_processing(self, text):
      from nltk.tokenize import sent_tokenize, word_tokenize
      text = self.text_normalize(text)
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
              tokens = word_tokenize(sentence)
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

              processed_tokens = [
                    token if token.endswith("%") else self.text_normalize(token)
                    for token in new_tokens
                    if token.endswith("%") or token.lower() not in self.stopword
                ]

              processed_sentences.append(" ".join(processed_tokens))

          except Exception as e:
              print(f"Error processing sentence: {e}")
              continue
      return processed_sentences
    
    def get_embeddings(self, texts: List[str]):
        import torch
        """
        Get sentence embeddings for a list of texts using a locally loaded transformer model.
        Uses dynamic batching and mean pooling over token embeddings.
        """
        num_texts = len(texts)

        # Dynamically choose a reasonable even batch size
        if num_texts <= 2:
            batch_size = num_texts
        else:
            batch_size = max(2, (num_texts // 4) * 2)  # Ensure it's even

        all_embeddings = []

        for i in range(0, num_texts, batch_size):
            batch = texts[i:i + batch_size]

            # Tokenize the batch with padding and truncation
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            # Forward pass (no gradients needed)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Mean pooling over the token dimension (dim=1)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # shape: (batch_size, hidden_dim)

            all_embeddings.append(batch_embeddings)

        # Concatenate all batched embeddings into one tensor
        return torch.cat(all_embeddings, dim=0)
    
    def calculate_similarity_matrix(self, embeddings):
        import torch
        import torch.nn.functional as F
       
        # embeddings: [num_sentences, hidden_dim]
        embeddings_tensor = embeddings.clone().detach().float()
        normed = F.normalize(embeddings_tensor, p=2, dim=1)  # Normalize each vector using L2 norm
        similarity_matrix = torch.mm(normed, normed.T).numpy()  # Cosine sim = dot product of normalized vectors
        np.fill_diagonal(similarity_matrix, 0)
        return similarity_matrix

    def extractive_summary(self, sentences, num_sentences=3):
        if len(sentences) < num_sentences:
            print(f"Warning: Number of sentences ({len(sentences)}) is less than requested ({num_sentences}). Returning all sentences.")
            return sentences
        
        sentence_embeddings = self.get_embeddings(sentences)
        from torch.nn.utils.rnn import pad_sequence
        sentence_embeddings =  pad_sequence(sentence_embeddings, batch_first=True, padding_value=0) # Convert list to tensor
        similarity_matrix = self.calculate_similarity_matrix(sentence_embeddings)

        # Build graph and compute PageRank scores
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph, max_iter=100)
        
        # Select top-ranked sentences
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

    def summarize(self, text, num_sentences = 5): # Doing sumamrization on 1 article
        sentences = self.pre_processing(text)
        if not sentences:
            print("No content after preprocessing.")
            return "", []
        extractive_output = self.extractive_summary(sentences, num_sentences=num_sentences)
        summary = self.abstractive_summarize(extractive_output)
        return summary
    
    def process_article(self, article):
        """Helper function to extract key sentences from a single article."""
        sentences = self.pre_processing(article['main_text'])
        if not sentences:
            return None
        
        article_sum = {
            "title": article["title"],
            "url": article["url"],
            "authors": article["author"],
            "date_publish": article["date_publish"],
            "extractive_sum": self.extractive_summary(sentences)
        }
        return article_sum  # Return structured dictionary (not string)

    def multi_source_extractive(self, articles, max_workers=4):
        """Extract key sentences from multiple articles using parallel processing."""

        if not articles:
            return []

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_article, article): article for article in articles}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting Sentences"):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing article: {e}")
        
        return results



    def multi_article_synthesis(self, articles, word_limit=200):
        """
        Generate a synthesized summary from multiple articles.

        Args:
            articles (list[dict]): A list of articles, each represented as a dictionary with field:
                'title', 'main_text', 'url', authors, date_publish 
            with title, url -> from Redis & url -> from scrapper
            num_sentences (int): Number of key sentences to extract per article.
            word_limit (int): Maximum word limit for the final synthesized summary.

        Returns:
            str: A cohesive abstractive summary of all input articles.
        """
        if not articles:
            return "No articles provided for synthesis."

        all_extractive_sentences = self.multi_source_extractive(articles)
    

        # Step 2: Generate a cohesive abstractive summary in a single API call
        prompt = f"""
        Bạn là một trợ lý AI chuyên tóm tắt nội dung từ nhiều bài báo.
        Dưới đây là các câu then chốt được trích xuất từ nhiều bài viết khác nhau - cùng với Meta-Data của chúng:

        ---
        {" ".join(f"- {sentence}" for sentence in all_extractive_sentences)}
        ---

                
        **Yêu cầu:**  
        1. Hợp nhất nội dung từ nhiều bài báo thành một đoạn tóm tắt mạch lạc.  
        2. Trình bày súc tích, dễ hiểu, giữ nguyên ý nghĩa chính.  
        3. Không lặp lại nguyên văn, diễn đạt tự nhiên.  
        4. Giới hạn độ dài tối đa khoảng {word_limit} từ.  
        5. Không được thêm bất cứ phản hồi hay biểu cảm nào dư thừa, chỉ được có tóm tắt.  
        6. Sử dụng trích dẫn trong văn bản theo định dạng APA 7th: [Tên tác giả, Năm].  
        - **Ví dụ:** ([Smith, 2020]) hoặc ([Nguyen & Tran, 2023]).  
        7. Chỉ tóm tắt, không thêm ý kiến chủ quan hay bình luận.  

        **Ví dụ về định dạng mong muốn:** *(CHỈ ĐƯỢC THAM KHẢO)*

        {{
            "synthesis_paragraph": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. [1] Curabitur feugiat ex at quam malesuada sagittis. Aliquam euismod eros tempor magna iaculis placerat. Nam vel faucibus nisl, et convallis nibh. Etiam tempor pulvinar scelerisque. Integer non felis quis risus varius congue nec et nisl [2]. Donec pretium sem eget luctus iaculis. Vestibulum eget condimentum lorem, vitae elementum dui. Vestibulum gravida a magna id imperdiet. Proin lacinia urna a volutpat convallis.",
            "reference": {{
                "1": {{
                    "title": "Author & Coauthor, 2021",
                    "url": "https://example.com"
                }},
                "2": {{
                    "title": "Researcher, 2023",
                    "url": "https://example.com"
                }}
            }}
        }}

        **HẾT VÍ DỤ**

        Hãy viết đoạn tóm tắt chính xác và khách quan:
        """

        try:
            response = self.abstractive_model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            print(f"Error during multi-article synthesis: {e}")
            return None

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
        
    
