"""
    This is the QuantAna module (Quantitative Analysis) module
    The Problem that QuantAna trying to is to provide the quantitative analysis
    on the article content
    Including: Sentimental Analysis - Toxicity Detection 

"""
import sys
from ViFinanceCrawLib.article_database.TextCleaning import TextCleaning as tc
import torch
from detoxify import Detoxify
from sentence_transformers import util
from transformers import pipeline, AutoTokenizer, AutoModel
from typing import List
import numpy as np
import pandas as pd
from vncorenlp.vncorenlp import VnCoreNLP
from dotenv import load_dotenv
import google.generativeai as genai
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import re

class QuantAnaInsAlbert:
    
    def __init__(self, device="cpu"):
        load_dotenv()
        genai.configure(api_key=os.getenv("API_KEY"))

        self.model_name = 'gemini-2.0-flash-thinking-exp-01-21'
        self.translator_model = genai.GenerativeModel(self.model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sentiment pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="tabularisai/multilingual-sentiment-analysis", 
            device=0 if self.device == "cuda" else -1
        )
        
        # Embedding model + tokenizer
        self.embed_model_name = "cservan/multilingual-albert-base-cased-32k"
        self.embed_tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
        self.embed_model = AutoModel.from_pretrained(self.embed_model_name).to(self.device)

        
        self._set_up_vncorenlp()
  
    def _set_up_vncorenlp(self):
        filename = "VnCoreNLP/VnCoreNLP-1.1.1.jar"
        file_path = Path.cwd() / filename
        if not file_path.exists():
            print(f"Error: Required file '{filename}' not found in the current directory.")
            sys.exit(1)
        self.rdrsegmenter = VnCoreNLP(str(file_path), annotators="wseg", max_heap_size='-Xmx500m')

    def get_embeddings(self, texts: List[str]):
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
            inputs = self.embed_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

            # Forward pass (no gradients needed)
            with torch.no_grad():
                outputs = self.embed_model(**inputs)

            # Mean pooling over the token dimension (dim=1)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # shape: (batch_size, hidden_dim)

            all_embeddings.append(batch_embeddings)

        # Concatenate all batched embeddings into one tensor
        return torch.cat(all_embeddings, dim=0)
    
    def compute_semantic_similarity(self, article1: str, article2: str) -> float:
        """Calculate Semantic Similarity between 2 articles using ALBERT embeddings"""
        article_list = [article1, article2]

        # Step 1: Get embeddings locally
        embeddings = self.get_embeddings(article_list)  # shape: [2, hidden_dim]

        # Step 2: Compute cosine similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        return float(similarity)

    def compute_multi_semantic_similarity(self, source_articles, query_article=None, display_table=False):
        """
        Calculate Semantic Similarity:
        - Between query_article and each source article (if provided)
        - Pairwise similarity between source articles

        Args:
            source_articles (List[str]): List of source article strings
            query_article (str, optional): Query article string. Default is None.
            display_table (bool): Whether to print similarity tables

        Returns:
            dict: {
                'query_to_sources': List[float] or None,
                'intersource': List[List[float]]
            }
        """
        try:
            # Step 1: Prepare text batch
            all_articles = source_articles.copy()
            if query_article:
                all_articles.insert(0, query_article)

            # Step 2: Generate embeddings locally using Hugging Face
            embeddings = self.get_embeddings(all_articles)  # List[np.ndarray], each of shape (hidden_dim,)

            # Step 3: Process similarity scores
            query_to_sources = None
            intersource_start_idx = 0

            if query_article:
                query_embedding = embeddings[0].reshape(1, -1)  # shape: (1, dim)
                source_embeddings = [e.reshape(1, -1) for e in embeddings[1:]]
                query_to_sources = [
                    float(cosine_similarity(query_embedding, src)[0][0])
                    for src in source_embeddings
                ]
                intersource_start_idx = 1
            else:
                source_embeddings = [e.reshape(1, -1) for e in embeddings]

            # Step 4: Pairwise source-source similarity matrix
            intersource = []
            for i, emb1 in enumerate(source_embeddings):
                row = []
                for j, emb2 in enumerate(source_embeddings):
                    sim = float(cosine_similarity(emb1, emb2)[0][0])
                    row.append(sim)
                intersource.append(row)

            # Step 5: Optional Display
            if display_table:
                if query_article:
                    query_df = pd.DataFrame({
                        'Source': [f'Source_{i+1}' for i in range(len(query_to_sources))],
                        'Matching_to_Query': query_to_sources
                    })
                    print("=== Query to Sources Similarity ===")
                    print(query_df.round(3))
                    print("\n")

                labels = [f"Source_{i+1}" for i in range(len(source_embeddings))]
                matrix_df = pd.DataFrame(intersource, index=labels, columns=labels)
                print("=== Intersource Similarity Matrix ===")
                print(matrix_df.round(3))

            return {
                'query_to_sources': query_to_sources,
                'intersource': intersource
            }

        except Exception as e:
            print(f"[ERROR] Local Semantic Similarity Failed: {str(e)}")
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
        try:
            result = self.sentiment_pipeline(article_text)[0]  # Only take top label

            sentiment_label = result['label']  # e.g., 'NEGATIVE', 'POSITIVE', 'NEUTRAL'
            sentiment_score = result['score']

            return {
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score
            }

        except Exception as e:
            print(f"[ERROR] Sentiment Analysis Failed: {str(e)}")
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
    
    def obsence_check(self,query):
        """
        Check the obscene level of the query
        """
        try:
            query=self.translation_from_Vie_to_Eng(query)
            prompt = f"""
            Check the query for obscene content, this includes:
            - Offensive language
            - Hate speech
            - Harassment
            - Profanity
            - Threats
            Return "True" if the query contains obscene content, otherwise return "False"
            """
            response = self.translator_model.generate_content(prompt)
            if not hasattr(response, "text") or not response.text:
                print("⚠️ Warning: Empty response from AI model.")
                return False

            return response.text.strip().lower() == "true"
        except Exception as e:
            print(f"❌ Error in obsence_check: {e}")
            return False
   