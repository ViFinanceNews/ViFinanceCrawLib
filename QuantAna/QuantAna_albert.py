"""
    This is the QuantAna module (Quantitative Analysis) module
    The Problem that QuantAna trying to is to provide the quantitative analysis
    on the article content
    Including: Sentimental Analysis - Toxicity Detection 

"""
import sys
from ViFinanceCrawLib.article_database.TextCleaning import TextCleaning as tc
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
# from sentence_transformers import util


class QuantAnaInsAlbert:
    
    def __init__(self, device="cpu", model_root="/app/models/hub"):
        load_dotenv()
        genai.configure(api_key=os.getenv("API_KEY"))

        self.model_name = 'gemini-2.0-flash-lite'
        self.translator_model = genai.GenerativeModel(self.model_name)
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # place holder for lazy-loading
        self._models_loaded = False
        self.sentiment_pipeline = None
        self.embed_model_name = None
        self.embed_tokenizer = None
        self.embed_model = None
        self.toxicity_model = None

        self._set_up_vncorenlp()
        print("Load QuantAna done !")
  
    def _set_up_vncorenlp(self):
        filename = "VnCoreNLP/VnCoreNLP-1.1.1.jar"
        file_path = Path.cwd() / filename
        if not file_path.exists():
            print(f"Error: Required file '{filename}' not found in the current directory.")
            sys.exit(1)
        self.rdrsegmenter = VnCoreNLP(str(file_path), annotators="wseg", max_heap_size='-Xmx500m')

    def load_models(self, model_root="/app/models/hub"):
        from detoxify import Detoxify
        from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
        print(f" check if the model been loaded{self._models_loaded}")
        if self._models_loaded:
            return  # Already loaded, do nothing
        
        self.device = "cpu"
        using_volume = os.path.isdir(model_root) and len(os.listdir(model_root)) > 0
        print(f"[INFO] using_volume = {using_volume}")

        if not using_volume:
            # Sentiment pipeline
            model_name = "tabularisai/multilingual-sentiment-analysis"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to("cpu")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=model, 
                device=-1, # Force Using CPU
                tokenizer=tokenizer
            )
            
            # Embedding model + tokenizer
            self.embed_model_name = "cservan/multilingual-albert-base-cased-32k"
            self.embed_tokenizer = AutoTokenizer.from_pretrained(self.embed_model_name)
            self.embed_model = AutoModel.from_pretrained(self.embed_model_name).to("cpu")
            self.toxicity_model = Detoxify(model_type="original-small", device = "cpu")
        
        else:
            root_model_dir = model_root
            albert_folder = self.find_model_folder("models--cservan--multilingual-albert-base-cased-32k", root_model_dir)
            sentiment_model_folder = self.find_model_folder("models--tabularisai--multilingual-sentiment-analysis", root_model_dir)
            
            if albert_folder is None or sentiment_model_folder is None:
                raise Exception("Model folder(s) not found!")
            self.embed_tokenizer =AutoTokenizer.from_pretrained(albert_folder)
            self.embed_model = AutoModel.from_pretrained(albert_folder).to("cpu")

            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model=sentiment_model_folder, 
                device= -1, # Force CPU
                tokenizer=sentiment_model_folder
            )
            toxicity_path = self.find_model_folder_checkpoint_keyword(keyword="original-albert", root_model_dir=root_model_dir)
            if toxicity_path is not None:
                model_name = "original-albert-0e1d6498.ckpt"
                self.toxicity_model = Detoxify(model_type="original-small", checkpoint=toxicity_path + "/" + model_name, device="cpu")
            else:
                self.toxicity_model = Detoxify(model_type="original-small", device="cpu")
        print("Sentiment Model  & Toxicity Model loaded Successfully")

    def sentiment_analysis(self, article_text):
        self.load_models()
        article_text = re.sub(r"\[Câu \d+\]\s*", "", article_text)
        print(article_text)
        try:
            result = self.sentiment_pipeline(article_text)[0]  # Only take top label
            print(f"result: {result}")
            sentiment_label = result['label']  # e.g., 'NEGATIVE', 'POSITIVE', 'NEUTRAL'
            raw_score = result['score']
            
            # Normalize score to 1–10
            discrete_score = int(round(raw_score * 9)) + 1
            discrete_score = min(max(discrete_score, 1), 10)  # Ensure it's in [1, 10]

            return {
                "sentiment_label": sentiment_label,  # e.g., 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
                "sentiment_score": discrete_score
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
        """
        Convert a float score in range [0.0, 1.0] to a discrete score in range [1, 10].
        """
        try:
            real_value = float(value)
            discrete_score = int(round(real_value * 9)) + 1  # Map 0.0 -> 1, 1.0 -> 10
            return min(max(discrete_score, 1), 10)  # Ensure within range
        except Exception:
            return 1  # Fallback if error occurs
    
    def detect_toxicity(self, article_text: str):
        """Detects toxicity and misinformation in the article.
           Parameter: article_text (Vietnamese String need to be pre-process and segmentized)
           Returns:
            A dictionary with the format: (the printed out result would be translate to Vietnamese)
                {"Toxicity: Score", "Insult": Score, "Threat": Score, "Identity Attack" : Score, "Obscene" :Score}
        """
        self.load_models()
        try:
            tokenized_text = self.rdrsegmenter.tokenize(article_text)
            print(f"tokenized_text:{tokenized_text}")
            pre_processed_sentences = self.combine_tokens(tokenized_text[0])
            
            translation = self.translation_from_Vie_to_Eng(pre_processed_sentences)
            print(f"Translation:{translation}")
            toxicity_score = self.toxicity_model.predict(translation)
            print(f"toxicity_score{toxicity_score}")
            
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
            Does the following text contain any of the following: obscene language, hate speech, harassment, profanity, or threats?

            MUST Return only one word: True or False. No explanation.

            Text: \"\"\"{query}\"\"\"
            """
            response = self.translator_model.generate_content(prompt)
            print(f"Obscence Check Result : {response.text}")
            if not hasattr(response, "text") or not response.text:
                print("⚠️ Warning: Empty response from AI model.")
                return False

            return response.text.strip().lower() == "true"
        except Exception as e:
            print(f"❌ Error in obsence_check: {e}")
            return False
   
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

