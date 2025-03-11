"""
 This module includes utility methods & objects for
 supporting article fact-checking.
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup  
from article_database.ArticleScraper import ArticleScraper
from article_database.SearchEngine import SearchEngine
from article_database.TitleCompleter import TitleCompleter

class GenModelUtility:
    
    def __init__(self, model_name='gemini-2.0-pro-exp-02-05'):
        # Load environment variables
        load_dotenv(dotenv_path=".devcontainer/devcontainer.env")
        
        # Validate API keys
        self.api_key = os.getenv("API_KEY")
        self.search_api_key = os.getenv("SEARCH_API_KEY")
        self.search_engine_id = os.getenv("SEARCH_ENGINE_ID")

        if not self.api_key:
            raise ValueError("❌ Missing API_KEY. Ensure it is set in the .env file.")
        if not self.search_api_key:
            raise ValueError("❌ Missing SEARCH_API_KEY. Ensure it is set in the .env file.")
        if not self.search_engine_id:
            raise ValueError("❌ Missing SEARCH_ENGINE_ID. Ensure it is set in the .env file.")

        # Configure the AI model
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

        # Initialize components
        self.tc = TitleCompleter()
        self.search_engine = SearchEngine(self.search_api_key, self.search_engine_id)
        self.article_scraper = ArticleScraper()

    def generate_search_queries(self, statement):
        """
        Generates search queries from a given statement in Vietnamese.
        """
        prompt = f"""

                    Hãy xác định các từ khóa và khái niệm chính từ thông tin sau. Dựa trên nội dung này, hãy tạo một câu hỏi trung lập, kích thích tư duy, khuyến khích thảo luận mà không giả định một quan điểm cụ thể. Câu hỏi phải có độ dài từ 10 đến 50 từ.

                    Trả về duy nhất câu hỏi đã tạo, không có văn bản xung quanh, và bạn PHẢI viết bằng tiếng Việt.

                    Sau đó, trích xuất từ khóa từ câu hỏi và trả về chúng ở định dạng:
                    [Từ khóa 1, Từ khóa 2, …]
        Thông tin: {statement}
        """

        try:
            response = self.model.generate_content(prompt)
            if not hasattr(response, "text") or not response.text:
                print("⚠️ Warning: Empty response from AI model.")
                return []

            claims = [claim.strip() for claim in response.text.split("\n") if claim.strip()]
            return claims
        
        except Exception as e:
            print(f"❌ Error in generate_search_queries: {e}")
            return []
    
    def generate_bias_analysis(self, article: str):
            """
            Generate a qualitative bias and logical fallacy analysis on the article,
            specifying the types of bias and logical fallacies to focus on.
            
            Parameters:
                article (str): The article content to analyze.
    
            Returns:
                str: The formatted prompt for LLM analysis.
            """
          
            prompt = f"""
                Bạn là một nhà báo phân tích phản biện, chuyên đánh giá độ chính xác và khách quan của thông tin.  
                
                Hãy phân tích bài viết sau để xác định các thiên kiến và lỗi lập luận có thể có.  
                - **Không chỉ dựa vào từ khóa**, hãy đánh giá ngữ cảnh và cách lập luận để nhận diện thiên kiến hoặc lỗi logic.  
                - Nếu bài viết trung lập, hãy kết luận trung lập. Nếu có thiên kiến hoặc lỗi lập luận, hãy đánh giá mức độ ảnh hưởng.  
                
                Xuất kết quả theo định dạng sau, chỉ bao gồm nội dung phân tích mà không thêm giải thích hoặc biểu cảm dư thừa:  

                - **Loại thiên kiến:** [Chính trị, giới tính, văn hóa, thiên kiến xác nhận, v.v.]  
                - **Mức độ ảnh hưởng:** [Nhẹ, vừa, nghiêm trọng]  
                - **Phân tích ngắn gọn:** [Giải thích thiên kiến trong tối đa 200 từ, dựa trên ngữ cảnh và lập luận của bài viết]  

                ---  
                **Câu hỏi phản biện để giúp người đọc có góc nhìn khách quan hơn:**  
                (Hãy đưa ra 3–5 câu hỏi theo phương pháp Socrates, khuyến khích người đọc suy nghĩ sâu hơn về lập luận trong bài viết)  

                Bài viết cần phân tích:  
                \"\"\"  
                {article}  
                \"\"\"
            """
            try:
                response = self.model.generate_content(prompt)
                if not hasattr(response, "text") or not response.text:
                    print("⚠️ Warning: Empty response from AI model.")
                    return []

                analysis = response.text
                return analysis
            
            except Exception as e:
                print(f"❌ Error in generate_search_queries: {e}")
                return []

    def understanding_the_question(self, query):
        """
        Method 1:  Reasoning - Understand the User Query using Gemini.

        This method sends the user's query to Gemini and asks it to
        explain its reasoning process for understanding the question.
        The reasoning is captured and returned, but not printed to the user directly.

        Args:
            query (str): The user's query.

        Returns:
            str: A string representing Gemini's reasoning process for understanding the query.
            Returns None if there's an error communicating with Gemini.
        """
        try:
            
            prompt = f"""
            
            Bạn là một chuyên gia phân tích dữ liệu có nhiệm vụ trả lời câu hỏi dựa trên dữ liệu tìm kiếm.  
            Trước khi tạo câu trả lời, **Hãy hiểu và suy luận về ý nghĩa và trọng tâm của câu hỏi** rồi trả lại
            kết quả đầu ra\n
            \n
            *Kết quả quá trình suy luận*\n 
            - Xác định vấn đề chính cần phải trả lời: [vấn đề 1, vấn đề 2, etc.]\n 
            - Xác định từ khóa tìm kiếm tối ưu. [Từ khoá 1, từ khoá 2, etc.]\n 
            - Xác định giả định tiềm ẩn (nếu có): [Giả thuyết 1, Giả thuyết 2, etc.] \n 
            \n 
            Câu hỏi của người dùng: '{query}' \n 
            
            """
            response = self.model.generate_content(prompt)
            reasoning = response.text
            return reasoning
        
        except Exception as e:
            print(f"Error in reasoning method: {e}")
        return None
        
    def synthesize_and_summarize(self, query, reasoning, evidence):
        """
        Method 2: Synthesis and Summarization - Generate a clear answer based on reasoning.

        This method takes the original user query and the reasoning obtained from
        reason_about_query(). It uses Gemini to synthesize evidence (implicitly from its knowledge)
        and summarize it into a clear and concise answer, guided by the provided reasoning.

        Args:
            query (str): The original user query.\n
            reasoning (str): The reasoning process obtained from reason_about_query().\n
            evidence (list[str]]): The list of evidence main_text\n

        Returns:
            str: A clear and concise summarized answer to the user's query.\n
            Returns None if there's an error communicating with Gemini.\n
        """
        if reasoning is None:
            return "Could not determine reasoning for the query. Please try again."

        try:     
            prompt = f"""
            Bạn là một trợ lý AI chuyên phân tích, tổng hợp và tóm tắt thông tin để trả lời truy vấn của người dùng một cách rõ ràng và chính xác.\n

            ## **Nhiệm vụ của bạn**:
            Dựa trên truy vấn của người dùng, lập luận đã có, và danh sách bằng chứng, hãy tổng hợp câu trả lời một cách logic, dễ hiểu, và ngắn gọn.
            \n
            ---\n

            ## **Dữ liệu đầu vào**:\n
            **Truy vấn**: {query}\n 
            **Lập luận hỗ trợ**: {reasoning}\n
            **Bằng chứng**:  {evidence}\n 

            ---

            ## **Yêu cầu đối với câu trả lời**:\n
            1. **Tóm tắt ngắn gọn nhưng đầy đủ**:\n  
            - Không chỉ trích dẫn mà phải tổng hợp thông tin từ bằng chứng.\n   
            - Đảm bảo câu trả lời có ý nghĩa ngay cả khi không có đầy đủ ngữ cảnh ban đầu. \n  
            \n
            2.**Sử dụng lập luận hợp lý**: \n  
            - Tận dụng reasoning để đưa ra kết luận logic. \n  
            - Nếu bằng chứng mâu thuẫn, hãy chỉ ra điểm khác biệt thay vì đưa ra một câu trả lời phiến diện.  \n 
            \n 
            3.**Định dạng câu trả lời**:\n 
            **Tóm tắt cuối cùng**: [Tóm tắt câu trả lời dựa trên bằng chứng]\n   
            **Nguồn tham khảo**: [Danh sách nguồn thông tin & nếu có thể hãy đính kèm link nguồn]  \n 
            \n 
            🎯 **Lưu ý**:\n 
            - Nếu không có đủ bằng chứng, hãy nêu rõ điều đó thay vì đưa ra câu trả lời suy đoán.\n 
            - Nếu có lỗi hoặc không thể tổng hợp được câu trả lời, trả về `"Không thể đưa ra kết luận rõ ràng."`\n 
            - Ví dụ cách chèn liên kết:
                - "Bằng chứng từ [nguồn này](URL) cho thấy rằng..."  
                - "Theo thông tin từ bài viết này ([link](URL)), ..."  
            """
            response = self.model.generate_content(prompt)
            summary = response.text
            return summary
        except Exception as e:
            print(f"Error in synthesis and summarization method: {e}")
            return None
    
    def search_web(self, query, num_results=5):
        """
        Searches the web for relevant articles based on the query.
        Filters out invalid articles and completes article titles.
        """
        articles = self.search_engine.search(query, num=num_results)
        valid_articles = []
        for article in articles:
            try:
                if not article.get("link"):
                    continue  # Skip articles with no links

                original_title = article["title"]
                # Scrape article content

                link = article["link"]
                article_data = ArticleScraper.scrape_article(link)

                if (
                    not article_data["main_text"]
                    or article_data["main_text"] == "No content available"
                    or article_data["author"] == "Unknown"
                ):
                    continue  # Skip invalid articles

                # ✅ Complete title AFTER filtering
                article_data["title"] = self.tc.complete_title(original_title=original_title, article=article)
                valid_articles.append(article_data)

            except Exception as e:
                print(f"❌ Error processing article {article['link']}: {e}")
        return valid_articles

    def analyze_evidence(self, statement, evidence):
        # Cretability metric - https://rusi-ns.ca/a-system-to-judge-information-reliability/
        """
        Analyzes evidence to determine if it supports, contradicts, or is neutral to the statement.
        """
        prompt = f"""Bạn là một trợ lý phân tích và đánh giá tính xác thực của thông tin. Dưới đây là một mệnh đề & thông tin và một tập hợp bằng chứng. Hãy đánh giá mức độ mà bằng chứng hỗ trợ, mâu thuẫn hoặc trung lập đối với thông tin, bằng cách xem xét:
                • Mối quan hệ logic giữa tuyên bố và bằng chứng.
                • Độ mạnh của bằng chứng, bao gồm nguồn gốc, tính chính xác và mức độ liên quan.
                • Bối cảnh và giả định tiềm ẩn có thể ảnh hưởng đến diễn giải bằng chứng.

            ### **Hệ thống đánh giá độ tin cậy**  
            Hãy đánh giá mức độ tin cậy của từng nguồn và từng thông tin bằng hệ thống NATO:  

            - **Đánh giá độ tin cậy của nguồn** (Chữ cái):  
            - **A**: Hoàn toàn đáng tin cậy  
            - **B**: Đáng tin cậy  
            - **C**: Khá đáng tin cậy  
            - **D**: Không đáng tin cậy  
            - **E**: Không thể đánh giá  

            - **Đánh giá độ chính xác của thông tin** (Chữ số):  
            - **1**: Đã được xác minh  
            - **2**: Có khả năng đúng  
            - **3**: Có thể đúng  
            - **4**: Không chắc chắn  
            - **5**: Không thể đánh giá  

            Kết quả đánh giá sẽ được biểu diễn dưới dạng **A1, B2, C3, v.v.**, trong đó:  
            - **A1** là thông tin đáng tin cậy nhất, có nguồn mạnh và đã được xác minh.  
            - **E5** là thông tin đáng tin cậy kém nhất, có nguồn yếu và không thể đánh giá.  

            Cuối cùng, hãy đưa ra điểm tin cậy (0-100) cho mức độ đánh giá của bạn, thể hiện mức độ chắc chắn về kết luận của mình.  

            Mệnh đề thông tin: {statement}  

            Bằng chứng:  
            {evidence}  

            ### **Hãy trả lời theo định dạng sau:**  
            - **Tổng Hợp Cuối Cùng**: [Tóm tắt thông tin đã kiểm tra để đưa ra kết luận cuối cùng về chủ đề.]  
            - **Kết luận**: [Hỗ trợ/Mâu thuẫn/Trung lập]  
            - **Mức độ tin cậy**: [Ví dụ: A1, B3, D5] và chú thích của mức độ [ví dụ: A1 - Đáng Tin Cậy và Đã Được Xác Minh]   
            - **Giải thích**: [Giải thích ngắn gọn về lý do của bạn, có đề cập đến nguồn bằng chứng và mức độ tin cậy của chúng. Nếu có URL trong bằng chứng, hãy chèn nó vào trong lời giải thích dưới dạng liên kết.]  
            - **Lời khuyên cho người dùng về cách nhìn nhận hiện tại**: [Một lời khuyên ngắn gọn]  

            ### **Ví dụ cách chèn liên kết:**  
            - "Bằng chứng từ [nguồn này](URL) cho thấy rằng..."  
            - "Theo thông tin từ bài viết này ([link](URL)), ..."  
            """

        try:
            response = self.model.generate_content(prompt)
            if not hasattr(response, "text") or not response.text:
                print("⚠️ Warning: AI model returned empty response for evidence analysis.")
                return "Không có phản hồi từ AI."

            return response.text

        except Exception as e:
            print(f"❌ Error in analyze_evidence: {e}")
            return "Đã xảy ra lỗi khi phân tích bằng chứng."