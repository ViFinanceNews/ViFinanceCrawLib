"""
 This module include a list of method & object of Utility for
 supporting the Article Fact Check
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv
import requests
import time
from typing import List, Optional
from bs4 import BeautifulSoup  
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Optional: For progress visualization
from ViFinanceCrawLib.article_database.ArticleScraper import ArticleScraper
from ViFinanceCrawLib.article_database.SearchEngine import SearchEngine
from ViFinanceCrawLib.article_database.SearchEngine import SearchEngine
from ViFinanceCrawLib.article_database.TitleCompleter import TitleCompleter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import json
import re
class ArticleFactCheckUtility():

    def __init__(self, model_name='gemini-2.0-flash-lite'):
        load_dotenv()
        genai.configure(api_key=os.getenv("API_KEY"))
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        self.tc = TitleCompleter()

        self.search_engine  =  SearchEngine(os.getenv("SEARCH_API_KEY"), os.getenv("SEARCH_ENGINE_ID"))
        self.article_scraper = ArticleScraper()
        return
        
    def generate_search_queries(self, statement):
        prompt = f"""Identify keywords and concepts from the following statement. Using this claim, generate a neutral, thought-provoking question that encourages discussion without assuming a particular stance. The question must be between 10 and 50 words long. Return only the generated question with no other surrounding text, and you MUST write it in Vietnamese.

        Statement: {statement}
        """
        response = self.model.generate_content(prompt)
        claims = response.text.split('\n')
        claims = [claim.strip() for claim in claims if claim.strip()]  # Clean up
        return claims
    
    def fact_check_article_using_query(self, article_text):
        """
        Fact-check an article by generating neutral, thought-provoking search queries
        based on its content.

        Parameters:
        - article_text (str): The full text of the article to fact-check.

        Returns:
        - List of generated Vietnamese search queries.
        """
        # Step 1: Create a prompt to extract key claims/questions from article
        prompt = f"""Identify keywords and concepts from the following article. 
                    Using this article, generate neutral, thought-provoking questions that encourage 
                    discussion without assuming a particular stance. Each question must be between 
                    10 and 50 words long. 

                    You must return the questions in the following format:

                    Query 1: <question in Vietnamese>
                    Query 2: <question in Vietnamese>
                    ...

                    Do not include any other text.

                    Article: {article_text}
                    """
        # Step 2: Get response from model
        response = self.model.generate_content(prompt)

        # Step 3: Process and clean the output
        raw_output = response.text.split('\n')
        search_queries = []
        for line in raw_output:
            line = line.strip()
            if line.startswith("Query"):
                # Extract after 'Query X:'
                query_text = line.split(":", 1)[1].strip()
                if query_text:
                    search_queries.append(query_text)

        return search_queries
    
    def search_web(self, query, num_results=5):
        tc = self.tc
        searchEngine = self.search_engine
        articles = searchEngine.search(query, num=num_results)
        article_scraper = self.article_scraper
        valid_articles = []
        for article in articles:
            try:
                if not article.get("link"):
                    continue  # Skip articles with no links

                original_title = article["title"]
                # Scrape article content
                article_data = article_scraper.scrape_article(article["link"])

                if (
                    not article_data["main_text"]
                    or article_data["main_text"] == "No content available"
                    or article_data["author"] == "Unknown"
                ):
                    continue  # Skip invalid articles

                # ✅ Complete title AFTER filtering
                article_data["title"] = tc.complete_title(original_title=original_title, article=article)
                valid_articles.append(article_data)

            except Exception as e:
                print(f"❌ Error processing article {article['link']}: {e}")
        return valid_articles
    
    def search_web_fast(self, query, num_results=5):
        """
        Main method: Search articles and scrape them in parallel.
        """
        articles = self.search_engine.search(query, num=num_results)
        valid_articles = self.scrape_articles_parallel(articles, batch_size=num_results)
        return valid_articles

    def scrape_articles_parallel(self, articles, batch_size=5):
        """
        Helper method: Scrape articles in parallel.
        Scrape articles concurrently in batches.
        """
        valid_articles = []

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(self.process_single_article, article) for article in articles]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping Articles"):
                result = future.result()
                if result:
                    valid_articles.append(result)

        return valid_articles

    def process_single_article(self, article):
        """
        Helper method: Process a single article.
        Process a single article: Scrape, validate, and format.
        """
        try:
            if not article.get("link"):
                return None  # Skip articles with no links

            original_title = article["title"]

            # Scrape article content
            article_data = self.article_scraper.scrape_article(article["link"])

            # Filter invalid articles
            if (
                not article_data["main_text"]
                or article_data["main_text"] == "No content available"
                or article_data["author"] == "Unknown"
            ):
                return None

            # Complete title after filtering
            article_data["title"] = self.tc.complete_title(original_title=original_title, article=article)
            return article_data

        except Exception as e:
            print(f"❌ Error processing article {article.get('link')}: {e}")
            return None

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

            ### ** DANH SÁCH Ý nghĩa của các tổ hợp đánh giá [PHẢI DÙNG ĐÚNG CHÍNH XÁC CÚ PHÁP NHƯ GỐC]**  
            A1: Hoàn toàn đáng tin cậy & Đã được xác minh  
            A2: Hoàn toàn đáng tin cậy & Có khả năng đúng  
            A3: Hoàn toàn đáng tin cậy & Có thể đúng  
            A4: Hoàn toàn đáng tin cậy & Không chắc chắn  
            A5: Hoàn toàn đáng tin cậy & Không thể đánh giá  
            B1: Đáng tin cậy & Đã được xác minh  
            B2: Đáng tin cậy & Có khả năng đúng  
            B3: Đáng tin cậy & Có thể đúng  
            B4: Đáng tin cậy & Không chắc chắn  
            B5: Đáng tin cậy & Không thể đánh giá  
            C1: Khá đáng tin cậy & Đã được xác minh  
            C2: Khá đáng tin cậy & Có khả năng đúng  
            C3: Khá đáng tin cậy & Có thể đúng  
            C4: Khá đáng tin cậy & Không chắc chắn  
            C5: Khá đáng tin cậy & Không thể đánh giá  
            D1: Không đáng tin cậy & Đã được xác minh  
            D2: Không đáng tin cậy & Có khả năng đúng  
            D3: Không đáng tin cậy & Có thể đúng  
            D4: Không đáng tin cậy & Không chắc chắn  
            D5: Không đáng tin cậy & Không thể đánh giá  
            E1: Không thể đánh giá & Đã được xác minh  
            E2: Không thể đánh giá & Có khả năng đúng  
            E3: Không thể đánh giá & Có thể đúng  
            E4: Không thể đánh giá & Không chắc chắn  
            E5: Không thể đánh giá & Không thể đánh giá  

                        

            Mệnh đề thông tin: {statement}  

            Bằng chứng:  
            {evidence}  

            ### **Hãy trả lời chú ý các ràng buộc phía duới:**  
            - Tổng Hợp Cuối Cùng: [Tóm tắt thông tin đã kiểm tra để đưa ra kết luận cuối cùng về chủ đề.]  
            - Kết luận: [Hỗ trợ/Mâu thuẫn/Trung lập]  
            - Phân tích bằng chứng: [Các dẫn chứng trên có mối liên hệ như thế nào trong việc đưa ra kết luận về vấn đề người dùng tìm hiểu]
            - Mức độ tin cậy: [Ví dụ: A1, B3, D5] và chú thích của CHÍNH XÁC NHƯ DANH SÁCH Ý nghĩa của các tổ hợp đánh giá  [ví dụ: A1 - Đáng Tin Cậy & Đã Được Xác Minh]   
            - Giải thích: [Giải thích ngắn gọn về lý do của bạn, có đề cập đến nguồn bằng chứng và mức độ tin cậy của chúng.]  
            - Lời khuyên cho người dùng về cách nhìn nhận hiện tại: [Một lời khuyên ngắn gọn]  
            - Danh sách các dẫn chứng (mỗi bài báo là một string):  
                [Số thứ tự bài báo ]: Tiêu đề - nguồn -  [url] \n
                [Số thứ tự bài báo ]: Tiêu đề - nguồn -  [url] \n
            ....

            LƯU Ý: Trường: [Mức độ tin cậy] CHỈ DƯỢC TRẢ LẠI MỘT GIÁ TRỊ KẾT QUẢ DUY NHẤT
            ### **Ví dụ cách chèn liên kết:**  
            - "Bằng chứng từ [Số thứ tự bài báo] cho thấy rằng..."  
            - "Theo thông tin từ bài viết này [Số thứ tự bài báo], ..."  

            **Ví dụ phù hợp của định dạng "key": "value" của json, nếu có dấu ngoặc kép (") trong nội dung của value hãy đổi thành dấu ngoặc đơn (') để đúng định dạng json, cấm dùng dấu ngoặc kép (") khi viết phần value:**
            "Tổng Hợp Cuối Cùng": "Các bằng chứng được cung cấp không chứa bất kỳ thông tin nào liên quan đến \"Công ty G\" hay lợi nhuận của công ty này trong năm 2025, cũng như không cung cấp dữ liệu so sánh lợi nhuận của các công ty trong ngành tài chính để xác định công ty có lợi nhuận cao nhất trong năm đó.",
            "Kết luận": "Trung lập",
            "Phân tích bằng chứng": "Các bằng chứng được cung cấp bao gồm các bài báo từ VnExpress và VnEconomy, là các nguồn tin tức kinh doanh uy tín. Tuy nhiên, nội dung của chúng không liên quan trực tiếp hoặc gián tiếp đến mệnh đề \"Công ty G có lợi nhuận cao nhất trong ngành tài chính năm 2025\". Bằng chứng [1] thảo luận về kế hoạch và kết quả kinh doanh của HDBank trong năm 2023 và 2024. Bằng chứng [2] đưa ra đánh giá về các kênh đầu tư tiềm năng trong năm 2024. Bằng chứng [3] giải thích về chỉ số PEG để định giá cổ phiếu. Bằng chứng [4] phân tích kỳ vọng của nhà đầu tư nước ngoài về chính sách kinh tế Việt Nam, chủ yếu tập trung vào năm 2023 và bối cảnh vĩ mô. Không có bằng chứng nào đề cập đến \"Công ty G\" hoặc cung cấp dữ liệu lợi nhuận dự kiến hoặc thực tế cho năm 2025 của bất kỳ công ty tài chính nào, đặc biệt là dữ liệu so sánh để xác định công ty dẫn đầu về lợi nhuận.",
            "Mức độ tin cậy": "B5 - Đáng Tin Cậy (Nguồn) và Không Thể Đánh Giá (Thông tin liên quan đến mệnh đề)",
            "Giải thích": "Đánh giá mức độ tin cậy là B5. Các nguồn tin VnExpress và VnEconomy là các báo điện tử có uy tín và được công nhận trong lĩnh vực kinh doanh và kinh tế tại Việt Nam (đáng tin cậy - B). Tuy nhiên, nội dung của tất cả các bằng chứng [1], [2], [3], [4] đều không chứa bất kỳ thông tin nào về \"Công ty G\" hoặc dữ liệu lợi nhuận của các công ty trong ngành tài chính dự kiến cho năm 2025. Do đó, dựa trên các bằng chứng này, tính chính xác của mệnh đề \"Công ty G có lợi nhuận cao nhất trong ngành tài chính năm 2025\" hoàn toàn \"không thể đánh giá\".",
            "Lời khuyên cho người dùng về cách nhìn nhận hiện tại": "Các bằng chứng hiện có không cung cấp thông tin nào để xác minh hoặc bác bỏ mệnh đề về lợi nhuận của \"Công ty G\" trong năm 2025. Để có được thông tin đáng tin cậy về lợi nhuận của các công ty tài chính và xếp hạng của họ trong tương lai, bạn cần tìm kiếm các báo cáo phân tích chuyên sâu từ các công ty chứng khoán uy tín, báo cáo tài chính dự kiến của các công ty, hoặc các nguồn tin tức tài chính chuyên ngành đưa ra dự báo cụ thể cho năm 2025.",
            "Danh sách các dẫn chứng": **ngoặc nhọn ở đây**
                "[1]":    
                        **ngoặc nhọn ở đây**
                            "title": "HDBank nâng mức chia cổ tức lên 30%", 
                            "publisher": "Báo VnExpress Kinh doanh", 
                            "url": "https://vnexpress.net/hdbank-nang-muc-chia-co-tuc-len-30-4737638.html" **ngoặc nhọn ở đây**
                        ,
                "[2]":    **ngoặc nhọn ở đây**
                        
                            "title": "Lựa chọn kênh đầu tư nào trong năm 2024", 
                            "publisher": "Báo VnExpress Kinh doanh", 
                            "url": "https://vnexpress.net/lua-chon-kenh-dau-tu-nao-trong-nam-2024-4699524.html"**ngoặc nhọn ở đây**
                        ,   
                "[3]":    **ngoặc nhọn ở đây**
                        
                            "title": "Chỉ số PEG là gì?", 
                            "publisher": "Báo VnExpress Kinh doanh", 
                            "url": "https://vnexpress.net/chi-so-peg-la-gi-4861277.html"**ngoặc nhọn ở đây**
                        ,
                "[4]":    **ngoặc nhọn ở đây**
                        
                            "title": "Nhà đầu tư nước ngoài kỳ vọng gì về những phản ứng chính sách của Việt Nam?", 
                            "publisher": "Nhịp sống kinh tế Việt Nam & Thế giới", 
                            "url": "https://vneconomy.vn/nha-dau-tu-nuoc-ngoai-ky-vong-gi-ve-nhung-phan-ung-chinh-sach-cua-viet-nam.htm" **ngoặc nhọn ở đây**
            **ngoặc nhọn ở đây**            
            

    

            Hãy đảm bảo trả lời giống như ví dụ, nhưng không để nội dung ví dụ ảnh hưởng đến đánh giá.
            
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
    
    def generate_bias_analysis(self, article : str):
            """
            Generate a qualitative bias and logical fallacy analysis on the article,
            specifying the types of bias and logical fallacies to focus on.
            
            Parameters:
                article (str): The article content to analyze.
    
            Returns:
                json: The formatted prompt with "key": "value".
            """
            # Hãy phân tích bài viết sau theo định dạng json để xác định các thiên kiến và lỗi lập luận có thể có.  
            prompt = f"""
                Bạn là một nhà báo phân tích phản biện, chuyên đánh giá độ chính xác và khách quan của thông tin.  
                
                
                - **Không chỉ dựa vào từ khóa**, hãy đánh giá ngữ cảnh và cách lập luận để nhận diện thiên kiến hoặc lỗi logic.  
                - Nếu bài viết trung lập, hãy kết luận trung lập. Nếu có thiên kiến hoặc lỗi lập luận, hãy đánh giá mức độ ảnh hưởng.  
                
                **Viết Chú ý các ràng buộc phía duới ,chỉ bao gồm nội dung phân tích mà KHÔNG THÊM GIẢI THÍCH:**  

                - Loại thiên kiến: [Chính trị, giới tính, văn hóa, thiên kiến xác nhận, v.v.]  
                - Mức độ ảnh hưởng: [Nhẹ, vừa, nghiêm trọng]  
                - Phân tích ngắn gọn: [Giải thích thiên kiến trong tối đa 200 từ, dựa trên ngữ cảnh và lập luận của bài viết]  

                ---  
                **Câu hỏi phản biện để giúp người đọc có góc nhìn khách quan hơn:**  
                (Hãy đưa ra 3–5 câu hỏi theo phương pháp Socrates, khuyến khích người đọc suy nghĩ sâu hơn về lập luận trong bài viết)  
            
                **Trả về định dạng như bên dưới - "key": "value" của json format, nếu có dấu ngoặc kép (") trong nội dung của value hãy đổi thành dấu ngoặc đơn (') để đúng định dạng json, cấm dùng dấu ngoặc kép (") khi viết phần nội dung trong value:**
                "Loại thiên kiến": "Lỗi lập luận: Mâu thuẫn trực tiếp",
                "Mức độ ảnh hưởng": "Nghiêm trọng",
                "Phân tích ngắn gọn": "Câu nói chứa mâu thuẫn logic trực tiếp: \"Công ty E thất bại thảm hại\" và \"vẫn là khoản đầu tư tốt nhất\". Theo định nghĩa thông thường trong tài chính, hai trạng thái này khó có thể tồn tại đồng thời một cách hợp lý. Sự mâu thuẫn này làm suy yếu nghiêm trọng tính hợp lý và đáng tin cậy của nhận định, cho thấy sự thiếu rõ ràng trong tiêu chí đánh giá hoặc lỗi trong lập luận, khiến người đọc khó hiểu và chấp nhận thông tin.",
                "Câu hỏi phản biện": [
                    "Những tiêu chí cụ thể nào được sử dụng để xác định đây là \"khoản đầu tư tốt nhất\" trong khi công ty được mô tả là \"thất bại thảm hại\"?",
                    "Làm thế nào để dung hòa nhận định về sự "thất bại thảm hại" với khẳng định về hiệu quả đầu tư \"tốt nhất\"?",
                    "Liệu định nghĩa về \"thất bại\" hoặc \"khoản đầu tư tốt nhất\" đang được sử dụng có khác biệt so với thông lệ tài chính không?",
                    "Có thông tin hoặc bối cảnh nào bị thiếu có thể giúp giải thích sự mâu thuẫn rõ ràng trong nhận định này không?"
                ]
                
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
    
    def filter_rank(self,query, valid_articles):
        corpus = [str(query)] + valid_articles # Set the query to a list of string to prevent out-of-bound index dues to the corpus-size >= valid_articles-size
        # Step 1: TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Step 2: Compute cosine similarity
        query_vector = tfidf_matrix[0]  # first is query
        candidate_vectors = tfidf_matrix[1:]

        similarities = cosine_similarity(query_vector, candidate_vectors).flatten()

        # Step 3: Rank answers
        ranked_indices = similarities.argsort()[::-1]  # Descending order
        ranked_articles = [valid_articles[i] for i in ranked_indices]
        return ranked_articles

    def generate_tags(self, article, predefined_tags=None):
        tags = []
        
        # Create the prompt
        if predefined_tags:
            predefined_str = ', '.join(predefined_tags)
            prompt = f"""
            cho một bài viết sau: {article}

            hãy tạo một danh sách thẻ [CHỈ ĐƯỢC CÓ ĐÚNG 4 THẺ] (tag) với yêu cầu sau: 
            Các thẻ tạo ra phải bao gồm nội dung chính của bài táo và không được trùng lặp với nhau. 
            Chỉ chọn đúng 1 thẻ từ danh sách sau: {predefined_str}, và tạo thêm 1 đến 3 thẻ liên quan khác. 
            Xuất kết quả theo định dạng sau, chỉ bao gồm nội dung phân tích mà không thêm giải thích hoặc biểu cảm dư thừa:  
            Nghiêm cấm thêm câu đệm như "Danh sách thẻ bao gồm:", "Các thẻ bao gồm:", v.v.  
            Nghiêm cấm thêm một số chú thích không cần thiết như: (assuming current year is 2024 as article say "this year")
            ĐỘ DÀI CỦA MỘT THẺ CHỈ ĐƯỢC TỐI ĐA 3 TỪ. (Ví dụ hợp lệ: "Chính Trị", "Kinh Tế", "2022") - VÍ DỤ KHÔNG HỢP LỆ: "Chính Trị và Kinh Tế", "Chính Trị và Kinh Tế và 2022")
            ** Ví dụ định dạng: 
                thẻ1, thẻ2, thẻ3
            ---
            Mẫu Ví Dụ: (Không phải kết quả thực tế)
                Chính Trị, Kinh Tế, 2022
            """
        else:
            prompt = f"""
            cho một bài viết sau: {article}

            hãy tạo một danh sách thẻ [CHỈ ĐƯỢC CÓ ĐÚNG 4 THẺ] (tag) dựa trên nội dung của bài viết được cung cấp với yêu cầu sau:
            Các thẻ tạo ra phải bao gồm nội dung chính của bài táo và không được trùng lặp với nhau. 
            Xuất kết quả theo định dạng sau, chỉ bao gồm nội dung danh sách thẻ mà không thêm giải thích, hồi đáp hoặc biểu cảm dư thừa:
            Nghiêm cấm thêm câu đệm như "Danh sách thẻ bao gồm:", "Các thẻ bao gồm:", v.v.  
            Nghiêm cấm thêm một số chú thích không cần thiết như: (assuming current year is 2024 as article say "this year")
            ĐỘ DÀI CỦA MỘT THẺ CHỈ ĐƯỢC TỐI ĐA 3 TỪ. (Ví dụ hợp lệ: "Chính Trị", "Kinh Tế", "2022") - VÍ DỤ KHÔNG HỢP LỆ: "Chính Trị và Kinh Tế", "Chính Trị và Kinh Tế và 2022")

            ** Ví dụ định dạng: 
                thẻ1, thẻ2, thẻ3
            ---
            Mẫu Ví Dụ: (Không phải kết quả thực tế)
                Chính Trị, Kinh Tế, 2022
            """

        # Generate content from model
        response = self.model.generate_content(prompt)
        #print("res: ",response.text)
        raw_tags = response.text.strip().split(',')

        # Clean up tags
        for tag in raw_tags:
            cleaned_tag = tag.strip()
            if cleaned_tag:  # Avoid empty tags
                tags.append(cleaned_tag)

        # Optional: Ensure at least 1 predefined tag if required
        if predefined_tags:
            predefined_lower = [t.lower() for t in predefined_tags]
            has_predefined = any(tag.lower() in predefined_lower for tag in tags)

            if not has_predefined:
                fallback_tag = predefined_tags[0]  # pick the first predefined tag
                tags.insert(0, fallback_tag)

        return tags
    
    def generate_tags_batch(self, articles: List[str], predefined_tags: Optional[List[str]] = None) -> List[List[str]]:
        all_tags = []

        # Create the prompt for batch processing
        article_text = ''.join([f'Bài viết {i+1}: {article}\n' for i, article in enumerate(articles)])

        if predefined_tags:
            predefined_str = ', '.join(predefined_tags)
            prompt = (
                "cho các bài viết sau:\n\n"
                f"{article_text}\n"
                "Hãy tạo một danh sách thẻ [CHỈ ĐƯỢC CÓ ĐÚNG 4 THẺ] cho mỗi bài viết với yêu cầu sau:\n"
                "- Các thẻ phải bao gồm nội dung chính của bài viết và không được trùng lặp với nhau.\n"
                f"- Chỉ chọn đúng 1 thẻ từ danh sách sau: {predefined_str}, và tạo thêm 1 đến 3 thẻ liên quan khác.\n"
                "- Độ dài của một thẻ chỉ được tối đa 3 từ.\n\n"
                "Xuất kết quả theo định dạng sau, chỉ bao gồm nội dung phân tích mà không thêm giải thích hoặc biểu cảm dư thừa:\n"
                "Bài viết 1: thẻ1, thẻ2, thẻ3, thẻ4\n"
                "Bài viết 2: thẻ1, thẻ2, thẻ3, thẻ4\n"
                "..."
            )
        else:
            prompt = (
                "cho các bài viết sau:\n\n"
                f"{article_text}\n"
                "Hãy tạo một danh sách thẻ [CHỈ ĐƯỢC CÓ ĐÚNG 4 THẺ] cho mỗi bài viết với yêu cầu sau:\n"
                "- Các thẻ phải bao gồm nội dung chính của bài viết và không được trùng lặp với nhau.\n"
                "- Độ dài của một thẻ chỉ được tối đa 3 từ.\n\n"
                "Xuất kết quả theo định dạng sau, chỉ bao gồm nội dung phân tích mà không thêm giải thích hoặc biểu cảm dư thừa:\n"
                "Bài viết 1: thẻ1, thẻ2, thẻ3, thẻ4\n"
                "Bài viết 2: thẻ1, thẻ2, thẻ3, thẻ4\n"
                "..."
            )
        # Generate content from model
        response = self.model.generate_content(prompt)
        
        raw_responses = response.text.strip().split('\n')

        # Process each article's tags
        for raw_tags in raw_responses:
            tags = []
            if ':' in raw_tags:
                _, tag_str = raw_tags.split(':', 1)
                raw_tag_list = tag_str.strip().split(',')
                for tag in raw_tag_list:
                    cleaned_tag = tag.strip()
                    if cleaned_tag:  # Avoid empty tags
                        tags.append(cleaned_tag)

                # Optional: Ensure at least 1 predefined tag if required
                if predefined_tags:
                    predefined_lower = [t.lower() for t in predefined_tags]
                    has_predefined = any(tag.lower() in predefined_lower for tag in tags)

                    if not has_predefined and predefined_tags:
                        fallback_tag = predefined_tags[0]  # pick the first predefined tag
                        tags.insert(0, fallback_tag)

            all_tags.append(tags)

        return all_tags

    def generate_brief_descriptions_batch(self, articles: List[str]) -> List[str]:
        all_descriptions = []

        # Create the prompt for batch processing
        article_text = ''.join([f'Bài viết {i+1}: {article}\n' for i, article in enumerate(articles)])

        prompt = (
            "Dưới đây là một loạt bài viết:\n\n"
            f"{article_text}\n"
            "Hãy viết một đoạn mô tả ngắn [CHỈ ĐƯỢC từ 10 đến 50 từ] cho mỗi bài viết. Mô tả cần nêu bật nội dung chính một cách súc tích, rõ ràng và không bao gồm các yếu tố dư thừa hoặc lặp lại.\n"
            "Định dạng kết quả như sau, chỉ bao gồm nội dung mô tả, không thêm giải thích hoặc biểu cảm:\n"
            "Bài viết 1: [mô tả ngắn]\n"
            "Bài viết 2: [mô tả ngắn]\n"
            "..."
        )

        # Generate content from model
        response = self.model.generate_content(prompt)
        raw_responses = response.text.strip().split('\n')

        # Process each article's short description
        for raw_desc in raw_responses:
            if ':' in raw_desc:
                _, desc_str = raw_desc.split(':', 1)
                cleaned_desc = desc_str.strip()
                all_descriptions.append(cleaned_desc)

        return all_descriptions

    def process_articles_in_batches(self, articles: List[str], predefined_tags: Optional[List[str]] = None, batch_size: int = 5):
        all_results = []
        total_articles = len(articles)
        for i in range(0, total_articles, batch_size):
            batch = articles[i:i + batch_size]
            batch_results = self.generate_tags_batch(batch, predefined_tags)
            all_results.extend(batch_results)

            # Implement rate limiting
            if (i + batch_size) < total_articles:
                time.sleep(6)  # Sleep for 6 seconds to maintain ~10 RPM

        return all_results

    def choose_the_batch_size(self,article_list):
        """
            Input: List of Article
            Output: the Ideal batch-size
            Dynamically decide the batch size based on the number of articles.
            If the number of articles is less than or equal to 5, process them all at once.
            Otherwise, process them in batches of 5.
            This function assumes that the `article_util` module has a method `process_articles_in_batches`
            that can handle the processing of articles in batches.
        """
        total_articles = len(article_list)
        
        if total_articles == 0:
            print("No articles to process!")
            return 0
        # Dynamic batch size decision:
        if total_articles <= 5:
            batch_size = total_articles  # Put all articles in one batch
        else:
            batch_size = 5  # Max allowed batch size
        
        return batch_size
    
    
