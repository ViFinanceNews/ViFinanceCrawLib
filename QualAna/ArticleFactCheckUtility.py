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

class ArticleFactCheckUtility():

    def __init__(self, model_name='gemini-2.0-flash-thinking-exp-01-21'):
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

            

            Mệnh đề thông tin: {statement}  

            Bằng chứng:  
            {evidence}  

            ### **Hãy trả lời theo định dạng sau:**  
            - **Tổng Hợp Cuối Cùng**: [Tóm tắt thông tin đã kiểm tra để đưa ra kết luận cuối cùng về chủ đề.]  
            - **Kết luận**: [Hỗ trợ/Mâu thuẫn/Trung lập]  
            - **Phân tích bằng chứng**: [Các dẫn chứng trên có mối liên hệ như thế nào trong việc đưa ra kết luận về vấn đề người dùng tìm hiểu]
            - **Mức độ tin cậy**: [Ví dụ: A1, B3, D5] và chú thích của mức độ [ví dụ: A1 - Đáng Tin Cậy và Đã Được Xác Minh]   
            - **Giải thích**: [Giải thích ngắn gọn về lý do của bạn, có đề cập đến nguồn bằng chứng và mức độ tin cậy của chúng. Nếu có URL trong bằng chứng, hãy chèn nó vào trong lời giải thích dưới dạng liên kết.]  
            - **Lời khuyên cho người dùng về cách nhìn nhận hiện tại**: [Một lời khuyên ngắn gọn]  
            - **Danh sách các dẫn chứng**:  
            + [1]: Tiêu đề - nguồn -  [url] \n
            + [2]: Tiêu đề - nguồn -  [url] \n
            ....
            ### **Ví dụ cách chèn liên kết:**  
            - "Bằng chứng từ [nguồn này](URL) cho thấy rằng..."  
            - "Theo thông tin từ bài viết này ([link](URL)), ..."  

            **Ví dụ phù hợp của định dạng:**
            Kết luận: Hỗ trợ  
            Mức độ tin cậy: A1  
            Giải thích: Tất cả các nguồn trong phần bằng chứng đều đề cập đến việc giá dầu tăng, hoặc các yếu tố dẫn tới/hệ quả của việc giá dầu tăng trong thời gian gần đây. Các bài viết đều từ nguồn *vneconomy.vn*, một trang tin kinh tế uy tín của Việt Nam.

            + Bài viết [Giá dầu đang gây áp lực đến lạm phát](https://vneconomy.vn/gia-dau-dang-gay-ap-luc-den-lam-phat.htm) chỉ ra rằng giá dầu đã tăng mạnh từ giữa tháng 8, với nhiều yếu tố tác động như việc cắt giảm sản lượng của Saudi Arabia và Nga, nhu cầu nhập khẩu cao của Trung Quốc, và triển vọng kinh tế khởi sắc. Bài viết cũng dự báo giá dầu có thể tiếp tục tăng trong quý 4/2023.  
            + Bài viết [OPEC+ có ảnh hưởng thế nào đến giá dầu và kinh tế toàn cầu?](https://vneconomy.vn/opec-co-anh-huong-the-nao-den-gia-dau-va-kinh-te-toan-cau.htm) giải thích vai trò của OPEC+ trong việc điều tiết nguồn cung và ảnh hưởng đến giá dầu toàn cầu. Việc cắt giảm sản lượng của OPEC+ là một yếu tố quan trọng đẩy giá dầu lên.  
            + Các bài viết còn lại đề cập các mặt hàng khác cũng tăng theo đà tăng của giá dầu.

            Danh Sách các dẫn chứng:  
            + [1]: Giá dầu đang gây áp lực đến lạm phát - Nhịp sống kinh tế Việt Nam & Thế giới -  [https://vneconomy.vn/gia-dau-dang-gay-ap-luc-den-lam-phat.htm]  
            + [2]: OPEC+ có ảnh hưởng thế nào đến giá dầu và kinh tế toàn cầu? - Nhịp sống kinh tế Việt Nam & Thế giới -  [https://vneconomy.vn/opec-co-anh-huong-the-nao-den-gia-dau-va-kinh-te-toan-cau.htm]  
            + [3]: 10 ảnh hưởng của đồng USD tăng giá mạnh - Nhịp sống kinh tế Việt Nam & Thế giới - [https://vneconomy.vn/10-anh-huong-cua-dong-usd-tang-gia-manh.htm]  
            + [4]: Lo ngoại tệ “vượt biên” vì vàng - Nhịp sống kinh tế Việt Nam & Thế giới - [https://vneconomy.vn/lo-ngoai-te-vuot-bien-vi-vang.htm]  
            + [5]: Xu thế dòng tiền: Thêm thông tin hỗ trợ, chứng khoán Việt có đi ngược thế giới? - Nhịp sống kinh tế Việt Nam & Thế giới - [https://vneconomy.vn/xu-the-dong-tien-them-thong-tin-ho-tro-chung-khoan-viet-co-di-nguoc-the-gioi.htm]  
            + [6]: Carry-trade yên Nhật thoái trào, chứng khoán toàn cầu “chịu trận” - Nhịp sống kinh tế Việt Nam & Thế giới -  [https://vneconomy.vn/carry-trade-yen-nhat-thoai-trao-chung-khoan-toan-cau-chiu-tran.htm]  
            + [7]: "Cơn sốt" giá cà phê thế giới có thể kéo dài - Nhịp sống kinh tế Việt Nam & Thế giới - [https://vneconomy.vn/con-sot-gia-ca-phe-the-gioi-co-the-keo-dai.htm]

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
        understanding_the_question(). It uses Gemini to synthesize evidence (implicitly from its knowledge)
        and summarize it into a clear and concise answer, guided by the provided reasoning.

        Args:
            query (str): The original user query.\n
            reasoning (str): The reasoning process obtained from understanding_the_question().\n
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
    
    def main(self):

        articles = [
        """
        **Bài viết 1: Tình hình kinh tế Việt Nam năm 2024**

        Năm 2024, kinh tế Việt Nam tiếp tục phát triển mạnh mẽ với GDP tăng trưởng 6,5%. Các ngành công nghiệp chủ chốt như sản xuất, dịch vụ và nông nghiệp đều ghi nhận sự tăng trưởng đáng kể. Đặc biệt, ngành công nghệ thông tin và truyền thông đã đóng góp lớn vào nền kinh tế, với nhiều startup công nghệ đạt được thành công trên thị trường quốc tế. Tuy nhiên, Việt Nam cũng đối mặt với thách thức về biến đổi khí hậu và cần có chiến lược phát triển bền vững để duy trì đà tăng trưởng.
        """,
        """
        **Bài viết 2: Sự phát triển của giáo dục trực tuyến tại Việt Nam**

        Trong những năm gần đây, giáo dục trực tuyến đã trở thành xu hướng tại Việt Nam. Với sự phát triển của công nghệ và internet, nhiều khóa học trực tuyến chất lượng cao đã được triển khai, giúp người học tiếp cận kiến thức một cách linh hoạt và tiết kiệm chi phí. Các nền tảng như Edtech Vietnam, Topica đã thu hút hàng triệu người dùng. Tuy nhiên, việc đảm bảo chất lượng và kiểm định các khóa học trực tuyến vẫn là một thách thức lớn.
        """,
        """
        **Bài viết 3: Du lịch bền vững tại Việt Nam**

        Việt Nam sở hữu nhiều danh lam thắng cảnh và di sản văn hóa phong phú, thu hút hàng triệu du khách mỗi năm. Tuy nhiên, du lịch ồ ạt đã gây ra nhiều tác động tiêu cực đến môi trường và cộng đồng địa phương. Do đó, du lịch bền vững đang trở thành xu hướng, với việc khuyến khích du khách tham gia vào các hoạt động bảo vệ môi trường, tôn trọng văn hóa địa phương và hỗ trợ kinh tế cho cộng đồng bản địa.
        """,
        """
        **Bài viết 4: Ứng dụng trí tuệ nhân tạo trong y tế Việt Nam**

        Trí tuệ nhân tạo (AI) đang được ứng dụng rộng rãi trong lĩnh vực y tế tại Việt Nam. Các bệnh viện và trung tâm y tế đã sử dụng AI để chẩn đoán hình ảnh, dự đoán bệnh tật và quản lý hồ sơ bệnh án. Ví dụ, Bệnh viện Bạch Mai đã triển khai hệ thống AI giúp chẩn đoán sớm bệnh ung thư phổi, cải thiện hiệu quả điều trị và giảm chi phí cho bệnh nhân. Tuy nhiên, việc đào tạo nhân lực và đảm bảo an toàn dữ liệu là những thách thức cần được giải quyết.
        """,
        """
        **Bài viết 5: Phát triển năng lượng tái tạo ở Việt Nam**

        Trước nhu cầu năng lượng ngày càng tăng và áp lực giảm phát thải khí nhà kính, Việt Nam đã đầu tư mạnh mẽ vào năng lượng tái tạo. Các dự án điện mặt trời và điện gió đã được triển khai tại nhiều tỉnh thành, đặc biệt là ở miền Trung và miền Nam. Chính phủ đặt mục tiêu đến năm 2030, năng lượng tái tạo sẽ chiếm 30% tổng công suất điện quốc gia. Tuy nhiên, việc tích hợp năng lượng tái tạo vào lưới điện và đảm bảo ổn định cung cấp điện là những thách thức cần được quan tâm.
        """,
        """
        **Bài viết 6: Thực trạng và giải pháp cho giao thông đô thị tại Hà Nội**

        Hà Nội, thủ đô của Việt Nam, đang đối mặt với vấn đề ùn tắc giao thông nghiêm trọng. Sự gia tăng nhanh chóng của số lượng xe cá nhân, hạ tầng giao thông chưa đáp ứng kịp và ý thức tham gia giao thông của người dân còn hạn chế là những nguyên nhân chính. Để giải quyết vấn đề này, thành phố đã triển khai nhiều giải pháp như phát triển hệ thống giao thông công cộng, xây dựng các tuyến đường vành đai và áp dụng công nghệ thông tin trong quản lý giao thông.
        """,
        """
        **Bài viết 7: Vai trò của phụ nữ trong kinh tế Việt Nam hiện đại**

        Phụ nữ Việt Nam ngày càng khẳng định vai trò quan trọng trong nền kinh tế. Họ không chỉ tham gia vào lực lượng lao động mà còn giữ nhiều vị trí lãnh đạo trong các doanh nghiệp và tổ chức. Các chương trình hỗ trợ khởi nghiệp cho phụ nữ đã giúp nhiều doanh nhân nữ thành công. Tuy nhiên, phụ nữ vẫn đối mặt với nhiều thách thức như chênh lệch thu nhập, định kiến giới và trách nhiệm gia đình.
        """,
        """
        **Bài viết 8: Ảnh hưởng của mạng xã hội đến giới trẻ Việt Nam**

        Mạng xã hội đã trở thành một phần không thể thiếu trong cuộc sống của giới trẻ Việt Nam. Nó mang lại nhiều lợi ích như kết nối, chia sẻ thông tin và giải trí. Tuy nhiên, việc sử dụng mạng xã hội quá mức cũng gây ra nhiều vấn đề như nghiện internet, giảm tương tác xã hội thực tế và ảnh hưởng đến sức khỏe tâm lý. Do đó, cần có sự hướng dẫn và giáo dục để giới trẻ sử dụng mạng xã hội một cách lành mạnh và hiệu quả.
        """,
        """
        **Bài viết 9: Bảo tồn văn hóa truyền thống trong thời kỳ hội nhập**

        Trong bối cảnh hội nhập quốc tế, việc bảo tồn và phát huy văn hóa truyền thống là một thách thức lớn đối với Việt Nam. Nhiều giá trị văn hóa đang dần bị mai một do ảnh hưởng của văn hóa ngoại lai và sự thay đổi của xã hội. Các chương trình giáo dục, lễ hội truyền thống và hoạt động cộng đồng đã được tổ chức nhằm giữ gìn và truyền bá văn hóa dân tộc cho thế hệ trẻ.
        """,
        """
        **Bài viết 10: Tác động của biến đổi khí hậu đến nông nghiệp Việt Nam**

        Biến đổi khí hậu đang ảnh hưởng nghiêm trọng đến nông nghiệp Việt Nam. Hiện tượng thời tiết cực đoan, mực nước biển dâng và sự thay đổi của mùa vụ đã gây ra nhiều khó khăn cho nông dân. Để thích ứng, nhiều biện pháp đã được áp dụng như chuyển đổi cơ cấu cây trồng, áp dụng công nghệ nông nghiệp thông minh và xây dựng hệ thống thủy lợi bền vững. Tuy nhiên, cần có sự hỗ trợ từ chính phủ và cộng đồng quốc tế để đảm bảo an ninh lương thực và sinh kế cho người dân.
        """
        ]
        result = self.process_articles_in_batches(articles, batch_size=5)
        print(result)
        print(type(result)) # a list of list
        for i, tags in enumerate(result):
            print(f"Article {i+1} Tags: {', '.join(tags)}")
        return