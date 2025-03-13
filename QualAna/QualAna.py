from QualAna.GenModelUtility import GenModelUtility
import concurrent.futures
import time
from article_database.TextCleaning import TextCleaning
class QualAnaIns():

    def __init__(self):
        self.utility = GenModelUtility()

    def fact_check(self, statement):
            """Performs the Qualitative & Quantitative fact-checking process."""
            queries = self.utility.generate_search_queries(statement)
            all_evidence = []
            for query in queries:
                search_results = self.utility.search_web(query)

                for result in search_results:
                    all_evidence.append(
                    f"Source: {result['title']}\n"
                    f"Author: {result['author']}\n"
                    f"URL: {result['url']}\n"
                    f"Snippet: {result['main_text']}\n"
                )

            evidence_string = "\n\n".join(all_evidence)

            analysis_results = self.utility.analyze_evidence(statement, evidence_string)
            return analysis_results
   
    def question_and_answer(self, query):
        "Answering a question or query"
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self.utility.understanding_the_question, query)
            future2 = executor.submit(self.utility.search_web, query=query)
            
            reasonings = future1.result()
            while not future2.done():
                time.sleep(0.1)  # Small delay to prevent CPU overuse
            evidences = future2.result()
            all_evidences = []
            for evidence in evidences:
                all_evidences.append(
                    f"Source: {evidence['title']}\n"
                    f"Author: {evidence['author']}\n"
                    f"URL: {evidence['url']}\n"
                    f"Snippet: {evidence['main_text']}\n"
                )
        
        answer = self.utility.synthesize_and_summarize(query=query, reasoning=reasonings, evidence=all_evidences)
        return(answer)
                  
    def bias_analysis(self, article):
        if not article.strip():  # Check if article is empty or only contains whitespace
            return "⚠️ Bài viết không có nội dung. Vui lòng cung cấp bài viết hợp lệ để phân tích."
        analysis = self.utility.generate_bias_analysis(article)
        return analysis

    def test(self):
        tc = TextCleaning()
        test_text = """
Nếu là dân văn phòng, hẳn bạn đã quen thuộc với cảm giác tâm trạng xuống dốc vào tầm 2-3 giờ chiều.

Sau giờ nghỉ trưa, thay vì được sạc thêm năng lượng cho buổi chiều làm việc hiệu quả, bạn lại thấy uể oải, tâm trạng đờ đẫn không thể tập trung làm gì. Và trùng hợp thay, cũng cứ đúng khung giờ này là đồng nghiệp khều bạn rủ order trà sữa để "cứu mood".

Hiện tượng này thực ra có hẳn thuật ngữ riêng, là afternoon slump (cơn uể oải buổi chiều). Thậm chí các chuyên gia xã hội học đến từ Đại học Cornell (Mỹ) đã dành hẳn 2 năm để nghiên cứu về nó.

Afternoon slump là gì?
Theo Trường Y Harvard, afternoon slump là sự sụt giảm tự nhiên về năng lượng và sự tỉnh táo thường diễn ra từ 1 giờ đến 4 giờ chiều, chủ yếu do nhịp sinh học của cơ thể (circadian rhythm) và sự dao động đường huyết sau bữa trưa.

Để tìm hiểu về hiện tượng này, hai nhà khoa học Scott A. Golder và Michael W. Macy đã dùng phương pháp phân tích ngôn ngữ để phân tích 500 triệu bài đăng từ 84 quốc gia trên mạng xã hội X trong 2 năm. Đây là nghiên cứu nhằm phân tích sự biến động tâm trạng theo chu kỳ trong ngày (diurnal) và theo mùa (seasonal).

Kết quả nghiên cứu cho biết, tâm trạng chúng ta tích cực nhất vào lúc sáng sớm, tệ nhất vào buổi chiều và đỡ hơn vào buổi tối. Một nghiên cứu khác của Đại học Northeastern (Mỹ) cũng cho kết quả tương tự, chỉ rõ thời điểm chúng ta "down mood" nhất là từ 3 giờ đến 5 giờ chiều. Ngoài ra, áp lực công việc trong ngày có thể làm giảm mức độ tích cực của cảm xúc.


Afternoon slump đã "hành hạ" bạn thế nào?
Mức độ ảnh hưởng của hiện tượng này còn tùy thuộc vào nhiều yếu tố như cơ địa, điều kiện sinh hoạt, thức ăn buổi trưa và nhiều biến số khác. Tuy nhiên, đây là danh sách rút gọn những ác mộng mà bạn sẽ trải qua khi afternoon slump ập đến:

Cơ thể kiệt quệ: Bạn thấy mắt nặng trĩu như muốn sụp xuống, cơ thể rã rời dù trước đó bạn không làm gì quá sức.
Đầu óc mông lung: Bạn khó có thể tập trung suy nghĩ, gặp hiện tượng "não bỏng ngô". Đồng thời bạn cũng khó tiếp thu thông tin mới, và tỷ lệ sai sót trong công việc gia tăng.
Tâm trạng lao dốc: Bạn dễ cáu kỉnh, bực bội hoặc mất kiên nhẫn với người khác, cảm thấy thiếu động lực để tiếp tục công việc.
Cơn thèm ăn vô tận: Cụ thể là bất cứ thứ gì có đường, caffeine hay tinh bột. Hèn chi bạn khó mà cưỡng lại kèo rủ order trà sữa của đồng nghiệp.
Nguyên nhân dẫn đến afternoon slump?
Để hiểu cơ chế hoạt động của afternoon slump, bạn cần biết về cách thức hoạt động của nhịp sinh học. Đó là những thay đổi về thể chất, tinh thần và hành vi theo chu kỳ hàng ngày, chủ yếu phản ứng với ánh sáng và bóng tối trong môi trường.

Hạch trên chéo (suprachiasmatic sucleus - SCN) trong vùng dưới đồi của não là bộ điều khiển chính của nhịp sinh học. SCN nhận tín hiệu từ ánh sáng mặt trời, điều chỉnh melatonin (hormone gây buồn ngủ) và cortisol (hormone giúp tỉnh táo) để giữ nhịp sinh học hoạt động ổn định.

Trong thời gian thực hiện hai nghiên cứu lần lượt vào các năm 1999 và 2009, Giáo sư Charles A. Czeisler cùng đồng nghiệp đã phát hiện mức cortisol của con người thường tăng cao vào 6 giờ đến 8 giờ sáng để giúp cơ thể tỉnh táo tự nhiên. Sau đó, chúng ta sẽ bước vào giai đoạn tỉnh táo nhất từ 9 giờ đến 12 giờ trưa.

Đến buổi chiều (1 giờ đến 4 giờ), sự suy giảm cortisol diễn ra tự nhiên trong cơ thể, và ảnh hưởng từ các chất trong bữa trưa khiến bạn trở nên “đuối” dần. Hiện tượng này sẽ trầm trọng hơn nếu đêm trước đó bạn không ngủ đủ giấc. Khi đêm đến, tuyến tùng (pineal gland) sẽ tiết melatonin để chuẩn bị đưa cơ thể vào trạng thái nghỉ ngơi. Nếu không giải trừ ảnh hưởng của afternoon slump kịp lúc, bạn sẽ trong tình trạng ngẩn ngơ, đờ đẫn.

Cách phòng bị trước afternoon slump?
Do afternoon slump là cơ chế tự nhiên của cơ thể, bạn cũng không hoàn toàn tránh được hiện tượng này. Tuy nhiên vẫn có đôi mẹo giúp bạn hạn chế tác động của nó để tập trung hơn cho công việc:

Tránh ăn trưa quá no: Hạn chế thực phẩm nhiều đường và tinh bột, vì nó sẽ khiến bạn buồn ngủ.
Ngủ trưa ngắn (10 - 20 phút): Giúp bạn tỉnh táo hơn vào buổi chiều mà không gây uể oải.
Vận động & thay đổi môi trường: Đi bộ, đứng làm việc hoặc nghe nhạc để kích thích tinh thần.
Dùng caffeine đúng cách: Thử uống một ly cafe ngay trước khi chợp mắt khoảng 15–20 phút. Như vậy khi bạn tỉnh dậy, caffeine bắt đầu có tác dụng, giúp bạn thấy sảng khoái và tỉnh táo hơn so với chỉ uống cafe hoặc chỉ ngủ ngắn.
Tiếp xúc với ánh sáng tự nhiên: Giúp điều chỉnh nhịp sinh học và tăng sự tỉnh táo.
Bắt đầu ngày với thói quen lành mạnh: Tập thể dục nhẹ, không bỏ bữa sáng để tránh sụt giảm năng lượng vào chiều.
Làm việc theo chu kỳ ngắn (pomodoro): Chia công việc thành các phiên 25-45 phút để tối ưu hiệu suất
            """
        # test_text = tc.clean_text(text_str=test_text)
        result = self.bias_analysis(test_text)
        print(result)


