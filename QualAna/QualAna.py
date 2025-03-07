from QualAna.GenModelUtility import GenModelUtility
import concurrent.futures
import time
class QualAnaIns():

    def __init__(self):
        print("QualAna initialized!")
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
                  

    def test(self):
        result = self.question_and_answer("giá dầu hiện nay như thế nào ?")
        print(result)


