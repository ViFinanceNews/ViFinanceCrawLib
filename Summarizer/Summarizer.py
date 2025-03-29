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
from ViFinanceCrawLib.QualAna.ArticleFactCheckUtility import ArticleFactCheckUtility

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
    
    def multi_article_synthesis(self, articles, num_sentences=4, word_limit=200):
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

        all_extractive_sentences = []

        # Step 1: Extract key sentences from each article
        for article in articles:
            sentences = self.pre_processing(article['main_text'])
            article_sum = dict(
                title = article["title"],
                url = article["url"],
                authors = article["authors"],
                date_publish = article["date_publish"]
            )
            if not sentences:
                continue
            extractive_summary = self.extractive_summary(sentences, num_sentences=num_sentences)
            article_sum["extractive_sum"] = extractive_summary
            dict_str = str(article_sum)
            print(dict_str)
            all_extractive_sentences.extend(dict_str)

        if not all_extractive_sentences:
            return "No relevant content extracted from the articles."

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
        6. Sử dụng trích dẫn trong văn bản theo định dạng **APA 7th**: **[Tên tác giả, Năm](URL)**.  
        - **Ví dụ:** ([Smith, 2020](https://example.com)) hoặc ([Nguyen & Tran, 2023](https://example.com)).  
        7. Chỉ tóm tắt, không thêm ý kiến chủ quan hay bình luận.  

        **Ví dụ về định dạng mong muốn:** (CHỈ ĐƯỢC THAM KHẢO) 

        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce varius velit ut turpis fermentum, id tincidunt orci malesuada ([Author, 2020](https://example.com)).  
        Vestibulum ac nunc in urna sodales condimentum nec id orci ([Author & Coauthor, 2021](https://example.com)).  
        Phasellus tincidunt, sapien at tristique vulputate, purus leo fringilla turpis, eu euismod nulla ligula a tortor ([Another Author, 2022](https://example.com)).  
        Integer nec turpis vitae metus tristique fermentum. Sed ut lectus vitae quam consectetur cursus ([Researcher, 2023](https://example.com)).  

        **HẾT VÍ DỤ***

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
        
    def test(self):
        articles = [
        {
            "title": "The Future of AI in Healthcare",
            "main_text": "Artificial Intelligence is transforming the healthcare industry by improving diagnostics, predicting diseases, and personalizing treatment plans. AI-powered systems analyze vast amounts of medical data, enabling early detection of conditions such as cancer and cardiovascular diseases. Machine learning algorithms help doctors make faster, more accurate diagnoses, reducing human error. Robotics and AI-driven tools assist in complex surgeries, improving precision and patient outcomes. Moreover, AI chatbots enhance patient interactions, providing instant support and medical guidance. Despite ethical concerns and data privacy issues, the potential of AI in healthcare continues to expand, promising a more efficient and accessible future.",
            "url": "https://example.com/ai-healthcare",
            "authors": ["Dr. John Smith"],
            "date_publish": "2024-01-15"
        },
        {
            "title": "Quantum Computing: A New Era",
            "main_text": "Quantum computing is revolutionizing industries by solving problems beyond classical computers' reach. Unlike traditional binary-based systems, quantum computers leverage qubits, enabling parallel computations. This technology accelerates cryptography, optimizing logistics, and advancing drug discovery. Companies like IBM and Google invest in developing scalable quantum processors, pushing computational boundaries. Researchers explore quantum algorithms to enhance AI and data security. However, challenges remain, including error correction and hardware stability. As quantum technology evolves, industries must prepare for its impact, reshaping finance, material science, and cybersecurity. The race for quantum supremacy continues, driving innovation and new technological breakthroughs worldwide.",
            "url": "https://example.com/quantum-computing",
            "authors": ["Alice Johnson", "Bob Williams"],
            "date_publish": "2024-02-10"
        },
        {
            "title": "Climate Change and Renewable Energy",
            "main_text": "Climate change remains one of the greatest challenges of our time, with rising temperatures and extreme weather events increasing globally. Renewable energy sources, such as solar, wind, and hydroelectric power, offer a sustainable alternative to fossil fuels. Governments and corporations invest in clean energy projects to reduce carbon emissions and slow global warming. Technological advancements enhance energy storage solutions, improving grid stability and efficiency. However, transitioning to renewables requires substantial investment and infrastructure upgrades. Public awareness and policy changes play a crucial role in accelerating this shift. Embracing renewable energy is vital for a cleaner and sustainable future.",
            "url": "https://example.com/climate-renewable",
            "authors": ["Emily Davis"],
            "date_publish": "2024-03-05"
        },
        {
            "title": "Advancements in Natural Language Processing",
            "main_text": "Natural Language Processing (NLP) has seen rapid advancements, enhancing machine comprehension of human language. AI-driven NLP models, such as GPT and BERT, enable chatbots, language translation, and sentiment analysis. These technologies improve search engines, automate customer service, and facilitate content creation. Pre-trained models learn from vast datasets, understanding context and semantics more accurately. However, ethical concerns arise, including biases in AI responses and misinformation risks. Researchers work to enhance NLP fairness and interpretability. With continuous improvements, NLP reshapes communication, making human-computer interactions more seamless, efficient, and intelligent across various industries, from healthcare to finance and education.",
            "url": "https://example.com/nlp-advancements",
            "authors": ["Michael Lee"],
            "date_publish": "2024-04-20"
        },
        {
            "title": "Cybersecurity Trends in 2024",
            "main_text": "Cybersecurity is a top priority as cyber threats grow in complexity. Organizations face increasing risks from ransomware, phishing, and data breaches. AI-driven security solutions enhance threat detection and response time. Zero-trust security models are gaining popularity, requiring continuous verification of user identities. Blockchain technology strengthens data security by decentralizing sensitive information. Companies invest in cybersecurity awareness training to mitigate risks from human error. Regulatory frameworks tighten, enforcing stricter compliance measures. As cybercriminals adopt advanced techniques, cybersecurity innovations must evolve to protect digital assets, ensuring businesses and individuals remain secure in an interconnected and data-driven world.",
            "url": "https://example.com/cybersecurity-trends",
            "authors": ["Sophia Martinez"],
            "date_publish": "2024-05-30"
        },
        {
            "title": "The Role of Blockchain in Finance",
            "main_text": "Blockchain technology is revolutionizing finance by enhancing transparency, security, and efficiency. Decentralized ledgers eliminate the need for intermediaries in transactions, reducing costs. Cryptocurrencies like Bitcoin and Ethereum showcase blockchain's potential in digital payments. Smart contracts enable automated and trustless agreements, streamlining business operations. Financial institutions adopt blockchain to improve cross-border transactions and fraud prevention. However, regulatory challenges and scalability issues persist. Central banks explore digital currencies (CBDCs) to modernize financial systems. As blockchain adoption grows, its impact on banking, investment, and asset management expands, reshaping traditional financial structures and fostering a more decentralized economic landscape.",
            "url": "https://example.com/blockchain-finance",
            "authors": ["Daniel Brown"],
            "date_publish": "2024-06-12"
        },
        {
            "title": "The Impact of AI on Job Markets",
            "main_text": "Artificial Intelligence is reshaping job markets by automating tasks and creating new opportunities. While AI streamlines repetitive work, concerns arise about job displacement in sectors like manufacturing and customer service. However, AI also generates demand for new roles, such as AI ethics specialists and data scientists. Upskilling and reskilling programs help workers adapt to this evolving landscape. Governments and businesses invest in AI education to bridge the skill gap. AI-human collaboration enhances productivity, allowing employees to focus on creative and strategic tasks. The future workforce will require adaptability as AI-driven transformation accelerates across industries worldwide.",
            "url": "https://example.com/ai-jobs",
            "authors": ["Olivia Wilson"],
            "date_publish": "2024-07-08"
        },
        {
            "title": "Autonomous Vehicles and Transportation",
            "main_text": "Self-driving cars are revolutionizing urban transportation by improving road safety and reducing traffic congestion. AI-powered sensors and cameras enable real-time decision-making, minimizing human errors. Autonomous vehicles (AVs) promise greater accessibility for individuals with mobility challenges. Ride-sharing services invest in AV technology, reshaping mobility services. However, regulatory and ethical challenges remain, including accident liability and cybersecurity risks. Cities must upgrade infrastructure to accommodate AV adoption. As testing and advancements continue, the future of autonomous transportation moves closer to reality. The transition to driverless technology requires collaboration between tech firms, policymakers, and the public for widespread acceptance.",
            "url": "https://example.com/autonomous-vehicles",
            "authors": ["David Clark", "Emma White"],
            "date_publish": "2024-08-22"
        },
        {
            "title": "Breakthroughs in Space Exploration",
            "main_text": "Space exploration advances with cutting-edge technology, enabling deeper cosmic exploration. NASA, SpaceX, and other agencies launch missions to Mars and beyond. Private companies innovate spacecraft and reusable rockets, reducing mission costs. The search for extraterrestrial life intensifies, with new telescopes analyzing distant planets. Lunar exploration aims at establishing sustainable bases for future space travel. Satellite technology enhances global communication and Earth monitoring. Challenges include funding constraints and space debris management. As technology progresses, interplanetary travel and asteroid mining become feasible. The future of space exploration holds promise, shaping humanity’s role in the cosmos.",
            "url": "https://example.com/space-exploration",
            "authors": ["Lucas Miller"],
            "date_publish": "2024-09-14"
        },
        {
            "title": "The Ethics of Artificial Intelligence",
            "main_text": "The ethical implications of artificial intelligence spark global debates. Bias in AI algorithms can reinforce discrimination, affecting hiring, lending, and law enforcement. Privacy concerns arise as AI systems collect vast amounts of personal data. Transparency in AI decision-making is crucial for accountability. Researchers advocate for ethical AI frameworks to prevent misuse. Governments and tech companies collaborate on AI regulations and responsible AI development. Public awareness of AI ethics grows as AI systems become more integrated into daily life. Ensuring fairness, accountability, and human-centric AI remains a critical challenge for future advancements.",
            "url": "https://example.com/ai-ethics",
            "authors": ["Sophia Garcia"],
            "date_publish": "2024-10-29"
        }
    ]

        synthesis = self.multi_article_synthesis(articles=articles)
        print("END RESULT" + "\n")
        print(synthesis)

    