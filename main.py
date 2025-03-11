from QualAna.QualAna import QualAnaIns
from QuantAna.QuantAna import QuantAnaIns
import time
import logging

logging.basicConfig(filename="./logging/pipeline.log", level=logging.ERROR, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

start_time = time.time()
quantAna = QuantAnaIns()
testStr = "Giá dầu tăng nhanh do các tập đoàn thao túng"
testStr2 = "Giá dầu những năm nay đang có xu hướng tăng nhanh do nhiều điều kiện tác động"
# toxicity_res = quantAna.detect_toxicity(testStr)
# sentiment_ana = quantAna.sentiment_analysis(testStr)
semantic_sim = quantAna.compute_semantic_similarity(testStr, testStr2)

print(semantic_sim)

end_time = time.time()
elapsed_time = end_time - start_time  # Calculate elapsed time 

print(f"Elapsed time: {elapsed_time:.4f} seconds")

