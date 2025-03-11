from QualAna.QualAna import QualAnaIns
from QuantAna.QuantAna import QuantAnaIns
import time
import logging

logging.basicConfig(filename="./logging/pipeline.log", level=logging.ERROR, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

start_time = time.time()
quantAna = QuantAnaIns()
testStr = "Oil prices are being manipulated by large companies"
testStr2 = "Although oil prices are currently being manipulated, they are still under government control"
# toxicity_res = quantAna.detect_toxicity(testStr)
# sentiment_ana = quantAna.sentiment_analysis(testStr)
semantic_sim = quantAna.compute_semantic_similarity(testStr, testStr2)

print(semantic_sim)

end_time = time.time()
elapsed_time = end_time - start_time  # Calculate elapsed time 

print(f"Elapsed time: {elapsed_time:.4f} seconds")

