from QualAna.QualAna import QualAnaIns
from dotenv import load_dotenv
import time
start_time = time.time()
qual_test = QualAnaIns()
qual_test.test()
end_time = time.time()
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Elapsed time: {elapsed_time:.4f} seconds")