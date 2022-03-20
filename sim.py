import warnings
from COC import COCPatternRecognition
from time import time
import torch
import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

torch.set_num_threads(8)

dateString = str(datetime.datetime.now())
batchNum = 32
epochs = 200
patternNum = 2
type = "full"

COC = COCPatternRecognition(dateString, batchNum, epochs, type, patternNum)
COC.build_from_file("49osc_2pattern.in")
start = time()
COC.run_simulation()
end = time()
print("The whole simulation took %.0fs to complete!" % (end-start))
COC.plot_LC_w_params("MLResult_Loss_")
COC.couplings_to_file("MLparams_")
COC.save_results("MLResult_data_")
COC.plot_graph("Graph_")
# COC.test_network()

if not COC.save:
    plt.show()
