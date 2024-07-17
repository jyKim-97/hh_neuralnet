import utils
import sys
from tqdm import tqdm
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')

import hhtools

summary_obj = hhtools.SummaryLoader("../gen_three_pop_samples_repr/data_mfast", load_only_control=True)

for nc in range(summary_obj.num_controls[0]):
    for nt in tqdm(range(summary_obj.num_controls[1]), desc="#%d"%(nc)):
        detail = summary_obj.load_detail(nc, nt)
        utils.export_mua(detail, dt=0.01, st=0.001)
        