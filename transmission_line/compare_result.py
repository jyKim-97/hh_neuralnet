import sys
sys.path.append("../include")
from pprint import pprint
import hhtools

cid = 8
summary_obj0 = hhtools.SummaryLoader("../gen_three_pop_samples_repr/data")

for k in summary_obj0.summary.keys():
    print(k)
    print("%.2f, %.2f, %.2f"%(summary_obj0.summary[k][cid-1,0,0],
                              summary_obj0.summary[k][cid-1,0,1],
                              summary_obj0.summary[k][cid-1,0,2]))

detail = summary_obj0.load_detail(cid-1, 0)
pprint(detail["info"][0])
