import cma
import glob
import numpy as np
import os
import fuse, causal_utils
model_path='/raid/can/nli_models/baseline_mind_distill/'
task='nli'
data_path='/ist/users/canu/debias_nlu/data/' + task + '/'
fusion = fuse.sum_fuse
test_set='test'
cma.report_CMA(model_path,task,data_path,test_set,fusion)