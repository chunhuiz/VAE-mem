import os
import glob
data_path = '/data0/JY/zwh/AllDatasets/SHT_fea_refine.h5'
with open('zwh_rgb.list', 'w+') as f:  ## the name of feature list
    f.write(data_path)
