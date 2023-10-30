import os
import glob

root_path = '/data0/JY/zwh/AllDatasets/UCF/features/rtfm-test'    ## the path of features
# root_path = '/data1/datasets/VAD/UCF/features/rtfm-train'    ## the path of features

files = sorted(glob.glob(os.path.join(root_path, "*.npy")))

large_files = ['Normal_Videos087', 'Normal_Videos136','Normal_Videos137','Normal_Videos138',
               'Normal_Videos171', 'Normal_Videos307','Normal_Videos308','Normal_Videos331',
               'Normal_Videos375', 'Normal_Videos396','Normal_Videos397','Normal_Videos449',
               'Normal_Videos450', 'Normal_Videos471','Normal_Videos472','Normal_Videos473',
               'Normal_Videos496', 'Normal_Videos520','Normal_Videos521','Normal_Videos533',
               'Normal_Videos540', 'Normal_Videos541','Normal_Videos544','Normal_Videos547',
               'Normal_Videos548', 'Normal_Videos551','Normal_Videos556','Normal_Videos563',
               'Normal_Videos565', 'Normal_Videos629','Normal_Videos633','Normal_Videos642',
               'Normal_Videos666', 'Normal_Videos687','Normal_Videos688','Normal_Videos701',
               'Normal_Videos713', 'Normal_Videos714','Normal_Videos769','Normal_Videos785',
               'Normal_Videos819', 'Normal_Videos946','Normal_Videos947','Normal_Videos449']

violents = []
normal = []
with open('zwh_rgb_test.list', 'w+') as f:  ## the name of feature list
    for file in files:
        if 'Normal' in file:
            if 'Normal_' + file.split('.')[0].split('_')[1] not in large_files:
                normal.append(file)
        else:
            newline = file+'\n'
            f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)
