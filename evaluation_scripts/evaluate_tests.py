import numpy as np
import os

def parse(f,_dict):
    
    key = ""
    for line in f:

        _line = line.split()
        
        if "pair" in _line[0]:
            key = _line[0][:-4]
            _dict[key] = {}

        if key != "":
            for idx in range(len(_line)):
                if _line[idx] == 'mean':
                    _dict[key]['mean'] = float(_line[idx+2])
                if _line[idx] == 'upper':
                    _dict[key]['upper'] = float(_line[idx+2])
                if _line[idx] == "lower":
                    _dict[key]['lower'] = float(_line[idx+2])

    return _dict


if __name__ == "__main__":

    discretized = ['pair0005', 'pair0006', 'pair0007', 'pair0008', 'pair0009', 'pair0010', 'pair0011', 'pair0012', 'pair0013', 'pair0014', 'pair0016', 'pair0017', 'pair0022', 'pair0023', 'pair0024', 'pair0025', 'pair0026', 'pair0027', 'pair0028', 'pair0029', 'pair0030', 'pair0031', 'pair0032', 'pair0033', 'pair0034', 'pair0035', 'pair0036', 'pair0037', 'pair0038', 'pair0039', 'pair0040', 'pair0041', 'pair0042', 'pair0043', 'pair0044', 'pair0045', 'pair0046', 'pair0047', 'pair0064', 'pair0068', 'pair0069', 'pair0070', 'pair0076', 'pair0085', 'pair0086', 'pair0087', 'pair0094', 'pair0095', 'pair0096', 'pair0097', 'pair0098', 'pair0099', 'pair0100', 'pair0101', 'pair0107']

    #my_output = 'tcep_no_des_tests/second_run_acc_61/N_bins512/backup2/'
    my_output = 'tcep_subsampled_tests/'

    files = os.listdir(my_output)
    files.sort()

    meta_file = 'benchmark_tests/tcep/pairmeta.txt'
    
    # Read in from 'my_output', where the results with evidences
    # are stored and then compare with the results from the 'meta_file'

    file_XY = []
    file_YX = []

    for file in files:
        if "X->Y" in file:
            file_XY.append(file)
        if "Y->X" in file:
            file_YX.append(file)
    
    dict_XY = {}
    dict_YX = {}
    for XY, YX in zip(file_XY, file_YX):
        fXY = open(my_output + XY, 'r')
        fYX = open(my_output + YX, 'r')
        
        dict_XY = parse(fXY, dict_XY)
        dict_YX = parse(fYX, dict_YX)

    my_direction = []
    keys = []
    for key in dict_XY.keys():
        if not (key in dict_YX.keys()):
            print(key)
            raise ValueError
        if not (key in discretized):
            if dict_XY[key]['mean'] >= dict_YX[key]['mean']:
                my_direction.append('X->Y')
            else:
                my_direction.append('Y->X')
            keys.append(key)

    true_direction = []
    weight = []
    total_weight = 0
    for line in open(meta_file, 'r'):
        _line = line.split()
        
        if ("pair" + _line[0]) in dict_XY.keys() and \
           not (("pair" + _line[0]) in discretized):
            tmp = [float(x) for x in _line[1:]]
            # Assuming here there are no directions except 'X->Y'
            # and 'Y->X'
            if \
                tmp[0]==1 and tmp[1] ==1 and tmp[2]==2 and tmp[3]==2:
                true_direction.append('X->Y')
            elif \
                tmp[0]==2 and tmp[1]==2 and tmp[2]==1 and tmp[3]==1:
                true_direction.append('Y->X')
            total_weight += tmp[-1] 
            weight.append(tmp[-1])

    accuracy = 0
    for i in range(len(true_direction)):
        if true_direction[i]==my_direction[i]:
            accuracy += weight[i]
        else:
            print("Not this one")
            print(keys[i])

    print("Total accuracy: {:03f}".format(accuracy/total_weight))
