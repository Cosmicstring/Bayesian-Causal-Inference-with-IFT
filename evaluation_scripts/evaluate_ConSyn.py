import numpy as np
import os

import nifty6 as ift

from evaluate_tests import parse

def get_mean_std(_dict):

    loc = float(_dict['mean'])

    std1 = float(_dict['upper']) - loc
    std2 = float(_dict['lower']) - loc

    std = max(std1, std2)

    return loc, std

def eval_direction(dict_xy, dict_yx, dict_conf, N=int(1e5)):

    locxy, stdxy = get_mean_std(dict_xy)
    locyx, stdyx = get_mean_std(dict_yx)
    locconf, stdconf = get_mean_std(dict_conf)

    # Draw 10^6 random samples
    samplesxy = np.random.normal(locxy, stdxy, N)
    samplesyx = np.random.normal(locyx, stdyx, N)
    samplesconf = np.random.normal(locconf, stdconf, N)

    # Set initial count to 1.0, for numerical errors
    # with the 'log' later
    scorexy, scoreyx, scoreconf = 1,1,1

    for sxy, syx, sconf in zip(samplesxy, samplesyx, samplesconf):

        _max = max(max(sxy, syx), sconf)

        if sxy == _max:
            scorexy +=1
        if syx == _max:
            scoreyx +=1
        if sconf == _max:
            scoreconf +=1

    # Divide by N+3 since above scores are set initially to 1
    # which accounts for 1 bonus sample for each dist
    total = N+3

    scorexy, scoreyx, scoreconf = scorexy / total, scoreyx / total, scoreconf / total

    lossxy, lossyx, lossconf = -np.log2(scorexy), -np.log2(scoreyx), -np.log2(scoreconf)

    min_loss = min(min(lossxy, lossyx), lossconf)

    if lossxy == min_loss:
        return "X->Y"
    elif lossyx == min_loss:
        return "Y->X"
    else:
        return "X<-Z->Y"

if __name__ == "__main__":

    my_output = '../ConSyn_tests/N_bins512/v1_vs_conf/'

    files = os.listdir(my_output)

    # Sort by batches
    files.sort(key = lambda x: x[-19:-13])

    meta_file = '../benchmark_tests/ConSyn/pairmeta.txt'

    # Read in from 'my_output', where the results with evidences
    # are stored and then compare with the results from the 'meta_file'

    file_XY = []
    file_YX = []
    file_conf = []

    for file in files:
        if "X->Y" in file:
            file_XY.append(file)
        if "Y->X" in file:
            file_YX.append(file)
        if "X<-Z->Y" in file:
            file_conf.append(file)

    dict_XY = {}
    dict_YX = {}
    dict_conf = {}
    for XY, YX, conf in zip(file_XY, file_YX, file_conf):
        with open(my_output + XY, 'r') as f:
            dict_XY = parse(f, dict_XY)

        with open(my_output + YX, 'r') as f:
            dict_YX = parse(f, dict_YX)

        with open(my_output + conf, 'r') as f:
            dict_conf = parse(f, dict_conf)

    if not (dict_conf.keys()==dict_XY.keys() and dict_conf.keys()==dict_YX.keys()):
        raise ValueError("Keys are not in same ordering! Comparing different testcases with eachother!")

    my_direction = []
    keys = []
    for key in dict_XY.keys():
        if not (key in dict_YX.keys()):
            print(key)
            raise ValueError

        my_direction.append(\
            eval_direction(dict_XY[key], dict_YX[key], dict_conf[key]))
        keys.append(key)

    true_direction = []
    weight = []
    total_weight = 0
    for line in open(meta_file, 'r'):
        _line = line.split()
        print(_line)
        if (_line[0]) in dict_XY.keys():
            tmp = [float(x) for x in _line[1:]]
            # Assuming here there are no directions except 'X->Y'
            # and 'Y->X'
            if \
                tmp[0]==1 and tmp[1] ==1 and tmp[2]==2 and tmp[3]==2:
                true_direction.append('X->Y')
            elif \
                tmp[0]==2 and tmp[1]==2 and tmp[2]==1 and tmp[3]==1:
                true_direction.append('Y->X')
            elif tmp[0]==1 and tmp[1]==2 and tmp[2]==3 and tmp[3]==3:
                true_direction.append("X<-Z->Y")
            total_weight += tmp[-1]
            weight.append(tmp[-1])

    print(true_direction)
    accuracy = 0
    for i in range(len(true_direction)):
        if true_direction[i]==my_direction[i]:
            accuracy += weight[i]
        else:
            print("Not this one")
            print(keys[i])

    print("Total accuracy: {:03f}".format(accuracy/total_weight))
