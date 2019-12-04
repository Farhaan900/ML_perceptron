"""
Created on Nov 1, 2019

@author: Mohammed Farhaan Shaikh
"""


import pandas as pd
import sys
import getopt
import math
import numpy as np


output_data = "" # Stores the data in the xml format as a string throughout the program


def read_tsv(file_name):

    csv_data = pd.read_csv(file_name, header=None, sep='\t', engine='python')
    return csv_data


def write_to_tsv(name, data):

    f = open(name, "w")
    f.write(data)


def perceptron(inputs, weights):

    y_out = inputs * weights
    y_out = y_out.sum(axis = 1)    
    y_out = (y_out > 0).astype(int)
    
    return y_out.astype(int)

def getError(actual, predicted):
    
    count = 0
    for x in range (0,actual.size) :
        if (actual[x] - predicted[x] != 0 ):
            count = count+1
    
    return count

def reCalculateWeights(weights, learningRate, expectedValue, actualValue, data):
    
    A =  (((expectedValue.ravel() - actualValue))*learningRate)

    newWeights = [0.0,0.0,0.0]
    for x in range (0, data.shape[0]):
        newWeights += data[x] * A[x]
        
    return (weights + newWeights)

def main(argv):

    input_file_name = ""
    output_file_name = ""
    error_constant_LR = ""
    error_annealing_LR = ""

    global output_data

    ''' This part is for the command line input '''
    unix_options = "hd:o"
    gnu_options = ["help", "data=", "output="]

    try:
        arguments, values = getopt.getopt(argv, unix_options, gnu_options)
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))
        print("decisiontree.py --data <PathToData> --output <PathToOutput>")
        sys.exit(2)

    if len(arguments) < 1:
        print("ERROR in input parameters !! \nPlease use this format")
        print("decisiontree.py --data <PathToData> --output <PathToOutput>")
        sys.exit(2)

    for currentArgument, currentValue in arguments:
        if currentArgument in ("-h", "--help"):
            print("decisiontree.py --data <PathToData> --output <PathToOutput>")
            sys.exit()
        elif currentArgument in ("-d", "--data"):
            input_file_name = currentValue
        elif currentArgument in ("-o", "--output"):
            output_file_name = currentValue

    learningRate = 1
    data = read_tsv(input_file_name)
    target_col_number = 0       
    data = data.replace("A",1).replace("B",0).dropna(1)
    r, c = data.shape
    weights = np.zeros((c), dtype=float)
    y = data.iloc[:,0:1].to_numpy()
    x = data.iloc[:,1:].to_numpy()
    x = np.concatenate((np.ones((r,1), dtype=float), x), axis=1)
    
    print (" \nCalculating with constant rate...")
    for i in range (1, 102):

        y_out = perceptron(x,weights)
        error = getError(y, y_out)
        error_constant_LR+=str(error)+"\t"
        weights = reCalculateWeights(weights,learningRate,y,y_out,x)
        
    print (" \nCalculating with annealing rate...")
    t = 1
    learningRate = (1 / t)
    weights = np.zeros((c), dtype=float)
    
    for i in range (1, 102):

        y_out = perceptron(x,weights)
        error = getError(y, y_out)
        error_annealing_LR+=str(error)+"\t"
        weights = reCalculateWeights(weights,learningRate,y,y_out,x)
        t = t + 1
        learningRate = (1 / t) 
    
    print(error_constant_LR)
    print(error_annealing_LR)

    write_to_tsv(output_file_name, (error_constant_LR+"\n"+error_annealing_LR))

    print("Program complete! \nData stored in >> ", output_file_name)
    

    

if __name__ == "__main__":
    main(sys.argv[1:])
