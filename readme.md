## Perceptron from scratch


This script trains a perceptron for the data present in a tsv file

To run the code, give the following command
```{r, engine='python', count_lines}
python perceptron.py --data <PathToData> --output <PathToOutputFile>
```

Two data sets are given named Example.tsv and Gauss2.tsv which can be used in this program

For example, the script can be run as follows 
```{r, engine='python', count_lines}
python decisiontree.py --data Example.tsv --output yourOutputFileName.tsv
```

Output will be stored in a tsv file mentioned at the end of execution.