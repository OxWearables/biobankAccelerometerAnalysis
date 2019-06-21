"""Command line tool to collate multiple confusion matrices *.txt into single .csv"""

import argparse
import os
import pandas as pd
import sys

parser = argparse.ArgumentParser(
        description="Collate multiple confusion matrices *.txt into single .csv",
        add_help=True
    )
# inputs
parser.add_argument('--matrixDIR', type=str, default="activityModels/",
        help="input dir with confusion matrix txt files")
# outputs
parser.add_argument('--outCSV', type=str, help="output main CSV matrix file",
        default="collatedMatrix.csv")
args = parser.parse_args()

phenoOrder = {'sleep':1, 'sedentary':2, 'tasks-light':3, 'walking':4, 'moderate':5}

def main():
    bigMatrix = None
    
    # combine confusion matrices of all participants
    all_files = [e for e in os.listdir(args.matrixDIR) if e.endswith('.txt') and e.startswith('confusion')]
    for pidMatrix in sorted(all_files):
        if bigMatrix is None:
            bigMatrix = pd.read_csv(args.matrixDIR + pidMatrix)
        else:
            userMatrix = pd.read_csv(args.matrixDIR + pidMatrix)
            bigMatrix = bigMatrix + userMatrix
    
    # rename y_true column as it contains many appended string duplicates e.g. sleepsleepsleepsleep...
    for state in phenoOrder.keys():
        bigMatrix.loc[bigMatrix['y_true'].str.contains(state), 'y_true'] = state
    bigMatrix['stateOrder'] = bigMatrix['y_true'].replace(phenoOrder)
    bigMatrix = bigMatrix.set_index('y_true')
    bigMatrix = bigMatrix.sort_values('stateOrder')
    outCols = bigMatrix.index.tolist()
    print(bigMatrix[outCols])
    
    bigMatrix[outCols].to_csv(args.outCSV)
    print('\n\nfinished')


if __name__ == '__main__':
    main()
