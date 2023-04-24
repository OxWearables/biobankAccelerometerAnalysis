import argparse
from accelerometer.utils import collateSummary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('resultsDir')
    parser.add_argument('--outputCsvFile', '-o', default="all-summary.csv")
    args = parser.parse_args()

    collateSummary(resultsDir=args.resultsDir,
                   outputCsvFile=args.outputCsvFile)


if __name__ == '__main__':
    main()
