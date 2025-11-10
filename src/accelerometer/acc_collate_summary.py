import argparse
from accelerometer.utils import collate_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('resultsDir')
    parser.add_argument('--outputCsvFile', '-o', default="all-summary.csv")
    args = parser.parse_args()

    collate_summary(results_dir=args.resultsDir,
                    output_csv_file=args.outputCsvFile)


if __name__ == '__main__':
    main()
