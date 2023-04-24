import argparse
from accelerometer.utils import writeCmds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('accDir')
    parser.add_argument('--outDir', '-d', required=True)
    parser.add_argument('--cmdsFile', '-f', type=str, default='list-of-commands.txt')
    parser.add_argument('--accExt', '-a', default='cwa', help='Acc file type e.g. cwa, CWA, bin, BIN, gt3x...')
    parser.add_argument('--cmdOptions', '-x', type=str, default="",
                        help='String of processing options e.g. --epochPeriod 10')
    args = parser.parse_args()

    writeCmds(accDir=args.accDir,
              outDir=args.outDir,
              cmdsFile=args.cmdsFile,
              accExt=args.accExt,
              cmdOptions=args.cmdOptions)


if __name__ == '__main__':
    main()
