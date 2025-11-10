import argparse
from accelerometer.utils import write_cmds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('accDir')
    parser.add_argument('--outDir', '-d', required=True)
    parser.add_argument('--cmdsFile', '-f', type=str, default='list-of-commands.txt')
    parser.add_argument('--accExt', '-a', default='cwa', help='Acc file type e.g. cwa, CWA, bin, BIN, gt3x...')
    parser.add_argument('--cmdOptions', '-x', type=str, default="",
                        help='String of processing options e.g. --epochPeriod 10')
    args = parser.parse_args()

    write_cmds(acc_dir=args.accDir,
               out_dir=args.outDir,
               cmds_file=args.cmdsFile,
               acc_ext=args.accExt,
               cmd_options=args.cmdOptions)


if __name__ == '__main__':
    main()
