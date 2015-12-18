"""
This command line application runs ActivitySummary.py on many files

"""

import sys

def main():
	print sys.argv
	#check that enough command line arguments are entered
    msg = " Usage: python batchProcess [directory]"
    msg += "\n e.g.: python batchProcess /"
    msg += "\n would process all files ending in .cwa in the current directory"
	if len(sys.argv)<2:
	    msg += "\n Invalid input, please enter at least 1 parameter, e.g."
	    msg += "\n python batchProcess.py inputFile.CWA \n"
	    print msg
	    sys.exit(0)
	print msg
	#store command line arguments to local variables
	rawFile = sys.argv[1]      
	funcParams = sys.argv[2:]

	os.listdir(sys.argv[2])
	