"""
This command line application runs ActivitySummary.py on many files
by searching through the supplied directory
TODO: search within subdirectories
"""

import sys
import os
import subprocess

# check the python file exists
pyfile = "ActivitySummary.py"
if not os.path.isfile(os.path.abspath(pyfile)):
	print "couldn't find " + pyfile
	print "must be in same folder!"
	sys.exit(0)

def main():
	#check that enough command line arguments are entered
	msg = "\nUsage: python batchProcess.py [directory] [args]\n"
	msg += "  This script will process all files ending in .cwa in [directory]\n"
	msg += "  using ActivitySummary.py [args]\n"
	msg += "  E.g.: python batchProcess.py ./ min_freq:10 \n"
	msg += "  would process all files ending in .cwa in the current directory\n"
	msg += "  using the argument min_freq:10 on all of them\n"
	if len(sys.argv)<2:
		msg += "\nError:\n"
		msg += "  Invalid input, please enter at least 1 parameter, e.g.\n"
		msg += "  python batchProcess.py C:\\directory \n "
		print msg
		sys.exit(0)
	print msg
	# print sys.argv
	#store command line arguments to local variables
	directory = sys.argv[1]      
	if not os.path.isdir(directory):
		msg = directory + "isn't a valid directory, trying :\n"
		# to use the path of the .py file
		# directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), directory)
		# use the command line path
		directory = os.path.abspath(directory)
		msg += directory
		print msg
	if not os.path.isdir(directory):
		print """Error: argument """ + sys.argv[1] + """ is not a valid directory
				\n use "./" for the current directory, or e.g.
				\n  python batchProcess.py C:\Users\username\Documents\\biobankAccelerometerAnalysis\\
				\n"""
		sys.exit(0)

	directory_files = os.listdir(directory)
	print "found these files:"
	# print directory_files
	file_queue = []
	for file in directory_files:
		if file.endswith(".cwa"):
			print "+ " + file + ""
			file_queue.append(os.path.join(directory,file))
		else:
			print "  " + file
	num = len(file_queue)
	if num == 0:
		print "no .cwa files were found, exiting.. "
		sys.exit(0)
	print "do you want to process " + str(num) + " .cwa file" \
					+ ("s" if num!=1 else "") + "? Y/N"
	ans = raw_input()
	if not ans.lower() in ["y", "yes", "go"]:
		print "\nyou chose no. exiting.. "
	else:
		print "\nprocessing " + str(num) + "files.. \n"
		n = 0
		for file in file_queue:
			n += 1
			print "starting [" + str(n) + "/" + str(num) +"]: " + file
			if not os.path.isfile(file):
				print "file has been deleted?"
			else:
				args = ["python", pyfile, file]
				for i in range(0, len(sys.argv[2:])):
					args.append(sys.argv[2 + i])
				print "cmd: ", args
				try:
					subprocess.call(args)
				except: 
					print "there was a problem processing this file.."
		print "finished"

main()