import os.path, subprocess
from distutils.version import LooseVersion
import sys

print(sys.executable)
# check python modules

import pip
def install(package):
    pip.main(['install', package])

def check_module(moduleName, version):
	try: 
		module = __import__(moduleName,globals(), locals(), [], -1)
		ver = module.__version__
		del module
		isGoodVersion = LooseVersion(ver) >= LooseVersion(version)
		if isGoodVersion:
			print moduleName + " version :" + str(ver) + " >= " + version
			return ""
		else:
			print moduleName + " version :" + str(ver) + " <  " + version + " (needs update!)"
			return moduleName + "==" + version
	except ImportError:
		print moduleName + " not installed"
		return moduleName + "==" + version

moduleChecks = []
print "now checking for required python modules:"
moduleChecks.append(check_module("argparse","1.4.0"))
moduleChecks.append(check_module("numpy","1.9.0"))
moduleChecks.append(check_module("pandas","0.15.0"))
moduleChecks.append(check_module("patsy","0.3.0"))
# moduleChecks.append(check_module("python-dateutil","2.2"))
moduleChecks.append(check_module("pytz","2014.7"))
moduleChecks.append(check_module("scipy","0.15.1"))
moduleChecks.append(check_module("simplejson","3.8.0"))
moduleChecks.append(check_module("six","1.8.0"))
moduleChecks.append(check_module("statsmodels","0.6.1"))
# moduleChecks.append(check_module("virtualenv","13.1.0"))
moduleChecks = filter(lambda x: len(x)>0, moduleChecks) # filter only modules that
print ""
print str(len(moduleChecks)) + " modules need installation/updating\n"

python_is_ok = True
if len(moduleChecks) > 0:	
	print "would now run the following commands:"
	for package in moduleChecks:
		print "pip install " + package
	print "do you want to install those " + str(len(moduleChecks)) + " modules? (type yes)"
	ans = raw_input()
	if not ans.lower() in ["yes"]:
		print "\nyou chose not to, continuing.. "
		python_is_ok

	else:
		for package in moduleChecks:
			try:
				pip.main(['install', package])
			except:
				print "Unable to install %(package)s using pip. Please read the instructions for \
				manual installation.. "
				print "Error: %s: %s" % (exc_info()[0] ,exc_info()[1])
				python_is_ok = False


# check java version
def check_java_version(java_version):
	if java_version.startswith("java version \""):
		beginquote = java_version.find( "\"")+1
		endquote = java_version.find( "\"", beginquote)
		ver = java_version[beginquote:endquote]
		isGoodVersion = LooseVersion(ver) >= LooseVersion("1.8.0_60")
		if isGoodVersion:
			print "java version is : " + ver + " >= 1.8.0_60 (good)"
			return True
		else :
			print "java version is : " + ver + " < 1.8.0_60 (must be updated!)"
	return False


print "\nrunning command : java -version"
try:
	java_version = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)
	print java_version
	java_is_ok = check_java_version(java_version)
except: # tested on windows (throws a WindowsError)
	print "An error occured, indicating java is not installed."
	print "Error: %s: %s" % (sys.exc_info()[0] ,sys.exc_info()[1])
	java_is_ok = False

print "\nIn Summary:\n"
# final summary
if python_is_ok and java_is_ok:
	print "Your python and java setup should be able to run this program, to do so type \"python ActivitySummary.py\" into the command line."
else:
	if not python_is_ok:
		print "Your python installation is missing required modules. Either install them or use the \"Anaconda\" python distribution."
	if not java_is_ok:
		print """Your java installation is probably either undetected or is not a high enough version to run this program. You can download the latest version from https://www.java.com/en/download/"""

print 
raw_input("press any key to exit\n")
