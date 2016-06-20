import os.path, subprocess
from distutils.version import LooseVersion
import sys
# check python modules
import pip
def install(package):
    pip.main(['install', package])

def check_module(moduleName, version):
	try: 
		module = __import__(moduleName,globals(), locals(), [], -1)
		ver = module.__version__
		del module
		isGoodVersion = LooseVersion(ver.strip()) >= LooseVersion(version.strip())

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
print "checking that the required python modules are installed:"
try: 
	requirements = open("./requirements.txt")
	for line in requirements:
		s = str.split(line,"==")
		if len(s)==2:
			moduleChecks.append(check_module(s[0],s[1]))

except:
	print "could not find/parse requirements.txt"
	moduleChecks.append(check_module("argparse","1.1"))
	moduleChecks.append(check_module("numpy","1.9.0"))
	moduleChecks.append(check_module("pandas","0.15.0"))
	moduleChecks.append(check_module("patsy","0.3.0"))
	moduleChecks.append(check_module("pytz","2014.7"))
	moduleChecks.append(check_module("scipy","0.15.1"))
	moduleChecks.append(check_module("statsmodels","0.6.1"))
# moduleChecks.append(check_module("virtualenv","13.1.0"))
moduleChecks = filter(lambda x: len(x)>0, moduleChecks) # filter only modules that haven't been installed

print ""
if len(moduleChecks) > 0:	
	print str(len(moduleChecks)) + " modules need installation/updating"
else:
	print "all the required modules are already installed correctly"
print ""

python_is_ok = True
if len(moduleChecks) > 0:	
	# print "would now run the following commands:"
	# for package in moduleChecks:
	# 	print "pip install " + package
	print "do you want to automatically install those " + str(len(moduleChecks)) + " modules? (type yes or no)"
	ans = raw_input().lower().strip()
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
def check_java_version():

	java_version = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)

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
	java_is_ok = check_java_version()
except: # tested on windows (throws a WindowsError)
	print "An error occured, indicating java is not installed."
	print "Error: %s: %s" % (sys.exc_info()[0] ,sys.exc_info()[1])
	java_is_ok = False

def attempt_java_install(skip_intro=False):
	if not skip_intro: print """you need to install java:\n
	 - type 'web' to open the java download page in your browser\n
	 - type 'download' to automatically download the latest java installer\n
	 - type 'skip' to skip this step and exit"""
	ans = raw_input().lower().strip()
	if ans == 'web':
		import webbrowser  
		webbrowser.open("https://java.com/en/download/manual.jsp", new=0, autoraise=True)
	elif ans == 'download':
		JRE_WIN_32 = """http://javadl.oracle.com/webapps/download/AutoDL?BundleId=210183"""
		JRE_WIN_64 = """http://javadl.oracle.com/webapps/download/AutoDL?BundleId=210185"""
		JRE_MAC    = """http://javadl.oracle.com/webapps/download/AutoDL?BundleId=207766"""
		youros = "unknown operating system"
		cpu = "unknown 32 or 64 bit" 
		from sys import platform as _platform
		if _platform == "linux" or _platform == "linux2":
		   youros = "Linux"
		   cpu = "" # doesn't matter
		elif _platform == "darwin":
		   youros = "OSX" 
		   cpu = "" # doesn't matter
		elif _platform == "win32":
		   youros = "Windows"
		   import platform
		   cpu = platform.architecture()[0]

		print "we have detected you are running %s" % (youros + cpu,)

		if youros=="Windows" or youros=="OSX":
			if youros == "Windows" and cpu=="32bit":
				java_dl_url = JRE_WIN_32
				filename = "JavaInstaller_32bit.exe"
			elif youros == "Windows" and cpu=="64bit":
				java_dl_url = JRE_WIN_64
				filename = "JavaInstaller_64bit.exe"
			elif youros == "OSX":
				java_dl_url = JRE_MAC
				filename = "JavaInstaller.dmg"
			print "attempting to download %s from %s " % (filename, java_dl_url)
			try:
				import urllib
				urllib.urlretrieve(java_dl_url, filename)
				print "downloading %s finished, now opening.." % (filename, )
				os.system(filename)
			except:
				print "download failed.."
			finally:
				if check_java_version():
					print "java is now installed correctly!"
					java_is_ok = True
					try:
						os.remove(filename)
					except:
						print "could not remove " + filename
				else:
					print "java installation failed"
					attempt_java_install()
		elif youros=="Linux":
			print """due to differences between linux versions,
			 we recommend using 'sudo apt-get install default-jre' 
			 to install java on Linux safely\n"""
			return
	elif ans == 'skip':
		return
	else:
		print "unrecognised command: " + ans
		attempt_java_install(skip_intro=True)

if not java_is_ok:
	attempt_java_install()

print "\nThis program has finished running:\n"
# final summary
if python_is_ok and java_is_ok:
	print "Your python and java setup should be able to run this program, to do so either run gui.py, or type \"python ActivitySummary.py\" into the command line."
else:
	if not python_is_ok:
		print "Your python installation is missing required modules. Either install them or use the \"Anaconda\" python distribution."
	if not java_is_ok:
		print """Your java installation is either undetected or is not a high enough version to run this program. You can download the latest version from https://www.java.com/en/download/"""
print 

ans = raw_input("press any key to exit\n")
print "you can now close this window"