import Tkinter, Tkconstants, tkFileDialog
import math
from functools import partial

# spawning child processes
import subprocess
from threading  import Thread, Timer
import sys
import time

class TkinterGUI(Tkinter.Frame):

    def __init__(self, root):

        Tkinter.Frame.__init__(self, root)

        # options for buttons
        button_opt = {'fill': Tkconstants.BOTH, 'padx': 5, 'pady': 5}
        # to know if a file has been selected
        self.targetfile = "" # file to process
        self.pycommand = "" # either ActivitySummary.py or batchProcess.py
        self.isexecuting = False # are we running a shell?
        self.threads = [] # keep track of threads so we can stop them


        frame = Tkinter.Frame()
        self.inputs = inputs = []
        # define buttons
        startingFile = ""
        inputs.append(Tkinter.Button(frame, text='Choose file', command=lambda:self.askopenfilename(startingFile = startingFile), width=35))
        inputs[-1].grid(row=0, column=0, padx=5, pady=5)
        # define options for opening or saving a file
        self.file_opt = options = {}
        # options['defaultextension'] = '.*'#'.cwa'
        options['filetypes'] = [('CWA files', '.cwa'), ('all files', '.*')]
        options['filetypes'] = [('all files', '.*'), ('CWA files', '.cwa')]
        # options['initialdir'] = 'C:\\'
        # options['initialfile'] = 'myfile.txt'
        options['parent'] = root
        options['title'] = 'Select file to process'

        # This is only available on the Macintosh, and only when Navigation Services are installed.
        #options['message'] = 'message'

        # if you use the multiple file version of the module functions this option is set automatically.
        #options['multiple'] = 1


        startingDir = ""
        inputs.append(Tkinter.Button(frame, text='Choose directory', command=lambda:self.askdirectory(startingDir = startingDir), width=35))
        inputs[-1].grid(row=0, column=1, padx=5, pady=5)
        # defining options for opening a directory
        self.dir_opt = options = {}
        # options['initialdir'] = 'C:\\'
        options['mustexist'] = False
        options['parent'] = root
        options['title'] = 'Select folder to process'

        # Textbox 
        txt = Tkinter.Text(frame, height = 10, width=80)
        txt.grid(row=1, column=0, columnspan=2, sticky=Tkconstants.N + Tkconstants.E + Tkconstants.S + Tkconstants.W   )
        Tkinter.Grid.grid_rowconfigure(frame, index=1, weight=1)
        Tkinter.Grid.grid_columnconfigure(frame, index=0, weight=1)
        Tkinter.Grid.grid_columnconfigure(frame, index=1, weight=1)
        txt.insert('insert', "Please select a file or folder")
        self.textbox = txt

        frame.pack(expand = 1, **button_opt)


        # boolean options
        self.checkboxes = {
            'skipCalibration': {'text':'Skip calibration step', 'default':False},
            'verbose': {'text':'Verbose mode', 'default':False},
            'deleteIntermediateFiles': {'text':'Delete intermediate files', 'default':True},
            'processRawFile': {'text':'Process the raw (.cwa) file', 'default':True}
        }

        frame = Tkinter.Frame()
        for key, value in self.checkboxes.iteritems():
            value['type'] = 'bool'
            value['variable'] = Tkinter.IntVar()
            value['variable'].set(value['default'])

            inputs.append(Tkinter.Checkbutton(frame, text=value['text'], variable=value['variable']))
            inputs[-1].pack(side='left',**button_opt)

        # print {key: value['variable'].get() for (key, value) in self.checkboxes.iteritems()}
        frame.pack(**button_opt)


        # more complicated options, we will just pass them in as text for now (hoping the user will put anything silly)
        self.floatboxes = {
            'calibrationOffset':{'text': 'Calibration offset', 'default':[0.0,0.0,0.0]},
            'calibrationSlope':{'text': 'Calibration slope linking offset to temperature', 'default':[1.0,1.0,1.0]},
            'calibrationTemperature':{'text': 'Mean temperature in degrees Celsius of stationary data for calibration', 'default':[0.0,0.0,0.0]},
            'meanTemperature':{'text': 'Mean calibration temperature in degrees Celsius', 'default':20.0},
            'epochPeriod':{'text': 'Length in seconds of a single epoch', 'default':5},
            'javaHeapSpace':{'text': 'Amount of heap space allocated to the java subprocesses, useful for limiting RAM usage (leave blank for no limit)', 'default':''},
            'rawDataParser':{'text': 'File containing java program to process (.cwa) binary files, must be [.class] type', 'default':'AxivityAx3Epochs'},
        }

        for key, value in self.floatboxes.iteritems():

            if isinstance(value['default'], list):
                value['type'] = 'multi'
            elif isinstance(value['default'], str):
                value['type'] = 'string'
            elif isinstance(value['default'], int):
                value['type'] = 'int'
            else:
                value['type'] = 'float'
                
            # need to make these variables pernament since if they get garbage collected tkinter will fail
            frame = Tkinter.Frame()

            value['labelvar'] = Tkinter.StringVar()
            value['labelvar'].set(value['text'])
            value['label'] = Tkinter.Label(frame, textvariable=value['labelvar'], width=50, wraplength=300)
            value['label'].pack(side='left')

            # print str(value['default'])
            value['variable'] = Tkinter.StringVar()
            value['variable'].set(self.formatargument(value['default']))

            inputs.append(Tkinter.Entry(frame,textvariable=value['variable'],width=50))
            inputs[-1].pack(side='right' , expand=1, fill=Tkconstants.X)
            frame.pack(**button_opt)

        # box for output location options
        frame = Tkinter.Frame()
        # global folder button
        # value['labelvar'] = Tkinter.StringVar()
        # value['labelvar'].set()
        value['label'] = Tkinter.Button(frame, text="Output folder for the generated summary files", width=50, wraplength=300)
        value['label'].pack(side='left')

        outputFolder = Tkinter.StringVar()
        outputFolder.set("")
        inputs.append(Tkinter.Entry(frame,textvariable=outputFolder,width=50))
        inputs[-1].pack(side='right')
        frame.pack(**button_opt)

        # Start button at bottom
        frame = Tkinter.Frame()
        self.startbutton = Tkinter.Button(frame, text='Start', width=35, command=self.start)
        self.startbutton.grid(row=0, column=0, padx=5, pady=5)
        Tkinter.Button(frame, text='Exit', width=35, command=lambda: self.quit()).grid(row=0, column=1, padx=5, pady=5)
        frame.pack()


        # merge the dicts
        self.vargs = merge_two_dicts(self.floatboxes, self.checkboxes)

        for key, value in self.vargs.iteritems():
            value['variable'].trace("w", partial(self.changed, key))

    def setCommand(self, name):

        """Replaces the text in the textbox"""

        print name
        self.textbox.delete(1.0, 'end')
        self.textbox.insert('insert', name)
        
    def generateFullCommand(self):
    
        """Generates a commandline from the options given"""
    
        if len(self.targetfile)>0 and len(self.pycommand)>0:
            # -u for unbuffered output (prevents hanging when reading stdout and stderr)
            cmdstr = "python -u " + self.pycommand
            if self.pycommand == "batchProcess.py": cmdstr += " " + self.targetfile
            
            for key,value in self.vargs.iteritems():
                if 'argstr' in value:
                    cmdstr += " " + value['argstr']

            if self.pycommand != "batchProcess.py": cmdstr += " " + self.targetfile

            self.setCommand(cmdstr)
                    
        else:
            self.setCommand("Please select a file or folder")


    def changed(self, key, *args):

        """Option button callback."""
        
        print key, args
        arg = self.vargs[key] 
        val = arg['variable'].get()
        if arg['type'] == 'bool' and val != arg['default']:
            print "bool not default"
            arg['argstr'] = '-' + key + " " + ("True" if val else "False")
        elif arg['type'] != 'bool' and  val != self.formatargument(arg['default']):
            arg['argstr'] = '-' + key + " " + val
            print "not default"
        else:
            print "is default"
            del arg['argstr']
        self.generateFullCommand()

    def formatargument(self, value):
        if isinstance(value, list):
            return ' '.join(map(str, value))
        else:
            return str(value)

    def askopenfilename(self, **args):

        """Returns a user-selected filename """
        if args['startingFile']:
            self.file_opt['initialfile'] = args['startingFile']
        filename = tkFileDialog.askopenfilename(**self.file_opt)
        if not filename:
            print "no filename returned"
        else:
            print filename
            # set for when user re-opens file dialogue
            self.file_opt['initialfile'] = filename
            self.targetfile = filename
            self.pycommand = "ActivitySummary.py"
            self.generateFullCommand()

    def askdirectory(self, **args):

        """Returns a  user-selected directoryname."""
        if args['startingDir']:
            self.dir_opt['initialdir'] = args['startingDir']
        dirname = tkFileDialog.askdirectory(**self.dir_opt)
        print dirname
        print args
        return dirname
        if not dirname:
            print "no dirname given"
        else:
            self.dir_opt['initialdir'] = dirname
            self.targetfile = dirname
            self.pycommand = "batchProcess.py"
            self.generateFullCommand()
    
    def start(self):

        """Start button pressed"""
        
        if self.isexecuting:
            self.stop()
        else:
            # cmd_line = "echo Hello!"
            cmd_line = "ping 1.0.0.0 -n 10 -w 10"
            cmd_line = self.textbox.get("1.0",Tkconstants.END)

            self.setCommand(cmd_line)
            print "running:  " + cmd_line
            self.textbox.insert("0.1", "running: ")
            p = subprocess.Popen(cmd_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)

            self.isexecuting = True
            self.startbutton['text'] = "Stop"
            t = Thread(target=self.pollStdout, args=(p,))
            t.daemon = True # thread dies with the program
            t.start()

            self.threads.append(p)
            self.enableInput(False)

    def stop(self):
        self.isexecuting = False
        while self.threads:
            p = self.threads.pop()
            t = Timer(10, p.terminate)
            t.start
            # p.terminate()# .kill()
            print p
        self.startbutton['text'] = "Start"
        self.enableInput(True)

    def pollStdout(self, p):

        """Poll the process p until it finishes"""

        start = -1 # the time when the process returns (exits)
        while True:
            retcode = p.poll() # returns None while subprocess is running
            if start==-1 and retcode is not None:
                print "thread ended", time.time()

                self.textbox.insert(Tkinter.END, "\nprocess exitied with code "+ str(retcode))
                self.textbox.see(Tkinter.END) 
                
                start = time.time()
            
            line = p.stdout.readline()

            if len(line)>0:
                print line
                self.textbox.insert(Tkinter.END, "\n" + line.rstrip())
                self.textbox.see(Tkinter.END) 

            line = p.stderr.readline()

            if len(line)>0:
                print line
                self.textbox.insert(Tkinter.END, "\nERROR: " + line.rstrip())
                self.textbox.see(Tkinter.END) 
            print "readline"

            # we stop reading after 0.5s to give error messages a chance to display
            if start != -1 and time.time() > start + 0.5:
                print "actually finished", time.time()
                self.stop()
                return
            time.sleep(0.01)

    def enableInput(self, enable=True):
        state = "normal" if enable else "disabled"
        for i in self.inputs:
            i['state'] = state


def merge_two_dicts(x, y):

    """Given two dicts, merge them into a new dict as a shallow copy."""

    z = x.copy()
    z.update(y)
    return z


if __name__=='__main__':
  root = Tkinter.Tk()
  TkinterGUI(root).pack()
  root.mainloop()
