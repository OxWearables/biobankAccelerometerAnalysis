import Tkinter
import Tkconstants, tkFileDialog
import ttk
# spawning child processes
import subprocess
from threading  import Thread, Timer
import time


class TkinterGUI(Tkinter.Frame):

    def __init__(self, root):

        Tkinter.Frame.__init__(self, root)
        root.title("Accelerometer Processing")
        # options for buttons
        pack_opts = {'fill': Tkconstants.BOTH, 'padx': 5, 'pady': 5}
        # to know if a file has been selected
        self.targetfile = ""  # file to process
        self.pycommand = ""  # either ActivitySummary.py or batchProcess.py
        self.isexecuting = False  # are we running a shell process?
        self.threads = []  # keep track of threads so we can stop them

        self.target_opts = {
            'filename':Tkinter.StringVar(), # we use stringVar so we can monitor for changes
            'dirname':Tkinter.StringVar(),
            'filenames': [], # for multiple file selections, unused due to dodgy Tkinter support
            'target_type': Tkinter.StringVar(),
            'file_opts': {
                'filetypes': [('all files', '.*'), ('CWA files', '.cwa')],
                'parent':root,
                'title': 'Select a file to process'
            },
            'dir_opts': {
                'mustexist': False,
                'parent': root,
                'title': 'Select a folder to process'
            }
        }

        def target_callback(type, *args):
            print "target_callback", type
            self.target_opts['target_type'].set(type)
            self.generateFullCommand()

        self.target_opts['filename'].trace("w", lambda *args: target_callback('filename', *args))
        self.target_opts['dirname'].trace("w", lambda *args: target_callback('dirname', *args))
        # vargs are in format {'command': name of -command, 'varable': StringVar() for value,
        # 'default': if same as default it's unchanged, 'type': 'bool', 'string', 'int', or 'float' }
        # -[command] [variable].get()   (only add if [variable] != [default])
        self.vargs = []
        # list of buttons that should be disabled when the processing has started
        self.inputs = []

        advanced_frame = Tkinter.Frame()
        frame = Tkinter.Frame(advanced_frame)
        # define buttons
        self.inputs.append(
            Tkinter.Button(frame,
                text='Choose file',
                command=lambda:self.target_opts['filename'].set(self.askopenfilename(initialFile = self.target_opts['filename'].get())),
                width=35))

        self.inputs[-1].grid(row=0, column=0, padx=5, pady=15)

        # define options for opening or saving a file
        self.file_opt = {
            'filetypes': [('all files', '.*'), ('CWA files', '.cwa')],
            'parent': root,
            'title': 'Select file to process'
        }
        # options['defaultextension'] = '.*'#'.cwa'

        # if you use the multiple file version of the module functions this option is set automatically.
        # options['multiple'] = 1

        self.inputs.append(
            Tkinter.Button(frame,
                text='Choose directory',
                command=lambda: self.target_opts['dirname'].set(
                        self.askdirectory(initialDir=self.target_opts['dirname'].get())),
                width=35))
        self.inputs[-1].grid(row=0, column=1, padx=5, pady=5)
        # defining options for opening a directory
        self.dir_opt = {
            'mustexist': False,
            'parent': root,
            'title': 'Select folder to process'
        }

        self.advancedOptionInputs = []
        self.showAdvancedOptions = False
        self.advancedOptionsButton = Tkinter.Button(frame,
                                        text='Hide advanced Options',
                                        command=self.toggleAdvancedOptions,
                                        width=35)
        self.advancedOptionsButton.grid(row=0, column=2, padx=5, pady=5)

        # Textbox
        self.textnbframe = ttk.Notebook(frame)  # for tabs above textbox
        page1 = ttk.Frame(self.textnbframe)
        page2 = ttk.Frame(self.textnbframe)
        self.textnbframe.add(page1, text='Command')
        self.textnbframe.add(page2, text='Files to process')

        self.textbox = Tkinter.Text(page1, height=10, width=70)
        self.textbox.pack(expand=1, **pack_opts)
        Tkinter.Grid.grid_rowconfigure(frame, index=1, weight=1)
        Tkinter.Grid.grid_columnconfigure(frame, index=0, weight=1)
        Tkinter.Grid.grid_columnconfigure(frame, index=1, weight=1)
        Tkinter.Grid.grid_columnconfigure(frame, index=2, weight=1)
        self.textbox.insert('insert', "Please select a file or folder")


        self.textnbframe.grid(row=1, column=0, columnspan=3,
                              sticky=Tkconstants.N + Tkconstants.E + Tkconstants.S + Tkconstants.W,
                              padx=5, pady=5)

        frame.pack(expand=1, **pack_opts)

        # boolean options
        self.checkboxes = {
            'skipCalibration': {'text':'Skip calibration step', 'default':False},
            'verbose': {'text':'Verbose mode', 'default':False},
            'deleteIntermediateFiles': {'text':'Delete intermediate files', 'default':True},
            'processRawFile': {'text':'Process the raw (.cwa) file', 'default':True}
        }

        frame = Tkinter.Frame(advanced_frame)
        for key, value in self.checkboxes.iteritems():
            value['type'] = 'bool'
            value['variable'] = Tkinter.IntVar()
            value['variable'].set(value['default'])

            self.vargs.append({'command': key, 'variable': value['variable'], 'default': value['default'], 'type':'bool'})

            self.inputs.append(Tkinter.Checkbutton(frame, text=value['text'], variable=value['variable']))
            self.inputs[-1].pack(side='left',**pack_opts)

        # print {key: value['variable'].get() for (key, value) in self.checkboxes.iteritems()}
        frame.pack(fill=Tkconstants.NONE, padx=pack_opts['padx'], pady=pack_opts['pady'])
        self.advancedOptionInputs.append({'frame':frame, 'pack_opts': frame.pack_info()})

        # more complicated options, we will just pass them in as text for now (hoping the user will put anything silly)
        option_groups = {
            'Calibration options': {
                'calibrationOffset': {'text': 'Calibration offset', 'default':[0.0,0.0,0.0]},
                'calibrationSlope': {'text': 'Calibration slope linking offset to temperature', 'default':[1.0,1.0,1.0]},
                'calibrationTemperature': {'text': 'Mean temperature in degrees Celsius of stationary data for calibration', 'default':[0.0,0.0,0.0]},
                'meanTemperature': {'text': 'Mean calibration temperature in degrees Celsius', 'default':20.0}
            },
            'Java options': {
                'javaHeapSpace': {'text': 'Amount of heap space allocated to the java subprocesses, useful for limiting RAM usage (leave blank for no limit)', 'default':''},
                'rawDataParser': {'text': 'Java (.class) file, which is used to process (.cwa) binary files', 'default':'AxivityAx3Epochs'}
            },
            'Epoch options': {
                'epochPeriod': {'text': 'Length in seconds of a single epoch', 'default': 5}
            }
        }
        frame = Tkinter.Frame(advanced_frame)
        for key, groups in option_groups.iteritems():
            labelframe = Tkinter.LabelFrame(frame, text=key)
            for key, value in groups.iteritems():

                if isinstance(value['default'], list):
                    value['type'] = 'multi'
                elif isinstance(value['default'], str):
                    value['type'] = 'string'
                elif isinstance(value['default'], int):
                    value['type'] = 'int'
                else:
                    value['type'] = 'float'

                # need to make these variables pernament since if they get garbage collected tkinter will fail
                rowFrame = Tkinter.Frame(labelframe)

                value['labelvar'] = Tkinter.StringVar()
                value['labelvar'].set(value['text'])
                value['label'] = Tkinter.Label(rowFrame, textvariable=value['labelvar'], width=50, wraplength=300)
                value['label'].pack(side='left')

                # print str(value['default'])
                value['variable'] = Tkinter.StringVar()
                value['variable'].set(self.formatargument(value['default']))

                self.vargs.append({'command': key, 'variable': value['variable'], 'default': value['default'], 'type':value['type']})

                self.inputs.append(Tkinter.Entry(rowFrame,textvariable=value['variable'],width=50))
                self.inputs[-1].pack(side='right' , expand=1, fill=Tkconstants.X)
                rowFrame.pack(**pack_opts)
            labelframe.pack(**pack_opts)
        frame.pack()

        self.advancedOptionInputs.append({'frame': frame, 'pack_opts': frame.pack_info() })
        folder_params = {
            'summaryFolder': {'text': 'Folder for summary output'},
            'nonWearFolder': {'text': 'Folder for non-wear time'},
            'epochFolder': {'text': 'Folder for epoch.json'},
            'stationaryFolder': {'text': 'Folder for stationary non-wear bouts'},
            'timeSeriesFolder': {'text': 'Folder for time series'}
        }

        # box for output folder options
        def chooseFolder(value):

            chosendir = self.askdirectory(initialDir=value.get())
            if chosendir.find(" ") != -1:
                value.set('\"' + chosendir + '\"')
            else:
                value.set(chosendir)

        frame = Tkinter.Frame(advanced_frame)
        labelframe = Tkinter.LabelFrame(frame, text="Folder options (default is same folder as input file)")
        for key, value in folder_params.iteritems():
            rowFrame = Tkinter.Frame(labelframe)

            value['variable'] = Tkinter.StringVar()
            value['variable'].set("")
            value['variable'].trace('w', lambda *args: self.generateFullCommand())

            self.vargs.append({'command': key, 'variable': value['variable'], 'default': '', 'type':'string'})

            self.inputs.append(Tkinter.Entry(rowFrame,textvariable=value['variable'],width=50))
            self.inputs[-1].pack(side='right', expand=1, fill= Tkconstants.X)

            value['label'] = Tkinter.Button(rowFrame, text=value['text'], command=lambda v=value['variable']: chooseFolder(v), width=50, wraplength=300)
            self.inputs.append(value['label'])
            value['label'].pack(side='left', padx=pack_opts['padx'], pady=pack_opts['pady'])

            rowFrame.pack(**pack_opts)
        labelframe.pack(**pack_opts)
        frame.pack()
        self.advancedOptionInputs.append({'frame':frame, 'pack_opts': frame.pack_info()})
        print "started"

        advanced_frame.pack(expand=1, fill=Tkconstants.Y)

        # Start button at bottom
        frame = Tkinter.Frame()
        self.startbutton = Tkinter.Button(frame, text='Start', width=35, command=self.start)
        self.startbutton.grid(row=0, column=0, padx=5, pady=5)
        Tkinter.Button(frame, text='Exit', width=35, command=lambda: self.quit()).grid(row=0, column=1, padx=5, pady=5)
        frame.pack()

        root.update()
        root.minsize(root.winfo_width(), 0)

        for obj in self.vargs:
            # lambda to generate scope -> forces obj (o) to stay the same
            obj['variable'].trace('w', lambda a,b,c, o=obj: self.changed(o))

    def setCommand(self, name):

        """Set text in the textbox"""

        print name
        self.textbox.configure(state='normal')
        self.textbox.delete(1.0, 'end')
        self.textbox.insert('insert', name)
        # self.textbox.configure(state='disabled')

    def generateFullCommand(self):

        """Generates a commandline from the options given"""

        target_type = self.target_opts['target_type'].get()
        if len(target_type) == 0:
            self.setCommand("Please select a file or folder")
            return ''
        else:
            print target_type, len(target_type)
            target = self.target_opts[target_type].get()

            if target:  # len(self.targetfile)>0 and len(self.pycommand)>0:
                # -u for unbuffered output (prevents hanging when reading stdout and stderr)
                if target.find(" ")!=-1:
                    # if filename has a space in it add quotes so it's parsed correctly as only one argument
                    cmdstr = "python -u ActivitySummary.py \"" + target + "\""
                else:
                    cmdstr = "python -u ActivitySummary.py " + target
                # if self.pycommand == "batchProcess.py": cmdstr += " " + self.targetfile

                for value in self.vargs:
                    if 'argstr' in value:
                        cmdstr += " " + value['argstr']

                # if self.pycommand != "batchProcess.py": cmdstr += " " + self.targetfile

                self.setCommand(cmdstr)
                return cmdstr
            else:
                self.setCommand("Please select a file or folder")
                return ''

    def changed(self, obj):

        """Option button callback."""

        print 'obj',obj
        # args = self.vargs[key] 
        val_type = obj['type']
        val = obj['variable'].get()
        print val_type, val, obj
        if val_type == 'bool' and val != obj['default']:
            print "bool not default"
            obj['argstr'] = '-' + obj['command'] + " " + ("True" if val else "False")
        elif val_type != 'bool' and val != self.formatargument(obj['default']):
            obj['argstr'] = '-' + obj['command'] + " " + val
            print "not default"
        else:
            print "is default"
            obj['argstr'] = ''
        self.generateFullCommand()

    def formatargument(self, value):
        if isinstance(value, list):
            return ' '.join(map(str, value))
        else:
            return str(value)

    def askopenfilename(self, **args):

        """Returns a user-selected filename. Tries to return the 'initialfile' default if nothing selected """
        print args
        if args['initialFile'] and len(args['initialFile'])>0:
            filename = tkFileDialog.askopenfilename(**self.file_opt)
        else:
            filename = tkFileDialog.askopenfilename(initialfile=args['initialFile'], **self.file_opt)

        if not filename:
            print "no filename returned"
            if args['initialFile']:
                return args['initialFile']
            else:
                return ''
        else:
            print filename
            return filename

    def askdirectory(self, **args):

        """Returns a user-selected directory name. Tries to return the 'initialdir' default if nothing selected """

        if args['initialDir'] and len(args['initialDir'])>0:
            dirname = tkFileDialog.askdirectory(initialdir = args['initialDir'], **self.dir_opt)
        else:
            dirname = tkFileDialog.askdirectory(**self.dir_opt)
        print dirname
        print args
        if not dirname:
            print "no directory name given"
            if args['initialDir']:
                return args['initialDir']
            else:
                return ''
        else:
            print dirname
            return dirname

    def start(self):

        """Start button pressed"""

        if self.isexecuting:
            self.stop()
        else:
            # cmd_line = "ping 1.0.0.0 -n 10 -w 10"
            # cmd_line = self.textbox.get("1.0", Tkconstants.END)
            cmd_line = self.generateFullCommand()

            print "running:  " + cmd_line
            self.textbox.insert("0.1", "running: ")
            p = subprocess.Popen(cmd_line, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)

            self.isexecuting = True
            self.startbutton['text'] = "Stop"
            t = Thread(target=self.pollStdout, args=(p,))
            t.daemon = True  # thread dies with the program
            t.start()

            self.threads.append(p)
            self.enableInput(False)

    def stop(self):
        self.isexecuting = False
        while self.threads:
            p = self.threads.pop()
            t = Timer(10, p.terminate)
            t.start()
            # p.terminate()# .kill()
            print p
        self.startbutton['text'] = "Start"
        self.enableInput(True)

    def pollStdout(self, p):

        """Poll the process p until it finishes"""

        start = -1  # the time when the process returns (exits)
        while True:
            retcode = p.poll()  # returns None while subprocess is running
            if start == -1 and retcode is not None:
                print "thread ended", time.time()

                self.textbox.insert(Tkinter.END, "\nprocess exited with code "+ str(retcode))
                self.textbox.see(Tkinter.END)

                start = time.time()

            line = p.stdout.readline()

            if len(line) > 0:
                print line
                self.textbox.insert(Tkinter.END, "\n" + line.rstrip())
                self.textbox.see(Tkinter.END)

            line = p.stderr.readline()

            if len(line) > 0:
                print line
                self.textbox.insert(Tkinter.END, "\nERROR: " + line.rstrip())
                self.textbox.see(Tkinter.END)
            print "read a line"

            # we stop reading after 2s to give error messages a chance to display
            if start != -1 and time.time() > start + 2:
                print "actually finished", time.time()
                self.stop()
                return
            time.sleep(0.01)

    def enableInput(self, enable=True):
        state = "normal" if enable else "disabled"
        for i in self.inputs:
            i['state'] = state

    def toggleAdvancedOptions(self, forceValue=None):
        if forceValue is not None:
            self.showAdvancedOptions = forceValue
        else:
            self.showAdvancedOptions = not self.showAdvancedOptions
        if self.showAdvancedOptions:
            self.advancedOptionsButton.config(text="Show advanced options")
            for i in self.advancedOptionInputs:
                i['frame'].pack_forget()
        else:
            self.advancedOptionsButton.config(text="Hide advanced options")
            for i in self.advancedOptionInputs:
                i['frame'].pack(**i['pack_opts'])
        pass


if __name__ == '__main__':
    root = Tkinter.Tk()
    TkinterGUI(root).pack()
    root.mainloop()
