import Tkinter as Tk
import Tkconstants, tkFileDialog
import os
import ttk
# spawning child processes
import subprocess
from threading import Thread, Timer
import time
from sys import platform as _platform


class VerticalScrolledFrame(Tk.Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling

    """
    def __init__(self, parent, *args, **kw):
        Tk.Frame.__init__(self, parent, *args, **kw)            

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Tk.Scrollbar(self, orient="vertical")
        vscrollbar.pack(fill="y", side="right", expand=False)
        canvas = Tk.Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=vscrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscrollbar.config(command=canvas.yview)

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Tk.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor="nw")
        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior(event):
            # update the scrollbars to match the size of the inner frame
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            # print size
            canvas.config(scrollregion="0 0 %s %s" % size)
            # print "scrollregion = 0 0 %s %s" % size
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config(width=interior.winfo_reqwidth())
        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())
        canvas.bind('<Configure>', _configure_canvas)

        def _on_mousewheel(event, is_OSX=False):
            if is_OSX:
                canvas.yview_scroll(-1*(event.delta), "units")
            else:
                canvas.yview_scroll(-1*(event.delta/120), "units")
        # Linux OS
        if _platform == "linux" or _platform == "linux2":
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)
        # Windows
        elif _platform == "win32":
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        # OSX
        elif _platform == "darwin":
            canvas.bind_all("<MouseWheel>", lambda evt: _on_mousewheel(evt, True))


class DateEntry(Tk.Frame):
    def __init__(self, master, frame_look={}, **look):
        args = dict(relief=Tk.SUNKEN, border=1)
        args.update(frame_look)
        Tk.Frame.__init__(self, master, **args)

        args = {'relief': Tk.FLAT}
        args.update(look)

        self.entry_1 = Tk.Entry(self, width=4, **args)
        self.label_1 = Tk.Label(self, text='-', **args)
        self.entry_2 = Tk.Entry(self, width=2, **args)
        self.label_2 = Tk.Label(self, text='-', **args)
        self.entry_3 = Tk.Entry(self, width=2, **args)
        self.label_3 = Tk.Label(self, text='T', **args)
        self.entry_4 = Tk.Entry(self, width=2, **args)
        self.label_4 = Tk.Label(self, text=':', **args)
        self.entry_5 = Tk.Entry(self, width=2, **args)

        self.entry_1.pack(side=Tk.LEFT)
        self.label_1.pack(side=Tk.LEFT)
        self.entry_2.pack(side=Tk.LEFT)
        self.label_2.pack(side=Tk.LEFT)
        self.entry_3.pack(side=Tk.LEFT)
        self.label_3.pack(side=Tk.LEFT)
        self.entry_4.pack(side=Tk.LEFT)
        self.label_4.pack(side=Tk.LEFT)
        self.entry_5.pack(side=Tk.LEFT)

        self.entry_1.bind('<KeyRelease>', self._e1_check)
        self.entry_2.bind('<KeyRelease>', self._e2_check)
        self.entry_3.bind('<KeyRelease>', self._e3_check)
        self.entry_4.bind('<KeyRelease>', self._e4_check)
        self.entry_5.bind('<KeyRelease>', self._e5_check)

    def _backspace(self, entry):
        cont = entry.get()
        entry.delete(0, Tk.END)
        entry.insert(0, cont[:-1])

    def _e1_check(self, e):
        cont = self.entry_1.get()
        if len(cont) >= 4:
            self.entry_2.focus()
        if len(cont) > 4 or (len(cont)>0 and not cont[-1].isdigit()):
            self._backspace(self.entry_1)
            self.entry_1.focus()

    def _e2_check(self, e):
        cont = self.entry_2.get()
        if len(cont) >= 2:
            self.entry_3.focus()
        if len(cont) > 2 or (len(cont)>0 and not cont[-1].isdigit()):
            self._backspace(self.entry_2)
            self.entry_2.focus()

    def _e3_check(self, e):
        cont = self.entry_3.get()
        if len(cont) >= 2:
            self.entry_4.focus()
        if len(cont) > 2 or (len(cont)>0 and not cont[-1].isdigit()):
            self._backspace(self.entry_3)
            self.entry_3.focus()

    def _e4_check(self, e):
        cont = self.entry_4.get()
        if len(cont) >= 2:
            self.entry_5.focus()
        if len(cont) > 2 or (len(cont)>0 and not cont[-1].isdigit()):
            self._backspace(self.entry_4)
            self.entry_4.focus()

    def _e5_check(self, e):
        cont = self.entry_5.get()
        if len(cont) > 2 or (len(cont)>0 and not cont[-1].isdigit()):
            self._backspace(self.entry_5)

    def get(self):
        return self.entry_1.get()+"-"+self.entry_2.get()+"-"+self.entry_3.get()+"T"+self.entry_4.get()+":"+self.entry_5.get()


class TkinterGUI(Tk.Frame):

    def __init__(self, root):

        Tk.Frame.__init__(self, root)
        root.title("Accelerometer Processing")
        # options for buttons
        pack_opts = {'fill': Tkconstants.BOTH, 'padx': 5, 'pady': 5}
        # to know if a file has been selected
        self.targetfile = ""  # file to process
        self.pycommand = ""  # either ActivitySummary.py or batchProcess.py
        self.isexecuting = False  # are we running a shell process?
        self.threads = []  # keep track of threads so we can stop them

        self.target_opts = {
            'filename': Tk.StringVar(), # we use stringVar so we can monitor for changes
            'dirname': Tk.StringVar(),
            'filenames': [], # for multiple file selections, unused due to dodgy Tkinter support
            'target_type': Tk.StringVar(),
            'file_list': [],
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

        # when either filename or dirname are changed, we use this callback to set the target_type refresh the display
        def target_callback(type):
            print "target_callback", type
            self.target_opts['target_type'].set(type)
            self.refreshFileList()
            self.generateFullCommand()
        self.target_opts['filename'].trace("w", lambda *args: target_callback('filename'))
        self.target_opts['dirname'].trace("w", lambda *args: target_callback('dirname'))

        # vargs are in format {'command': name of -command, 'varable': StringVar() for value,
        # 'default': if same as default it's unchanged, 'type': 'bool', 'string', 'int', or 'float' }
        # -[command] [variable].get()   (only add if [variable] != [default])
        self.vargs = []
        # list of buttons that should be disabled when the processing has started
        self.inputs = []

        self.advanced_frame = VerticalScrolledFrame(root)

        frame = Tk.Frame()
        # define buttons
        self.inputs.append(
            Tk.Button(frame,
                text='Choose file',
                command=lambda: self.target_opts['filename'].set(
                        self.askopenfilename(initialFile=self.target_opts['filename'].get())
                ),
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
            Tk.Button(frame,
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

        self.showAdvancedOptions = False
        self.advancedOptionsButton = Tk.Button(frame,
                                        text='Hide advanced Options',
                                        command=self.toggleAdvancedOptions,
                                        width=35)
        self.advancedOptionsButton.grid(row=0, column=2, padx=5, pady=5)

        self.textbox = Tk.Text(frame, height=10, width=70)
        self.textbox.insert('insert', "Please select a file or folder")
        self.textbox.grid(row=1, column=0, columnspan=3,
                              sticky=Tkconstants.N,# + Tkconstants.E + Tkconstants.S + Tkconstants.W,
                              padx=5, pady=5)

        frame.pack(expand=0, **pack_opts)

        # boolean options
        self.checkboxes = {
            'skipCalibration': {'text': 'Skip calibration step', 'default': False},
            'verbose': {'text': 'Verbose mode', 'default': False},
            'deleteIntermediateFiles': {'text':'Delete intermediate files', 'default': True},
            'processRawFile': {'text': 'Process the raw (.cwa) file', 'default': True},
            'rawOutput': {'text': 'Raw 100Hz data to (.csv) file', 'default': False},
            'timeSeriesDateColumn': {'text': 'dateTime column', 'default': False}
        }

        frame = Tk.Frame(self.advanced_frame.interior)
        for key, value in self.checkboxes.iteritems():
            value['type'] = 'bool'
            value['variable'] = Tk.IntVar()
            value['variable'].set(value['default'])

            self.vargs.append({
                'command': key,
                'variable': value['variable'],
                'default': value['default'],
                'type': 'bool'})

            self.inputs.append(Tk.Checkbutton(frame, text=value['text'], variable=value['variable']))
            self.inputs[-1].pack(side='left',**pack_opts)

        # print {key: value['variable'].get() for (key, value) in self.checkboxes.iteritems()}
        frame.pack(fill=Tkconstants.NONE, padx=pack_opts['padx'], pady=pack_opts['pady'])

        # more complicated options, we will just pass them in as text for now (hoping the user will put anything silly)
        option_groups = {
            'Calibration options': {
                'calibrationOffset': {'text': 'Calibration offset',
                                      'default': [0.0,0.0,0.0]},
                'calibrationSlope': {'text': 'Calibration slope linking offset to temperature',
                                     'default': [1.0,1.0,1.0]},
                'calibrationTemperature': {'text': 'Mean temperature in degrees Celsius of stationary data for'
                                                   ' calibration',
                                           'default': [0.0,0.0,0.0]},
                'meanTemperature': {'text': 'Mean calibration temperature in degrees Celsius',
                                    'default': 20.0}

            },
            'Java options': {
                'javaHeapSpace': {'text': 'Amount of heap space allocated to the java subprocesses, useful for limiting'
                                          ' RAM usage (leave blank for no limit)',
                                  'default': ''},
                'rawDataParser': {'text': 'Java (.class) file, which is used to process (.cwa) binary files',
                                  'default': 'AxivityAx3Epochs'}
            },
            'Epoch options': {
                'epochPeriod': {'text': 'Length in seconds of a single epoch',
                                'default': 5}
            },
            'Multi-threading options': {
                'numWorkers': {'text': 'Number of processing threads to execute simultaneously (for multiple files)',
                                'default': 1}
            },
            'Start/end time options': {
                'startTime': {'text': 'Start date in format 2016-04-08T17:10',
                                'default': ""},
                'endTime': {'text': 'End date in format 2016-04-08T17:10',
                                'default': ""}
            }
        }
        frame = Tk.Frame(self.advanced_frame.interior)
        for key, groups in option_groups.iteritems():
            labelframe = Tk.LabelFrame(frame, text=key)
            for key, value in groups.iteritems():

                if isinstance(value['default'], list):
                    value['type'] = 'multi'
                elif isinstance(value['default'], str):
                    value['type'] = 'string'
                elif isinstance(value['default'], int):
                    value['type'] = 'int'
                else:
                    value['type'] = 'float'

                # need to make these variables permanent since if they get garbage collected tkinter will fail
                rowFrame = Tk.Frame(labelframe)

                value['labelvar'] = Tk.StringVar()
                value['labelvar'].set(value['text'])
                value['label'] = Tk.Label(rowFrame, textvariable=value['labelvar'], width=50, wraplength=300)
                value['label'].pack(side='left')

                # print str(value['default'])
                value['variable'] = Tk.StringVar()
                value['variable'].set(self.formatargument(value['default']))

                self.vargs.append({'command': key, 'variable': value['variable'],
                                   'default': value['default'], 'type': value['type']})

                self.inputs.append(Tk.Entry(rowFrame,textvariable=value['variable'],width=50))
                self.inputs[-1].pack(side='right' , expand=1, fill=Tkconstants.X)
                rowFrame.pack(**pack_opts)
            labelframe.pack(**pack_opts)
        frame.pack()

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

        frame = Tk.Frame(self.advanced_frame.interior)
        labelframe = Tk.LabelFrame(frame, text="Folder options (default is same folder as input file)")
        for key, value in folder_params.iteritems():
            rowFrame = Tk.Frame(labelframe)

            value['variable'] = Tk.StringVar()
            value['variable'].set("")
            value['variable'].trace('w', lambda *args: self.generateFullCommand())

            self.vargs.append({'command': key, 'variable': value['variable'], 'default': '', 'type':'string'})

            self.inputs.append(Tk.Entry(rowFrame,textvariable=value['variable'],width=50))
            self.inputs[-1].pack(side='right', expand=1, fill= Tkconstants.X)

            value['label'] = Tk.Button(rowFrame,
                                            text=value['text'],
                                            command=lambda v=value['variable']: chooseFolder(v),
                                            width=50, wraplength=300)
            self.inputs.append(value['label'])
            value['label'].pack(side='left', padx=pack_opts['padx'], pady=pack_opts['pady'])

            rowFrame.pack(**pack_opts)
        labelframe.pack(**pack_opts)
        frame.pack()
        print "started"


        # Start button at bottom
        frame = Tk.Frame()
        self.startbutton = Tk.Button(frame, text='Start', width=35, command=self.start)
        self.startbutton.grid(row=0, column=0, padx=5, pady=5)
        Tk.Button(frame, text='Exit', width=35, command=lambda: self.quit()).grid(row=0, column=1, padx=5, pady=5)
        frame.pack()
        self.advanced_frame.pack(expand=1, fill="both")

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

    def refreshFileList(self):
        if self.target_opts['target_type'].get() != 'dirname':
            print self.target_opts['target_type'].get()
            print 'no dir!'
            return
        # self.tab2.configure(state='enabled')
        dir = self.target_opts['dirname'].get()
        self.target_opts['file_list'] = [f for f in os.listdir(dir) if any([f.lower().endswith(ext) for ext in ['.cwa','.bin']])]

        # print  self.target_opts['file_list']
        # self.listbox.delete(0, Tkconstants.END)

        # for f in self.target_opts['file_list']:
        #     self.listbox.insert(Tkconstants.END, f)

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

                self.textbox.insert(Tk.END, "\nprocess exited with code "+ str(retcode))
                self.textbox.see(Tk.END)

                start = time.time()

            line = p.stdout.readline()

            if len(line) > 0:
                print line
                self.textbox.insert(Tk.END, "\n" + line.rstrip())
                self.textbox.see(Tk.END)

            line = p.stderr.readline()

            if len(line) > 0:
                print line
                self.textbox.insert(Tk.END, "\nERROR: " + line.rstrip())
                self.textbox.see(Tk.END)
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
            self.advanced_frame.pack_forget()
        else:
            self.advancedOptionsButton.config(text="Hide advanced options")
            self.advanced_frame.pack(expand=1, fill="both")
        pass


if __name__ == '__main__':
    root = Tk.Tk()
    TkinterGUI(root).pack()
    root.mainloop()
