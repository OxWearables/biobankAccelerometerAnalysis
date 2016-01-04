import Tkinter, Tkconstants, tkFileDialog
import math
from functools import partial

class TkinterGUI(Tkinter.Frame):

    def __init__(self, root):

        Tkinter.Frame.__init__(self, root)

        # options for buttons
        button_opt = {'fill': Tkconstants.BOTH, 'padx': 5, 'pady': 5}
        txt_opt = {'fill': Tkconstants.X, 'padx': 5, 'pady': 5}
        # to know if a file has been selected
        self.chosen_file = ""
        frame = Tkinter.Frame()


        # define buttons
        Tkinter.Button(frame, text='Choose file', command=self.askopenfilename, width=35).grid(row=0, column=0, padx=5, pady=5)
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

        Tkinter.Button(frame, text='Choose directory', command=self.askdirectory, width=35).grid(row=0, column=1, padx=5, pady=5)
        # defining options for opening a directory
        self.dir_opt = options = {}
        # options['initialdir'] = 'C:\\'
        options['mustexist'] = False
        options['parent'] = root
        options['title'] = 'Select folder to process'

        # Textbox 
        txt = Tkinter.Text(frame, height = 4, width=80)
        txt.grid(row=1, column=0, columnspan=2)
        txt.insert('insert', "Please select a file or folder")
        self.textbox = txt

        frame.pack(**button_opt)


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

            Tkinter.Checkbutton(frame, text=value['text'], variable=value['variable']).pack(side='left',**button_opt)

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
                
            print value['type'] 
            # need to make these variables pernament since if they get garbage collected tkinter will fail
            frame = Tkinter.Frame()

            value['labelvar'] = Tkinter.StringVar()
            value['labelvar'].set(value['text'])
            value['label'] = Tkinter.Label(frame, textvariable=value['labelvar'], width=50, wraplength=300)
            value['label'].pack(side='left')

            # print str(value['default'])
            value['variable'] = Tkinter.StringVar()
            value['variable'].set(self.formatargument(value['default']))

            Tkinter.Entry(frame,textvariable=value['variable'],width=50).pack(side='right')
            frame.pack(**button_opt)

        # Start button at bottom (todo)
        frame = Tkinter.Frame()
        Tkinter.Button(frame, text='Start (todo)', width=35, command=self.askdirectory).grid(row=0, column=0, padx=5, pady=5)
        Tkinter.Button(frame, text='Exit (todo)', width=35, command=self.askdirectory).grid(row=0, column=1, padx=5, pady=5)
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
    
        print dir(self)
        if len(self.chosen_file)>0:
            cmdstr = "python ActivitySummary.py"
            for key,value in self.vargs.iteritems():
                if 'argstr' in value:
                    cmdstr += " " + value['argstr']

            cmdstr += " " + self.chosen_file
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
            arg['argstr'] = '--' + key + " " + ("True" if val else "False")
        elif arg['type'] != 'bool' and  val != self.formatargument(arg['default']):
            arg['argstr'] = '--' + key + " " + val
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

    def askopenfilename(self):

        """Returns an opened file in read mode.
        This time the dialog just returns a filename and the file is opened by your own code.
        """

        # get filename
        filename = tkFileDialog.askopenfilename(**self.file_opt)
        # open file on your own
        if not filename:
            print "no filename returned"
        else:
            print filename
            self.file_opt['initialfile'] = filename
            self.setCommand(filename)
            self.chosen_file = filename
            self.generateFullCommand()

    def askdirectory(self):

        """Returns a selected directoryname."""

        dirname = tkFileDialog.askdirectory(**self.dir_opt)
        print dirname
        if not dirname:
            return
        self.setCommand(dirname)
        self.dir_opt['initialdir'] = dirname
        self.chosen_file = filename
        self.generateFullCommand()
        return


def merge_two_dicts(x, y):

    """Given two dicts, merge them into a new dict as a shallow copy."""

    z = x.copy()
    z.update(y)
    return z

if __name__=='__main__':
  root = Tkinter.Tk()
  TkinterGUI(root).pack()
  root.mainloop()
