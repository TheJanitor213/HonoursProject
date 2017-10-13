from Tkinter import *
import tkFileDialog
import ttk
from tkFileDialog import askopenfilename
import os
filename=''

def input(*args):
    global filename
    Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
    root.update()
    filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    feet_entry.insert(10,filename)


def run():
    print(filename)
    os.system("python main.py "+ filename)
root = Tk()
root.title("Image Processing")

mainframe = ttk.Frame(root, padding="30 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

feet = StringVar()
meters = StringVar()

feet_entry = ttk.Entry(mainframe, width=10, textvariable=feet)
feet_entry.grid(column=0, row=1, sticky=(W, E))

ttk.Label(mainframe, text="Welcome to the Receipt Analysis System").grid(column=0, row=0, sticky=(W, E))
ttk.Button(mainframe, text="Process", command=run).grid(column=0, row=3, sticky=W)

ttk.Button(mainframe, text="Upload", command=input).grid(column=3, row=1, sticky=W)
ttk.Button(mainframe, text="Exit", command=exit).grid(column=3, row=3, sticky=W)

for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

feet_entry.focus()


root.mainloop()
exit()
