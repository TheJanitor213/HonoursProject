#!/usr/bin/env python
import Tkinter as tk
import os
import tkFont
import cv2
from PIL import ImageTk, Image

top = tk.Tk()
top.configure(background='light blue')
tex = tk.Text(width = 40, height=30, font=("Helvetica", 20),master=top)
tex.configure(background="light blue")
tex.pack(side=tk.RIGHT)
with open("itemsBought.txt") as f:
    content = f.readlines()

bop = tk.Frame(width=100,height=100)
bop.pack(side=tk.BOTTOM)

img = ImageTk.PhotoImage(Image.open("copy.jpg"))

panel = tk.Label(image = img,width=1200,height=1200)
panel.configure(background="light blue")
panel.pack(side = tk.LEFT)

#remove whitespace at the end of each line
content = [x.strip() for x in content]

for k in content[1:]:
	tex.insert(tk.END,str(k)+"\n")
	tex.see(tk.END)


def back():
	top.destroy()
	os.system("python gui.py")


tk.Button(bop, text='Back', command=back).pack()
top.mainloop()

exit()
