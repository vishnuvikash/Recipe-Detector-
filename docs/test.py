'''
orderwidth=1)
        self.frame.grid(row=1,column=1,pady=4,padx=5,sticky=W)
        self.label=Label(self.frame,text="Scope:")
        self.label.pack(side="left", fill=None, expand=False)

        self.var = StringVar()
        self.var.set("today")
                        "last month")
        self.list.pack(side="left", fill=None, expand=False)

        self.fetchButton = Button(self.frame, text="Fetch",command=self.handle)
        self.fetchButton.pack(side="left", fill=None, expand=False)

        self.area = Text(self,height=15,width=60)
        self.area.grid(row=2,column=1,rowspan=1,pady=4,padx=5)

        self.scroll = Scrollbar(self)
        self.scroll.pack(side=RIGHT, fill=Y)

        self.area.config(yscrollcommand=self.scroll.set)
'''
from tkinter import Tk, Label, Button

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("A simple GUI")

        self.label = Label(master, text="This is our first GUI!")
        self.label.pack()

        self.greet_button = Button(master, text="Greet", command=self.greet)
        self.greet_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def greet(self):
        print("Greetings!")

root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()
