import tkinter as tkr
from datetime import datetime
import numpy as np
import random
import csv

class ExpInfoUI(tkr.Frame):
    def __init__(self, master=None):
        tkr.Frame.__init__(self, master)
        self.existing_subj_ids_filename = 'existing_subj_ids.txt'
        self.existing_subj_ids = np.loadtxt(self.existing_subj_ids_filename)
        self.pack()
        self.createWidgets()

        # by default, we assume that this is a second session
        # only if the experimenter generates new ID, we set session to 1
        self.session = 2

    def createWidgets(self):
        self.subj_id_label = tkr.Label(self, text='Participant number').pack()
        self.subj_id_entry = tkr.Entry(self)
        self.subj_id_entry.pack()

        self.generate_button = tkr.Button(self, text='Generate', command=self.generate)
        self.generate_button.pack()
        
        tkr.Label(self, text='Session').pack()
        self.session_var = tkr.IntVar(value=1)
        session_radios = [tkr.Radiobutton(self, text=str(session), padx=20, 
                                               variable=self.session_var, value=session) 
                                for session in [1, 2]]
        for session_radio in session_radios:
            session_radio.pack(anchor=tkr.W)

        tkr.Label(self, text='Route').pack()
        self.route_var = tkr.IntVar(value=1)
        route_radios = [tkr.Radiobutton(self, text=str(route), padx=20, 
                                               variable=self.route_var, value=route) 
                                for route in [1, 2, 3, 4]]
        for route_radio in route_radios:
            route_radio.pack(anchor=tkr.W)

        self.start_button = tkr.Button(self, text='Start experiment', command=self.proceed)
        self.start_button.pack()

    def generate(self):
        subj_id = int(random.uniform(111, 999))
        while subj_id in self.existing_subj_ids:
            subj_id = int(random.uniform(111, 999))

        self.subj_id_entry.delete(0, tkr.END)
        self.subj_id_entry.insert(0, subj_id)

        self.start_button.bind('<Button-1>', self.write_id)

    def write_id(self, event):
        with open(self.existing_subj_ids_filename, 'a') as fp:
            writer = csv.writer(fp, delimiter = '\t')
            writer.writerow([self.subj_id_entry.get()])

    def proceed(self):
        self.exp_info = {'subj_id': int(self.subj_id_entry.get()),
                         'session': self.session_var.get(),
                         'route': self.route_var.get(),
                         'start_time': datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')}
        print(self.exp_info)
        self.quit()
