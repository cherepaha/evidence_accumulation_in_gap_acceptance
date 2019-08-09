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

        self.start_button = tkr.Button(self, text='Start experiment', command=self.proceed)
        self.start_button.pack()

    def generate(self):
        subj_id = int(random.uniform(111, 999))
        while subj_id in self.existing_subj_ids:
            subj_id = int(random.uniform(111, 999))

        self.subj_id_entry.delete(0, tkr.END)
        self.subj_id_entry.insert(0, subj_id)

        self.session = 1

        self.start_button.bind('<Button-1>', self.write_id)

    def write_id(self, event):
        with open(self.existing_subj_ids_filename, 'a') as fp:
            writer = csv.writer(fp, delimiter = '\t')
            writer.writerow([self.subj_id_entry.get()])

    def proceed(self):
        self.exp_info = {'subj_id': int(self.subj_id_entry.get()),
                         'session': self.session,
                         'start_time': datetime.strftime(datetime.now(), '%Y_%m_%d_%H_%M')}
        self.quit()
