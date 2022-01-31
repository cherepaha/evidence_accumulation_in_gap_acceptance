import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ddm
import os
import csv
from matplotlib.lines import Line2D

def merge_csv(directory):
    fout = open(directory+"_parameters_fitted.csv","w+")
    header_written = False
    for i, file_name in enumerate(os.listdir(directory)):
        file_path = os.path.join(directory, file_name)
        if file_path.endswith(".csv"):
            f = open(file_path)
            if header_written:
                # skip the header for the first row
                next(f) 
            for line in f:
                fout.write(line)
            f.close()
            header_written = True
            print(file_path)
    fout.close()

def differentiate(t, x):
    # To be able to reasonably calculate derivatives at the end-points of the trajectories,
    # I append three extra points before and after the actual trajectory, so we get N+6
    # points instead of N       
    x = np.append(x[0]*np.ones(3), np.append(x, x[-1]*np.ones(3)))
    
    # Time vector is also artificially extended by equally spaced points
    # Use median timestep to add dummy points to the time vector
    timestep = np.median(np.diff(t))
    t = np.append(t[0]-np.arange(1,4)*timestep, np.append(t, t[-1]+np.arange(1,4)*timestep))

    # smooth noise-robust differentiators, see: 
    # http://www.holoborodko.com/pavel/numerical-methods/ \
    # numerical-derivative/smooth-low-noise-differentiators/#noiserobust_2
    v = (1*(x[6:]-x[:-6])/((t[6:]-t[:-6])/6) + 
         4*(x[5:-1] - x[1:-5])/((t[5:-1]-t[1:-5])/4) + 
         5*(x[4:-2] - x[2:-4])/((t[4:-2]-t[2:-4])/2))/32
    
    return v

def write_to_csv(directory, filename, array, write_mode="a"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, filename), write_mode, newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(array)
       
def fit_model(model, training_data, loss_function):
    training_sample = ddm.Sample.from_pandas_dataframe(df=training_data, 
                                                       rt_column_name="RT", 
                                                       correct_column_name="is_go_decision")
    return ddm.fit_adjust_model(sample=training_sample, model=model, lossfunction=loss_function, verbose=False)
