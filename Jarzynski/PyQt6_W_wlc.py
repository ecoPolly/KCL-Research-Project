#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

import sys
import json
import os
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
import matplotlib.pyplot as plt
import csv
plt.style.use('dark_background')
from IPython.display import display,HTML
display(HTML("<style>.container{width:95% !important;}</style>"))
import scipy


from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QSlider, QPushButton, QLCDNumber
from PyQt5.QtCore import QSize
from PyQt5.uic import loadUiType
from PyQt5.QtGui import QIcon

from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit





qtCreatorFile = "UI_Jarzynski_dG_W.ui" # Enter file here.

Ui_MainWindow, QtBaseClass = loadUiType(qtCreatorFile)        

Total_Work = []
Work_polymer = []

global file_path
file_path = ""

def WLC(x, Lc, lp, L0):
    return (4.11 / lp) * (0.25 * (1 - (x - L0) / Lc) ** (-2) + (x - L0) / Lc - 0.25)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    


    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        super().__init__()
        
        self.setupUi(self)
        self.setWindowTitle('RAMP ANALYZER')
        self.Open_JSON.clicked.connect(self.File_open)
            
        ## SELF BUTTONS 
        self.Plot_button.clicked.connect(self.Plotter)
        self.Plot_avg_traj.clicked.connect(self.Integrate_curve)
        self.wlc_work.clicked.connect(self.WLC_work)
        #self.Histo.clicked.connect(self.Do_histogram)
        self.Save_file.clicked.connect(self.Save_to_file)
   
        grafico=FigureCanvas(Figure(figsize=(8,6),constrained_layout=True, edgecolor='black', frameon=True))
        self.Plotting_box.addWidget(grafico)
        self.grafico=grafico.figure.subplots()
        self.grafico.set_xlabel('Extension (nm)', size=8)
        self.grafico.set_ylabel('Force (pN)', size=8)
        
        x=np.linspace(0,0.03, 100)
        self.grafico.plot(x, 0*x)

        # initial guess for gui 
        self.Fmin_2.setValue(4.5)
        self.Fmin.setValue(12)
        self.Fmax_2.setValue(7.8)
        self.Fmax.setValue(20)
        self.Guess_4.setValue(50.0) 
        self.Guess_5.setValue(0.6)  
        self.Guess_6.setValue(5.0) 
        self.Guess_1.setValue(110.0)  
        self.Guess_2.setValue(0.4)
        self.Guess_3.setValue(5.0) 

        # collego save_df a Wclean
        self.FJC_W.clicked.connect(self.FJC)
        # bottone per W tot + hist W tot
        self.compute_sgW.clicked.connect(self.sg_Wtot)

        grafico_hist = FigureCanvas(Figure(figsize=(8, 6), constrained_layout=True))
        self.W_hist.layout().addWidget(grafico_hist)  
        self.grafico_hist = grafico_hist.figure.subplots()

        # global fit 
        self.global_fit.clicked.connect(self.WLC_global_fit)        
  
    def File_open(self):
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Open JSON File")
        global file_path
        file_path = name[0]
        
        return file_path
      
   
      
    def Load_JSON(self):        
        with open(file_path , 'r') as f:
           d=json.load(f) 
         
      #  d = self.File_open()  
        xs, forces, times, current = [], [], [], []
        # Pulse_num=self.Pulse_num.value()
        for i in range (len(d)):
            xs.append(np.array(d["Pulse_Number_"+str(i)]["z"]))
            forces.append(np.array(d["Pulse_Number_"+str(i)]["force"]))
            times.append(np.array(d["Pulse_Number_"+str(i)]["time"])-d["Pulse_Number_"+str(i)]["time"][0])    
            current.append(np.array(d["Pulse_Number_"+str(i)]["current"]))
            
        return xs, forces, times, current
        
        
    def Plotter(self, plot_avg=False):
        xs, forces, times, current = self.Load_JSON()
        Pulse_num = self.Pulse_num.value()
        # xs_smth = savgol_filter(xs, 51, 4)
        xs_smth = [savgol_filter(arr, 81, 4) for arr in xs]


        for i in range(len(forces)):
            forces[i] = np.array((3.48e-5) * current[i]**2 - 0.0029 * current[i])

        self.grafico.clear()
        self.grafico.set_xlabel('Extension (nm)', size=6)
        self.grafico.set_ylabel('Force (pN)', size=6)
        # self.grafico.set_xlim(0, 105)

        if plot_avg:
            x_avg, f_avg = self.avg_traj(xs, forces, n=100)
            self.grafico.plot(x_avg, f_avg, '-', lw=1, alpha=0.5,  color='orange')
        else:
            self.grafico.plot(xs[Pulse_num], forces[Pulse_num], alpha=0.5)
            self.grafico.plot(xs_smth[Pulse_num], forces[Pulse_num], alpha=0.5)
        self.grafico.figure.canvas.draw()

        return Pulse_num, xs_smth, forces, times

    
    
    def avg_traj(self, xs_smth, forces, n=100):
            min_len = min(len(xs_smth[i]) for i in range(n))  # prendo stesso numero di punti per pulse 
            xs_array = np.array([xs_smth[i][:min_len] for i in range(n)])
            fs_array = np.array([forces[i][:min_len] for i in range(n)])
            x_avg = np.mean(xs_array, axis=0)
            f_avg = np.mean(fs_array, axis=0)

            return x_avg, f_avg
    
    # load for each sg work tot
    def Load(self):
        with open(file_path, 'r') as f:
            d = json.load(f)

        xs, forces_calc, times, current = [], [], [], []

        for i in range(len(d)):
            key = f"Pulse_Number_{i}"

            # estrazione base
            x = np.array(d[key]["z"])
            t = np.array(d[key]["time"]) - d[key]["time"][0]
            I = np.array(d[key]["current"])

            # calcolo forza teorica per ogni pulse
            F = (3.48e-5) * I**2 - 0.0029 * I

            # salvataggio
            xs.append(x)
            times.append(t)
            current.append(I)
            forces_calc.append(F)

        return xs, forces_calc, times, current


    def sg_Wtot(self):
        xs, forces, times, current = self.Load()
        work_list = []

        xs_smth = [savgol_filter(x, 51, 4) for x in xs]

        for idx in range(min(100, len(forces))):
            xmin = next(i for i in range(1, len(forces[idx]))
                        if forces[idx][i] > 6 and forces[idx][i - 1] < 6)
            xmax = next(i for i in range(1, len(forces[idx]))
                        if forces[idx][i] > 20 and forces[idx][i - 1] < 20)

            integrate_force = forces[idx][xmin:xmax]
            integrate_x = xs_smth[idx][xmin:xmax]
            W = scipy.integrate.simpson(integrate_force, integrate_x)
            work_list.append(W)

        # self.Label_TotalWork.setText(",".join(f"{w:.2f}" for w in work_list))
        self.grafico_hist.set_xlabel('Total Work  [pNnm]', size=4)
        self.grafico_hist.hist(work_list, bins=100, edgecolor='black', alpha=0.6, color='orange')
        self.grafico_hist.figure.canvas.draw()

        return work_list



    def Integrate_curve(self):
        Pulse_num, xs_smth, forces, times = self.Plotter(plot_avg=True)  # togli contenuto quando vuoi plot sg 
        x_avg, f_avg = self.avg_traj(xs_smth, forces, n=100)

        # == calcolo limits si traj media == 
        for i in range(1, len(f_avg)):
            if f_avg[i] > 6 and f_avg[i - 1] < 6:
                xmin = i
            if f_avg[i] > 20 and f_avg[i - 1] < 20:
                xmax = i

        xmin_reale = x_avg[xmin]
        xmax_reale = x_avg[xmax]
        print("xmin reale:", xmin_reale, "xmax reale:", xmax_reale)

        return xmin_reale, xmax_reale, x_avg, f_avg
    

    def compute_WLC_dx(self, xmin_reale, xmax_reale, Lc_dx, Lp_dx, L0):
            x_interp = np.linspace(xmin_reale, xmax_reale, 1000)  
            x_vals = []
            y_vals = []

            for x in x_interp:
                try:
                    F = WLC(x, Lc_dx, Lp_dx, L0)
                    x_vals.append(x)
                    y_vals.append(F)
                except:
                    continue
            
            area_wlc = abs(scipy.integrate.simpson(y_vals, x_vals))
            return area_wlc
    
    def compute_WLC_sx(self, xmin_reale,Lc_sx, Lp_sx, L0):
            x  = xmin_reale
            x_vals_sx = []
            y_vals_sx = []
            while x > -5:  # estendiamo anche in negativo
                try:
                    f = WLC(x, Lc_sx, Lp_sx, L0)
                except:
                    break
                if f < 0.1:
                    x_zero = x
                    break
                x_vals_sx.append(x)
                y_vals_sx.append(f)
                x -= 0.05

            x_vals_sx = x_vals_sx[::-1]
            y_vals_sx = y_vals_sx[::-1]
            area_wlc = scipy.integrate.simpson(y_vals_sx, x_vals_sx)
            return area_wlc
    
    def WLC_work(self):
        xmin_reale, xmax_reale, x_avg, f_avg = self.Integrate_curve()
        Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0 = self.WLC_global_fit()
        work_dx = self.compute_WLC_dx(xmin_reale, xmax_reale, Lc_dx, Lp_dx, L0)
        work_sx = self.compute_WLC_sx(xmin_reale, Lc_sx, Lp_sx, L0)

        self.Label_WLCfit_dx.setText(
        f"Lc_dx={Lc_dx:.2f}  Lp_dx={Lp_dx:.2f}  L0_dx={L0:.2f}")
        self.Label_WLCfit_sx.setText(
        f"Lc_sx={Lc_sx:.2f}  Lp_sx={Lp_sx:.2f}  L0_sx={L0:.2f}")

        self.Label_WorkWLC_dx.setText(f"Work WLC dx={work_dx:.2f}")
        self.Label_WorkWLC_sx.setText(f"Work WLC sx={work_sx:.2f}")

        return work_dx, work_sx
    
    ## == GLOBAL FIT ==
    def WLC_global_fit(self):
        xmin_reale, xmax_reale, x_avg, f_avg = self.Integrate_curve()
        fmin_dx = self.Fmin.value()
        fmax_dx = self.Fmax.value()
        fmin_sx = self.Fmin_2.value()
        fmax_sx = self.Fmax_2.value()

        # dx wlc 
        for i in range(1, len(f_avg)):
            if f_avg[i] > fmin_dx and f_avg[i-1] < fmin_dx:
                i_min_dx = i
            if f_avg[i] > fmax_dx and f_avg[i-1] < fmax_dx:
                i_max_dx = i
        xfit_dx = x_avg[i_min_dx:i_max_dx]
        yfit_dx = f_avg[i_min_dx:i_max_dx]

        # sx wlc 
        for i in range(1, len(f_avg)):
            if f_avg[i] > fmin_sx and f_avg[i-1] < fmin_sx:
                i_min_sx = i
            if f_avg[i] > fmax_sx and f_avg[i-1] < fmax_sx:
                i_max_sx = i
        xfit_sx = x_avg[i_min_sx:i_max_sx]
        yfit_sx = f_avg[i_min_sx:i_max_sx]

        def WLC(x, Lc, Lp, L0):
            return (4.11 / Lp) * (0.25 * (1 - (x - L0) / Lc) ** (-2) + (x - L0) / Lc - 0.25)

        def cost(params):
            Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0 = params
            try:
                pred_dx = WLC(xfit_dx, Lc_dx, Lp_dx, L0)
                pred_sx = WLC(xfit_sx, Lc_sx, Lp_sx, L0)
                return np.mean((yfit_dx - pred_dx)**2) + np.mean((yfit_sx - pred_sx)**2)
            except:
                return 1e10

        guess = [self.Guess_1.value(), self.Guess_2.value(), self.Guess_4.value(), self.Guess_5.value(), self.Guess_3.value()]
        bounds = [(60, 150), (0.2, 0.7), (20, 90), (0.2, 0.7), (0, 6)]

        # minimize loss function 
        result = scipy.optimize.minimize(cost, guess, method='L-BFGS-B', bounds=bounds)

        if result.success:
            Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0 = result.x
            self.Label_WLCfit_dx.setText(f"[Global Fit] Lc_dx={Lc_dx:.2f}  Lp_dx={Lp_dx:.2f}")
            self.Label_WLCfit_sx.setText(f"[Global Fit] Lc_sx={Lc_sx:.2f}  Lp_sx={Lp_sx:.2f}  L0={L0:.2f}")
        else:
            print("Global fit failed:", result.message)

        #plot
        yfit_dx = WLC(xfit_dx, Lc_dx, Lp_dx, L0)
        yfit_sx = WLC(xfit_sx, Lc_sx, Lp_sx, L0)

        self.grafico.plot(xfit_dx, yfit_dx, '--', lw=1.2, color='orange', label='Global Fit DX')
        self.grafico.plot(xfit_sx, yfit_sx, '--', lw=1.2, color='purple', label='Global Fit SX')
        self.grafico.relim()
        self.grafico.autoscale_view()
        self.grafico.figure.canvas.draw()

        return Lc_dx, Lp_dx, Lc_sx, Lp_sx, L0

    # salvo tutti i valori in df 
    def FJC(self):
        from scipy.integrate import simpson

        Lc, kT = 40, 4.11 # poi automatizza 
        F = np.linspace(0.01, 25, 100)
        f0 = 4  
        beta = F * Lc / kT
        extension = Lc * (1 / np.tanh(beta) - 1 / beta)

        mask = F <= f0
        F_sel = F[mask]
        ext_sel = extension[mask]
        beta0 = f0 * Lc / kT
        y0 = Lc * (1 / np.tanh(beta0) - 1 / beta0)
        area = simpson(y0 - ext_sel, F_sel)
        print ("area:", area)

        self.Label_fjc.setText(f"{area:.2f} pNnm")
        plt.fill_between(F_sel, ext_sel, y0, color="green", alpha=0.5)
        plt.plot(F, extension)
        plt.axvline(x = 4, linestyle = "--")
        plt.xlabel("Force (pN)")
        plt.ylabel("Extension (nm)")
        plt.title("FJC Extension vs Force")
        plt.grid(True)
        plt.show()


    
  
        # salva tutti i values stored con save works in un file csv 
    def Save_to_file(self):
        #F= self.Step_detect()
        work_list = self.sg_Wtot()
        work_dx, work_sx = self.WLC_work()
        df = pd.DataFrame({"Pulse": list(range(len(work_list))), "Work_tot": work_list})

        # Aggiungi righe finali per lavori del polimero
        df_polymer = pd.DataFrame({"Pulse": ["WLC_dx", "WLC_sx"],"Work_tot": [work_dx, work_sx]})
        df_finale = pd.concat([df, df_polymer], ignore_index=True)
        df_finale.to_csv("work_results.csv", index=False)


# if __name__ == "__main__":
def launch_W_wlc():
     app = QtWidgets.QApplication(sys.argv)
     window = MyApp()
     window.show()
     sys.exit(app.exec_())   