import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal import fftconvolve
from numpy.fft import fftshift
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QVBoxLayout
from PyQt5 import uic
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
plt.style.use('dark_background')

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("UI_Jarzynski_2.ui", self)
        
        # Canvas matplotlib per il plot principale (traiettoria)
        self.canvas = FigureCanvas(Figure())
        self.ax = self.canvas.figure.subplots()
        lay = QVBoxLayout(self.widgetPlot)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.canvas)
        
        # Canvas matplotlib per il plot U(x)
        self.canvas_ux = FigureCanvas(Figure())
        self.ax_ux = self.canvas_ux.figure.subplots()
        lay_ux = QVBoxLayout(self.widgetUx)
        lay_ux.setContentsMargins(0,0,0,0)
        lay_ux.addWidget(self.canvas_ux)

        self.dwell_down_canvas = FigureCanvas(Figure())
        self.dwell_up_canvas = FigureCanvas(Figure())

        lay_dwell_down = QVBoxLayout(self.widget_dwell_down)
        lay_dwell_down.setContentsMargins(0,0,0,0)
        lay_dwell_down.addWidget(self.dwell_down_canvas)

        lay_dwell_up = QVBoxLayout(self.widget_dwell_up)
        lay_dwell_up.setContentsMargins(0,0,0,0)
        lay_dwell_up.addWidget(self.dwell_up_canvas)

        self.btn_dwell.clicked.connect(self.dwell_analysis)
        self.Plot_button.clicked.connect(self.open_and_plot_txt)
        self.btn_update.clicked.connect(self.update_trajectory_plot)
        self.btn_filter.clicked.connect(self.apply_savgol_filter)

        self.gaussian_figure = Figure()
        self.gaussian_canvas = FigureCanvas(self.gaussian_figure)
        self.verticalLayout_gaussian.addWidget(self.gaussian_canvas)
        self.pushButton_sigma.clicked.connect(self.plot_gaussian)
        self.pushButton_sigma.clicked.connect(self.deconvolvi_Ux)
        self.button_deconv.clicked.connect(self.deconvolvi_Ux)


# oppure crea un nuovo bottone se preferisci

        # Variabili dati
        self.t = None
        self.x = None
        self.U_x = None      # U(x) calcolata una sola volta
        self.bin_centers = None

    def open_and_plot_txt(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Apri TXT", "", "TXT Files (*.txt)")
        if fname:
            data = np.loadtxt(fname, skiprows=1)
            self.t = data[:, 0]
            self.x = data[:, 1]
            # Plotta traiettoria completa
            self.t_shown = self.t
            self.x_shown = self.x
            self.plot_trajectory(self.t_shown, self.x_shown)

            # Calcola e mostra U(x) SOLO UNA VOLTA (su tutti i dati!)
            self.U_x, self.bin_centers = self.compute_potential(self.x)
            self.plot_potential(self.bin_centers, self.U_x)

            # Calcola rate SOLO UNA VOLTA
            D = self.get_D_from_gui()# ... dopo aver calcolato i minimi/massimi e MFPT ...
            rate_folding, rate_unfolding, dG_fold, dG_unfold, x_folded, x_unfolded, x_barrier = self.mfpt_analysis(self.bin_centers, self.U_x, D)

            # Aggiorna le label dei rate/barriere
            self.label_folding_rate.setText(f"Folding rate: {rate_folding:.2e} s⁻¹")
            self.label_unfolding_rate.setText(f"Unfolding rate: {rate_unfolding:.2e} s⁻¹")
            self.label_folding_barrier.setText(f"Folding Barrier: {dG_fold:.2f} kBT")
            self.label_unfolding_barrier.setText(f"Unfolding Barrier: {dG_unfold:.2f} kBT")

            # Aggiorna le label delle posizioni chiave
            self.label_x_folded.setText(f"x_folded: {x_folded:.2f} nm")
            self.label_x_unfolded.setText(f"x_unfolded: {x_unfolded:.2f} nm")
            self.label_x_barrier.setText(f"x_barrier: {x_barrier:.2f} nm")


    def update_trajectory_plot(self):
        """Aggiorna SOLO la traiettoria (usando filtro tmin/tmax)"""
        if self.t is None or self.x is None:
            return
        try:
            tmin = float(self.edit_tmin.text())
        except:
            tmin = np.min(self.t)
        try:
            tmax = float(self.edit_tmax.text())
        except:
            tmax = np.max(self.t)
        mask = (self.t >= tmin) & (self.t <= tmax)
        self.t_shown = self.t[mask]
        self.x_shown = self.x[mask]
        self.plot_trajectory(self.t[mask], self.x[mask])

    def plot_trajectory(self, t, x, max_points=5_000):
        self.ax.clear()
        n = len(t)
        if n > max_points:
            idx = np.linspace(0, n-1, max_points, dtype=int)
            t_plot = t[idx]
            x_plot = x[idx]
        else:
            t_plot = t
            x_plot = x

        self.ax.plot(t_plot, x_plot)
        self.ax.set_xlabel('t[sec]')
        self.ax.set_ylabel('x[nm]')
        self.ax.set_title("Trajectory")
        self.canvas.draw()

    def apply_savgol_filter(self):
        from scipy.signal import savgol_filter

        if self.t is None or self.x is None:
            print("Carica prima la traiettoria!")
            return
        t = self.t_shown
        x = self.x_shown
        try:
            win_len = int(self.edit_window_len.text())
            if win_len < 5: win_len = 5
            if win_len % 2 == 0: win_len += 1
            if win_len >= len(self.x): 
                win_len = len(self.x) - 1
                if win_len % 2 == 0: win_len -= 1
        except Exception as e:
            print("Errore nella finestra, uso 81. Dettaglio:", e)
            win_len = 81

        x_smooth = savgol_filter(x, window_length=win_len, polyorder=2)
        self.ax.clear()
        self.ax.plot(t, x, color="gray", alpha=0.5, label="raw data")
        self.ax.plot(t, x_smooth, color="royalblue", label=f"smooth (win={win_len})")
        self.canvas.draw()

    def compute_potential(self, x):
        bins = 200
        counts, bin_edges = np.histogram(x, bins=bins, density=True)
        self.counts = counts
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        P_x = counts + 1e-12
        U_x = -np.log(P_x)
        U_x -= np.min(U_x)
        return U_x, bin_centers

    def plot_potential(self, x, U):
        self.ax_ux.clear()
        self.ax_ux.plot(x, U, label='U(x) = -log(P(x))')
        self.ax_ux.set_xlabel('x[nm]')
        self.ax_ux.set_ylabel('U(x) [kBT]')
        self.ax_ux.set_title('Potential energy profile')
        self.canvas_ux.draw()

    def get_D_from_gui(self):
        try:
            D_text = self.edit_diffusion.text()
            D = float(D_text)
        except Exception as e:
            print("Errore nel valore di D, uso D=3000")
            D = 3000
        return D
    
    def dwell_analysis(self):
    # Prendi soglie dalla GUI
        try:
            thr_down = float(self.lineEdit_thr_down.text())
            thr_up = float(self.lineEdit_thr_up.text())

        except Exception as e:
            QMessageBox.warning(self, "Errore soglie", f"Errore nelle soglie: {e}")
            return

        x = self.x  # tutta la traiettoria
        t = self.t

        # Stati: 0 (down), 1 (up), np.nan = esclusi
        state = np.full_like(x, np.nan)
        state[x < thr_down] = 0
        state[x > thr_up] = 1

        dwell_up, dwell_down = [], []
        last_state, last_t = None, None

        for i in range(len(state)):
            if np.isnan(state[i]):
                continue
            if last_state is None:
                last_state = state[i]
                last_t = t[i]
                continue
            if state[i] != last_state:
                dwell = t[i] - last_t
                if last_state == 0:
                    dwell_down.append(dwell)
                elif last_state == 1:
                    dwell_up.append(dwell)
                last_state = state[i]
                last_t = t[i]

        # Istogrammi (popup matplotlib)
       # DOWN
        self.dwell_down_canvas.figure.clear()
        ax_down = self.dwell_down_canvas.figure.subplots()
        ax_down.hist(dwell_down, bins=30, color='dodgerblue')
        ax_down.set_xlabel("Dwell time DOWN [s]")
        ax_down.set_ylabel("Count")
        ax_down.set_title("DOWN state")
        self.dwell_down_canvas.draw()

        # UP
        self.dwell_up_canvas.figure.clear()
        ax_up = self.dwell_up_canvas.figure.subplots()
        ax_up.hist(dwell_up, bins=30, color='orangered')
        ax_up.set_xlabel("Dwell time UP [s]")
        ax_up.set_ylabel("Count")
        ax_up.set_title("UP state")
        self.dwell_up_canvas.draw()

        # Calcola rate folding/unfolding
        mean_dwell_up = np.mean(dwell_up) if len(dwell_up) > 0 else np.nan
        mean_dwell_down = np.mean(dwell_down) if len(dwell_down) > 0 else np.nan
        rate_fold = 1 / mean_dwell_up if mean_dwell_up > 0 else np.nan
        rate_unfold = 1 / mean_dwell_down if mean_dwell_down > 0 else np.nan

        #aggionro label GUI
        self.label_rate_fold.setText(f"Dwell/Folding rate: {rate_fold:.4g} s⁻¹")
        self.label_rate_unfold.setText(f"Dwell/Unfolding rate: {rate_unfold:.4g} s⁻¹")

        # Stampa su console e aggiorna label GUI se esiste
        print(f"Mean dwell UP: {mean_dwell_up:.4f} s,  Rate fold: {rate_fold:.4g} 1/s")
        print(f"Mean dwell DOWN: {mean_dwell_down:.4f} s,  Rate unfold: {rate_unfold:.4g} 1/s")
        if hasattr(self, "label_rate_fold"):
            self.label_rate_fold.setText(f"Rate fold: {rate_fold:.4g} s⁻¹")
        if hasattr(self, "label_rate_unfold"):
            self.label_rate_unfold.setText(f"Rate unfold: {rate_unfold:.4g} s⁻¹")
 

    def mfpt_analysis(self, x, U, D, prominence=0.4, width=10, distance=10, debug=True):
        try:
            mask = np.isfinite(U)
            bin_c_clean = x[mask]
            U_clean = U[mask]
            U_interp = interp1d(bin_c_clean, U_clean, kind='cubic', fill_value='extrapolate')

            min_idx, _ = find_peaks(-U, prominence=prominence, width=width, distance=distance)
            max_idx, _ = find_peaks(U, prominence=prominence, width=width, distance=distance)
            if len(min_idx) < 2 or len(max_idx) < 1:
                print("Non trovati sufficienti minimi/massimi per calcolo rate")
                return (np.nan, np.nan, np.nan, np.nan)

            # Trova le posizioni chiave
            x_folded = x[min_idx[0]]
            x_unfolded = x[min_idx[1]]
            x_barrier = x[max_idx[0]]

            U_fold = U[min_idx[0]]
            U_barrier = U[max_idx[0]]
            U_unfold = U[min_idx[1]]

            dG_fold = U_barrier - U_unfold
            dG_unfold = U_barrier - U_fold

            if debug:
                print("\n==== MFPT DEBUG ====")
                print(f"x_folded (nm): {x_folded}")
                print(f"x_unfolded (nm): {x_unfolded}")
                print(f"x_barrier (nm): {x_barrier}")
                print(f"U_fold (kBT): {U_fold}")
                print(f"U_unfold (kBT): {U_unfold}")
                print(f"U_barrier (kBT): {U_barrier}")
                print(f"Diffusion D (nm^2/s): {D}")
                print(f"Folding barrier dG_fold (kBT): {dG_fold}")
                print(f"Unfolding barrier dG_unfold (kBT): {dG_unfold}")
                print(f"x range folding: [{x_folded}, {x_unfolded}]")
                print(f"Potenziale U min/max: {np.min(U)}, {np.max(U)}")

            # Folding: da folded → unfolded (attraversando la barriera)
            xs = np.linspace(x_folded, x_unfolded, 2000)
            Us = U_interp(xs)
            expU = np.exp(Us)
            exp_minusU = np.exp(-Us)
            inner = np.cumsum(exp_minusU) * (xs[1] - xs[0])
            outer = np.sum(expU * inner) * (xs[1] - xs[0])
            mfpt_fold = outer / D

            # Unfolding: da unfolded → folded (direzione opposta)
            xs2 = np.linspace(x_unfolded, x_folded, 2000)
            Us2 = U_interp(xs2)
            expU2 = np.exp(Us2)
            exp_minusU2 = np.exp(-Us2)
            inner2 = np.cumsum(exp_minusU2) * (xs2[1] - xs2[0])
            outer2 = np.sum(expU2 * inner2) * (xs2[1] - xs2[0])
            mfpt_unfold = outer2 / D

            # I rate sono l'inverso degli MFPT nei due sensi
            rate_folding = 1 / mfpt_unfold if mfpt_unfold > 0 else np.nan
            rate_unfolding = 1 / mfpt_fold if mfpt_fold > 0 else np.nan

            if debug:
                print(f"MFPT folding (s): {mfpt_fold}")
                print(f"MFPT unfolding (s): {mfpt_unfold}")
                print(f"Rate folding (1/s): {rate_folding}")
                print(f"Rate unfolding (1/s): {rate_unfolding}")
                if (mfpt_fold < 1e-6) or (mfpt_unfold < 1e-6):
                    print("** WARNING: MFPT molto basso, controlla le barriere e le unità! **")
                if (dG_fold < 2) or (dG_unfold < 2):
                    print("** WARNING: Barriera inferiore a 2 kBT, rate saranno fisicamente molto alti! **")
                print("====================\n")
            
            return rate_folding, rate_unfolding, dG_fold, dG_unfold, x_folded, x_unfolded, x_barrier

        except Exception as e:
            print("Errore calcolo rate:", e)
            return (np.nan, np.nan, np.nan, np.nan)
        


        # deconvolution 
    def plot_gaussian(self):
        try:
            sigma = float(self.sigma_input.text())   # sigma_input = QLineEdit nel .ui
            if sigma <= 0:
                raise ValueError("Sigma must be positive.")
        except Exception as e:
            print(f"Errore sigma: {e}")
            return

        import numpy as np
        import matplotlib.pyplot as plt

        x = np.linspace(-5*sigma, 5*sigma, 500)
        y = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*(x/sigma)**2)

        self.gaussian_figure.clear()  # gaussian_figure = matplotlib Figure promoted nel .ui
        ax = self.gaussian_figure.add_subplot(111)
        ax.plot(x, y, label=f"σ = {sigma}")
        ax.set_title("Gaussian Distribution")
        ax.legend()
        self.gaussian_canvas.draw()   # gaussian_canvas = FigureCanvas associato

    def jansson_deconvolution(self, observed, kernel, iterations=700, alpha=0.02):
        from scipy.signal import fftconvolve
        estimate = np.copy(observed).astype(np.float64)
        for _ in range(iterations):
            reconvolved = fftconvolve(estimate, kernel, mode='same')
            correction = observed - reconvolved
            update = alpha * correction * estimate
            estimate += update
            estimate[estimate < 0] = 0
        return estimate

    def deconvolvi_Ux(self):
        try:
            sigma = float(self.sigma_input.text())
            alpha = float(self.alpha_input.text())
            n_iter = int(self.niter_input.text())
        except Exception as e:
            print(f"Errore nei parametri: {e}")
            return

        # Costruisci la PSF gaussiana (kernel)
        L = int(6 * sigma + 1)
        if L % 2 == 0:
            L += 1
        x = np.linspace(-3*sigma, 3*sigma, L)
        kernel = np.exp(-x**2 / (2*sigma**2))
        kernel /= kernel.sum()
        kernel = fftshift(kernel)

        # Usa i dati salvati nell'oggetto
        restored = self.jansson_deconvolution(self.counts, kernel, iterations=n_iter, alpha=alpha)
        # Rinormalizza
        bin_centers = self.bin_centers
        restored /= np.sum(restored) * (bin_centers[1] - bin_centers[0])
        U_dec = -np.log(restored + 1e-10)
        shift = bin_centers[np.argmin(self.U_x)] - bin_centers[np.argmin(U_dec)]
        bin_centers_shifted = bin_centers + shift
        U_dec_shifted = U_dec - np.min(U_dec)

        # Plot su ax_ux (come detto sopra)
        self.ax_ux.clear()
        self.ax_ux.plot(bin_centers, self.U_x, label="drift removed (U(x))", color="green")
        self.ax_ux.plot(bin_centers_shifted, U_dec_shifted, label="deconvolution", color="red")
        self.ax_ux.set_xlabel('x [nm]')
        self.ax_ux.set_ylabel('U(x) [kBT]')
        self.ax_ux.set_title('Potential energy profile')
        self.ax_ux.legend()
        self.canvas_ux.draw()


        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MyMainWindow()
    win.show()
    sys.exit(app.exec_())
