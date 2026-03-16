import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget

# Importa le tue funzioni dai file modificati
from PyQt6_Deconvolution import launch_Deconvolution
from PyQt6_Bell_fit import launch_Bell_fit
from PyQt6_W_wlc import launch_W_wlc

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Analysis")
        self.setGeometry(100, 100, 300, 200)

        self.deconv_window = None
        self.bell_window = None
        self.wlc_window = None

        layout = QVBoxLayout()

        btn_ramp = QPushButton("Constant Force Analysis")
        btn_ramp.clicked.connect(launch_Deconvolution)
        layout.addWidget(btn_ramp)

        self.deconvolution_button = QPushButton("Deconvolution Analysis")
        self.deconvolution_button.clicked.connect(self.open_deconv)  # Collegamento alla funzione
        layout.addWidget(self.deconvolution_button)

        btn_jarz = QPushButton("Unfolding F analysis")
        btn_jarz.clicked.connect(launch_Bell_fit)
        layout.addWidget(btn_jarz)

        btn_cf = QPushButton("Jarzynski Work analusis")
        btn_cf.clicked.connect(launch_W_wlc)
        layout.addWidget(btn_cf)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def open_deconv(self):
        self.deconv_window = launch_Deconvolution()

    def open_bell(self):
        self.bell_window = launch_Bell_fit()

    def open_wlc(self):
        self.wlc_window = launch_W_wlc()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
