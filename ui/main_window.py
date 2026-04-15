# ui

import sys
import math
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QFrame, QGridLayout)
from PySide6.QtCore import Qt

from core.astrodynamics import hohmann_transfer
from core.rocket_math import calculate_initial_mass

class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        # Window properties
        self.setWindowTitle("Orbital Mechanics Flight Planner")
        self.setGeometry(100,100,800,600)
        # Style sheet
        self.setStyleSheet("""
                    QMainWindow {
                        background-color: #020408;
                    }
                    QLabel {
                        color: #e8edf5;
                        font-family: 'Outfit', sans-serif;
                        font-size: 14px;
                    }
                    QLabel#Header {
                        color: #00d4ff;
                        font-family: 'Syne', sans-serif;
                        font-size: 28px;
                        font-weight: 800;
                        letter-spacing: 1px;
                    }
                    QLabel#Eyebrow {
                        color: #7b61ff;
                        font-family: 'DM Mono', monospace;
                        font-size: 12px;
                        text-transform: uppercase;
                        letter-spacing: 2px;
                    }
                    QFrame#Panel {
                        background-color: #080d14;
                        border: 1px solid rgba(0, 212, 255, 0.12);
                        border-radius: 8px;
                    }
                    QLineEdit {
                        background-color: #020408;
                        color: #00d4ff;
                        border: 1px solid rgba(0, 212, 255, 0.3);
                        border-radius: 4px;
                        padding: 10px;
                        font-family: 'DM Mono', monospace;
                        font-size: 14px;
                    }
                    QLineEdit:focus {
                        border: 1px solid #00d4ff;
                        background-color: rgba(0, 212, 255, 0.05);
                    }
                    QPushButton {
                        background-color: transparent;
                        color: #00d4ff;
                        border: 1px solid #00d4ff;
                        border-radius: 4px;
                        padding: 12px;
                        font-family: 'Syne', sans-serif;
                        font-weight: 700;
                        font-size: 14px;
                        letter-spacing: 1px;
                    }
                    QPushButton:hover {
                        background-color: #00d4ff;
                        color: #020408;
                    }
                    QLabel#ResultLabel {
                        color: #ff6b35;
                        font-family: 'DM Mono', monospace;
                        font-size: 16px;
                        margin-top: 20px;
                    }
                """)

        #center container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        # Headers

        eyebrow = QLabel("Mission Control || Trajectory Planning")
        eyebrow.setObjectName("Eyebrow")
        header = QLabel("HOHMANN TRANSFER")
        header.setObjectName("Header")
        main_layout.addWidget(eyebrow)
        main_layout.addWidget((header))

        # input panel
        panel = QFrame()
        panel.setObjectName("Panel")
        panel_layout = QGridLayout(panel)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(15)

        #Input fields
        panel_layout.addWidget(QLabel("Initial LEO Altitude (km):"), 0, 0)
        self.input_r1 = QLineEdit("300")
        panel_layout.addWidget(self.input_r1, 0, 1)

        panel_layout.addWidget(QLabel("Target GEO Altitude (km):"), 1, 0)
        self.input_r2 = QLineEdit("35786")
        panel_layout.addWidget(self.input_r2, 1, 1)

        panel_layout.addWidget(QLabel("Payload Mass (kg):"), 2, 0)
        self.input_mass = QLineEdit("2000")
        panel_layout.addWidget(self.input_mass, 2, 1)

        #UI Elements
        self.leo_label = QLabel("Starting LEO Altitude (km):")
        self.leo_input = QLineEdit("300") # default text
        self.calc_button = QPushButton("Calculate Transfer")


        #Adding widgets to layout
        layout.addWidget(self.leo_label)
        layout.addWidget(self.leo_input)
        layout.addWidget(self.calc_button)

        central_widget.setLayout(layout)




def run_app():
    # creating app
    app = QApplication(sys.argv)

    #create and show
    window = Dashboard()
    window.show()
    # run the main
    sys.exit(app.exec())

if __name__ == "__main__":
    run_app()
