# ui

import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QLabel, QLineEdit, QPushButton)

class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        # Window properties
        self.setWindowTitle("Orbital Mechanics Flight Planner")
        self.setGeometry(100,100,800,600)

        #center container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        #Vertical Layout
        layout = QVBoxLayout

        #UI Elements
        self.leo_label = QLabel("Starting LEO Altitude (km):")
        self.leo_input = QLineEdit("300") # default text
        self.calc_button = QPushButton("Calculate Transfer")




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
