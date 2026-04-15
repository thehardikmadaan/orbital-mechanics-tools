# ui

import sys
from PySide6.QtWidgets import QApplication, QMainWindow

class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        # Window properties
        self.setWindowTitle("Orbital Mechanics Flight Planner")
        self.setGeometry(100,100,800,600)

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
