# visualisation / plot_orbit.py

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

class OrbitPlotter(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        # matplotlib figure
        self.fig = Figure(figsize=(5,5), dpi=100, facecolor='#080d14')
        self.ax = self.fig.add_subplot(111)

        # convert the canvas to pyside6 widget
        super().__init__(self.fig)
        self.setParent(parent)

        # Inital plot config
        self.ax.set_facecolor('#080d14')
        self.ax.tick_params(colors='#5a6a80')
        for spine in self.ax.spines.values():
            spine.set_color('rgba(0, 212, 255, 0.12)')

        # draw a blank template to start
        self.draw_orbits(300, 35786)

    def draw_orbits(self, alt1_km, alt2_km):
        """Clears the canvas and redraws the orbits based on user input."""