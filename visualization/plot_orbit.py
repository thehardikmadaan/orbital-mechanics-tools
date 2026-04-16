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
        self.ax.clear() # clear prev drawing

        r_earth = 6371
        r1 = r_earth + alt1_km
        r2 = r_earth + alt2_km

        # array of angles from o to 2pi
        theta = np.linspace(0, 2*np.pi, 100)
        # Earth coordinates
        earth_x = r_earth* np.cos(theta)
        earth_y = r_earth* np.sin(theta)

        #LEO coordinates (r1)
        leo_x = r1* np.cos(theta)
        leo_y = r1* np.sin(theta)

        #GEO coordinates (r2)
        geo_x = r2* np.cos(theta)
        geo_y = r2* np.sin(theta)

        # plotting lines
        self.ax.fill(earth_x, earth_y, color='#00d4ff', alpha=0.3, label="Earth")
        self.ax.plot(leo_x, leo_y, color='#7b61ff', linestyle='--', label="Initial Orbit")
        self.ax.plot(geo_x, geo_y, color='#ff6b35', linestyle='--', label="Target Orbit")

        # Formatting
        self.ax.set_aspect('equal')  # Prevents circular orbits from looking like ovals
        self.ax.legend(loc='upper right', facecolor='#020408', edgecolor='none', labelcolor='#e8edf5')
        self.fig.tight_layout()
        self.draw()