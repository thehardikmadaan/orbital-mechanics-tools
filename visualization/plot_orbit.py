# visualisation / plot_orbit.py
import os
import numpy as np
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

class OrbitPlotter(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        # matplotlib figure
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor='#080d14') # Changed 5 to 8
        self.ax = self.fig.add_subplot(111)

        # convert the canvas to pyside6 widget
        super().__init__(self.fig)
        self.setParent(parent)

        # Inital plot config
        self.ax.set_facecolor('#080d14')
        self.ax.tick_params(colors='#5a6a80')
        for spine in self.ax.spines.values():
            spine.set_color((0.0, 0.831, 1.0, 0.12))

        # draw a blank template to start
        self.draw_orbits(300, 35786)

    def draw_orbits(self, alt1_km, alt2_km, maneuver="Hohmann Transfer", rb_km=None):
        """Clears the canvas and redraws the orbits based on user input."""
        self.ax.clear() # clear prev drawing

        r_earth = 6371
        r1 = r_earth + alt1_km
        r2 = r_earth + alt2_km

        theta = np.linspace(0, 2 * np.pi, 100)

        # 1. Base Orbits
        earth_x, earth_y = r_earth * np.cos(theta), r_earth * np.sin(theta)
        leo_x, leo_y = r1 * np.cos(theta), r1 * np.sin(theta)
        geo_x, geo_y = r2 * np.cos(theta), r2 * np.sin(theta)

        self.ax.fill(earth_x, earth_y, color='#00d4ff', alpha=0.3, label="Earth")
        self.ax.plot(leo_x, leo_y, color='#7b61ff', linestyle='--', label="Initial Orbit")
        self.ax.plot(geo_x, geo_y, color='#ff6b35', linestyle='--', label="Target Orbit")

        # 2. Trajectory Logic (We will store the flight path here)
        self.flight_path_x = []
        self.flight_path_y = []

        if maneuver == "Hohmann Transfer":
            a = (r1 + r2) / 2
            c = a - r1
            b = np.sqrt(a ** 2 - c ** 2)
            theta_transfer = np.linspace(0, np.pi, 100)

            self.flight_path_x = a * np.cos(theta_transfer) - c
            self.flight_path_y = b * np.sin(theta_transfer)

            self.ax.plot(self.flight_path_x, self.flight_path_y, color='#00d4ff', linewidth=2, label="Transfer Path")

        # 3. The Spacecraft Marker (Starts hidden)
        # We use 'wo' which means White Circle (o)
        self.rocket_marker, = self.ax.plot([], [], 'wo', markersize=8, zorder=5, label="Spacecraft")

        self.ax.set_aspect('equal')
        # Hide the raw axis numbers for a cleaner HUD look
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        # Push the legend outside the plot to the right
        self.ax.legend(
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),  # 1.05 puts it just outside the right edge!
            facecolor='#020408',
            edgecolor='none',
            labelcolor='#e8edf5'
        )

        # Adjust the margins so the legend doesn't get cut off by the window edge
        self.fig.subplots_adjust(right=0.75, left=0.05, top=0.95, bottom=0.05)
        self.draw()

        # 3. The Spacecraft Marker
        try:
            # Get the absolute path to the project root folder
            # This assumes your project root is the current working directory
            current_folder = os.getcwd()
            image_path = os.path.join(current_folder, 'rocket.png')

            # Load the image
            rocket_img = mpimg.imread(image_path)

            # zoom=0.05 shrinks the image. If you still can't see it after fixing the path,
            # change this to zoom=0.2 or 0.5!
            self.imagebox = OffsetImage(rocket_img, zoom=0.05)

            # Place the image box
            self.rocket_marker = AnnotationBbox(self.imagebox, (0, 0), frameon=False)
            self.rocket_marker.set_visible(False)

            # Set zorder very high so it always draws ON TOP of the lines
            self.rocket_marker.set_zorder(10)

            self.ax.add_artist(self.rocket_marker)
            self.using_image = True

        except FileNotFoundError:
            # THIS WILL PRINT TO YOUR TERMINAL IF IT FAILS!
            self.rocket_marker, = self.ax.plot([], [], 'wo', markersize=8, zorder=5, label="Spacecraft")
            self.using_image = False

    def update_rocket_position(self, x, y):
        """Moves the white dot to a new coordinate and redraws just the canvas."""
        self.rocket_marker.set_data([x], [y])
        self.draw()