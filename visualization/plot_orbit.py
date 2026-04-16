# visualisation / plot_orbit.py
import os
import numpy as np
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

class OrbitPlotter(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor='#080d14')
        self.ax = self.fig.add_subplot(111)

        super().__init__(self.fig)
        self.setParent(parent)

        self.ax.set_facecolor('#080d14')
        self.ax.tick_params(colors='#5a6a80')
        for spine in self.ax.spines.values():
            spine.set_color((0.0, 0.831, 1.0, 0.12))

        self.draw_orbits(300, 35786)

    def draw_orbits(self, alt1_km, alt2_km, maneuver="Hohmann Transfer", rb_km=None):
        """Clears the canvas and redraws the orbits based on user input."""
        self.ax.clear()

        r_earth = 6371
        r1 = r_earth + alt1_km
        r2 = r_earth + alt2_km

        theta = np.linspace(0, 2 * np.pi, 100)

        earth_x, earth_y = r_earth * np.cos(theta), r_earth * np.sin(theta)
        leo_x, leo_y = r1 * np.cos(theta), r1 * np.sin(theta)
        geo_x, geo_y = r2 * np.cos(theta), r2 * np.sin(theta)

        self.ax.fill(earth_x, earth_y, color='#00d4ff', alpha=0.3, label="Earth")
        self.ax.plot(leo_x, leo_y, color='#7b61ff', linestyle='--', label="Initial Orbit")
        self.ax.plot(geo_x, geo_y, color='#ff6b35', linestyle='--', label="Target Orbit")

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
        elif maneuver == "Bi-Elliptic Transfer":
            if rb_km is None:
                rb_km = 100000  # Fallback safety
            rb = r_earth + rb_km

            # 1. First Transfer (Outward to Deep Space)
            a1 = (r1 + rb) / 2
            c1 = a1 - r1
            b1 = np.sqrt(a1 ** 2 - c1 ** 2)
            theta1 = np.linspace(0, np.pi, 100)
            x1 = a1 * np.cos(theta1) - c1
            y1 = b1 * np.sin(theta1)

            # 2. Second Transfer (Inward to Target Orbit)
            a2 = (r2 + rb) / 2
            c2 = (rb - r2) / 2  # The center of this ellipse is shifted differently
            b2 = np.sqrt(a2 ** 2 - c2 ** 2)
            # The second ellipse starts at deep space (angle pi) and flies to r2 (angle 2*pi)
            theta2 = np.linspace(np.pi, 2 * np.pi, 100)
            x2 = a2 * np.cos(theta2) - c2
            y2 = b2 * np.sin(theta2)

            # Combine the two paths so the animation flows perfectly!
            self.flight_path_x = np.concatenate((x1, x2))
            self.flight_path_y = np.concatenate((y1, y2))

            # Plot the two distinct transfer lines
            self.ax.plot(x1, y1, color='#00d4ff', linewidth=2, label="Transfer 1 (Outward)")
            self.ax.plot(x2, y2, color='#7b61ff', linewidth=2, linestyle='-.', label="Transfer 2 (Inward)")


        self.ax.set_aspect('equal')
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.legend(
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            facecolor='#020408',
            edgecolor='none',
            labelcolor='#e8edf5'
        )
        self.fig.subplots_adjust(right=0.75, left=0.05, top=0.95, bottom=0.05)
        self.draw()

        try:
            # __file__ always points to this script's directory — reliable regardless of cwd
            current_folder = os.path.dirname(__file__)
            image_path = os.path.join(current_folder, 'rocket.png')

            rocket_img = mpimg.imread(image_path)
            rocket_img = np.rot90(rocket_img, k=2)  # Rotate 180°
            self.imagebox = OffsetImage(rocket_img, zoom=0.1)

            self.rocket_marker = AnnotationBbox(self.imagebox, (0, 0), frameon=False)
            self.rocket_marker.set_visible(False)
            self.rocket_marker.set_zorder(10)

            self.ax.add_artist(self.rocket_marker)
            self.using_image = True

            # 4. Thrust/Velocity Vector Arrow (Starts hidden)
            # shrinkA and shrinkB prevent the arrow from detaching from the coordinates
            self.vector_arrow = self.ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                                 arrowprops=dict(arrowstyle="->", color='#00d4ff', lw=2, shrinkA=0,
                                                                 shrinkB=0),
                                                 zorder=9)
            self.vector_arrow.set_visible(False)

        except FileNotFoundError:
            self.rocket_marker, = self.ax.plot([], [], 'wo', markersize=8, zorder=5, label="Spacecraft")
            self.using_image = False

    def update_rocket_position(self, x, y, dx=0, dy=0, vector_type="Prograde"):
        """Moves the spacecraft and its directional thrust vector."""
        if self.using_image:
            self.rocket_marker.xybox = (x, y)
            self.rocket_marker.set_visible(True)
        else:
            self.rocket_marker.set_data([x], [y])

        # Handle the Vector Arrow
        if dx != 0 or dy != 0:
            # Scale factor: forces the arrow to be 6000km long visually so it stands out
            scale = 6000 / np.hypot(dx, dy)

            if vector_type == "Retrograde":
                self.vector_arrow.arrow_patch.set_color('#ff6b35')  # Orange for braking
                self.vector_arrow.xy = (x - dx * scale, y - dy * scale)  # Point backward
            else:
                self.vector_arrow.arrow_patch.set_color('#00d4ff')  # Cyan for speeding up
                self.vector_arrow.xy = (x + dx * scale, y + dy * scale)  # Point forward

            self.vector_arrow.set_position((x, y))  # Lock the base of the arrow to the rocket
            self.vector_arrow.set_visible(True)
        else:
            # Hide the arrow when the engine is off (not moving)
            self.vector_arrow.set_visible(False)

        self.draw()