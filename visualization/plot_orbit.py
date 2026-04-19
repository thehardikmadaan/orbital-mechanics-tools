# visualization/plot_orbit.py
# Orbital diagram renderer and spacecraft animator.
#
# This module draws the 2D coplanar orbital diagram using Matplotlib embedded
# inside the Qt window (FigureCanvasQTAgg). It shows:
#   - The central body (Earth, Moon, or Mars) as a filled circle
#   - The parking orbit and target orbit as dashed rings
#   - The transfer path (Hohmann arc, Bi-Elliptic double arc, or Phasing loop)
#   - A rocket sprite that animates along the transfer path
#   - A velocity vector arrow showing the spacecraft's instantaneous direction
#   - Burn markers (gold circles) at the points where engines fire
#
# The diagram is NOT to scale — if it were, Earth would be a 1-pixel dot inside
# a GEO orbit the size of this screen. We normalise to the orbit radii so
# everything is readable.

import os
import numpy as np
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

# Body colour accents for the filled circle in the centre of the diagram
BODY_COLOURS = {
    "Earth": "#00d4ff",   # Cyan — water and atmosphere
    "Moon":  "#aaaacc",   # Pale grey-blue — lunar regolith
    "Mars":  "#d05030",   # Rust red — Martian surface
}

# Mean radius in km for each body — used for scaling the central body circle
BODY_RADIUS_KM = {
    "Earth": 6371.0,
    "Moon":  1737.4,
    "Mars":  3389.5,
}


class OrbitPlotter(FigureCanvasQTAgg):
    """
    Matplotlib canvas that lives inside the Qt window.

    FigureCanvasQTAgg is the Qt 'widget' that wraps a Matplotlib Figure.
    We subclass it so Qt treats this canvas like any other widget and can
    resize it, add it to layouts, etc.
    """

    def __init__(self, parent=None):
        # Create the figure and axes (the actual plot area inside the figure)
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor='#080d14')
        self.ax  = self.fig.add_subplot(111)

        super().__init__(self.fig)
        self.setParent(parent)

        # Style the axes to match the dark HMI theme
        self.ax.set_facecolor('#080d14')
        self.ax.tick_params(colors='#2a3a50')

        # Make the axis spines (border lines) very subtle — just a hint of cyan
        for spine in self.ax.spines.values():
            spine.set_color((0.0, 0.831, 1.0, 0.08))

        # Flight path coordinates — updated by draw_orbits(), read by main_window.py
        # for the frame-by-frame animation
        self.flight_path_x = []
        self.flight_path_y = []
        self.using_image   = False

        # Draw the default view on startup (Earth LEO → GEO)
        self.draw_orbits(300, 35786, body="Earth")

    def draw_orbits(self, alt1_km, alt2_km, maneuver="Hohmann Transfer",
                    rb_km=None, body="Earth"):
        """
        Clear the canvas and draw a fresh orbital diagram for the given scenario.

        Inputs:
          alt1_km  — Parking orbit altitude above surface (km)
          alt2_km  — Target orbit altitude above surface (km)
          maneuver — Transfer type string (matches the UI combo box text)
          rb_km    — Bi-Elliptic intermediate apogee altitude (km), or None
          body     — Central body name ("Earth", "Moon", "Mars")
        """
        self.ax.clear()

        # Look up this body's radius and display colour
        r_body  = BODY_RADIUS_KM.get(body, 6371.0)
        colour  = BODY_COLOURS.get(body, "#00d4ff")

        # Absolute orbital radii in km from body centre
        r1 = r_body + alt1_km
        r2 = r_body + alt2_km

        # Full-circle angle array for drawing complete circular orbits
        theta = np.linspace(0, 2 * np.pi, 300)

        # ── Central body ───────────────────────────────────────────────────────
        # The filled circle represents the planet/moon. We add two layers:
        # a solid core and a soft 'atmosphere' glow ring for visual depth.
        body_x = r_body * np.cos(theta)
        body_y = r_body * np.sin(theta)
        self.ax.fill(body_x, body_y, color=colour, alpha=0.25, zorder=2)
        self.ax.plot(body_x, body_y, color=colour, linewidth=1.5, alpha=0.6,
                     zorder=3, label=body)

        # Atmosphere glow: a slightly larger ring at 10% opacity
        atm_r = r_body * 1.06
        self.ax.plot(atm_r * np.cos(theta), atm_r * np.sin(theta),
                     color=colour, linewidth=6, alpha=0.06, zorder=1)

        # ── Parking orbit (r1) ─────────────────────────────────────────────────
        self.ax.plot(r1 * np.cos(theta), r1 * np.sin(theta),
                     color='#7b61ff', linestyle='--', linewidth=1.2,
                     alpha=0.7, zorder=4, label=f"Parking Orbit ({alt1_km:,.0f} km)")

        # ── Target orbit (r2) — skip for Phasing (r2 == r1) ───────────────────
        if maneuver != "Phasing Orbit":
            self.ax.plot(r2 * np.cos(theta), r2 * np.sin(theta),
                         color='#ff6b35', linestyle='--', linewidth=1.2,
                         alpha=0.7, zorder=4, label=f"Target Orbit ({alt2_km:,.0f} km)")

        # ── Transfer path and flight path coordinates ──────────────────────────
        self.flight_path_x = []
        self.flight_path_y = []

        # We use 200 points per arc — enough for smooth animation without
        # making the frame updates sluggish.
        N = 200

        if maneuver == "Hohmann Transfer":
            # A Hohmann transfer traces the upper half of an ellipse.
            # The ellipse has:
            #   perigee at r1 (left side, θ = π)  → launch point
            #   apogee  at r2 (right side, θ = 0)
            # We draw from θ = π → 0 to fly left-to-right visually.
            a  = (r1 + r2) / 2         # Semi-major axis of transfer ellipse
            c  = a - r1                # Distance from ellipse centre to focus
            b  = np.sqrt(a**2 - c**2)  # Semi-minor axis (Pythagorean theorem)

            theta_arc = np.linspace(0, np.pi, N)
            self.flight_path_x = a * np.cos(theta_arc) - c
            self.flight_path_y = b * np.sin(theta_arc)

            self.ax.plot(self.flight_path_x, self.flight_path_y,
                         color='#00d4ff', linewidth=2, zorder=5,
                         label="Transfer Path")

            # Burn markers: gold circles at perigee (r1) and apogee (r2)
            self._mark_burn(r1, 0,    "Burn 1")   # Starting point on r1 (θ=0)
            self._mark_burn(-r2, 0,   "Burn 2")   # Arrival point on r2 (θ=π)

        elif maneuver == "Bi-Elliptic Transfer":
            if rb_km is None:
                rb_km = 100000
            rb = r_body + rb_km

            # ── First transfer ellipse: r1 (perigee) → rb (apogee) ────────────
            a1 = (r1 + rb) / 2
            c1 = a1 - r1
            b1 = np.sqrt(a1**2 - c1**2)
            th1 = np.linspace(0, np.pi, N)
            x1  = a1 * np.cos(th1) - c1
            y1  = b1 * np.sin(th1)

            # ── Second transfer ellipse: rb (apogee) → r2 (perigee) ───────────
            a2 = (r2 + rb) / 2
            c2 = (rb - r2) / 2    # Centre shifts because rb > r2
            b2 = np.sqrt(a2**2 - c2**2)
            th2 = np.linspace(np.pi, 2 * np.pi, N)
            x2  = a2 * np.cos(th2) - c2
            y2  = b2 * np.sin(th2)

            # Concatenate so the animation flows continuously through both arcs
            self.flight_path_x = np.concatenate((x1, x2))
            self.flight_path_y = np.concatenate((y1, y2))

            self.ax.plot(x1, y1, color='#00d4ff', linewidth=2, zorder=5,
                         label="Transfer 1 (Outbound)")
            self.ax.plot(x2, y2, color='#7b61ff', linewidth=2,
                         linestyle='-.', zorder=5, label="Transfer 2 (Inbound)")

            # Three burn markers
            self._mark_burn(r1, 0,    "Burn 1")
            self._mark_burn(-rb, 0,   "Burn 2")
            self._mark_burn(r2, 0,    "Burn 3")

        elif maneuver == "Phasing Orbit":
            # A phasing orbit is a slightly smaller (faster) loop within r1.
            # It starts and ends at the same point on r1, completing one
            # full revolution to catch up to the target ahead.
            a_ph = r1 * 0.92          # Smaller semi-major axis → shorter period
            c_ph = r1 - a_ph          # Offset so perigee touches r1
            b_ph = np.sqrt(a_ph**2 - c_ph**2)

            th_ph = np.linspace(0, 2 * np.pi, N)
            self.flight_path_x = a_ph * np.cos(th_ph) + c_ph
            self.flight_path_y = b_ph * np.sin(th_ph)

            self.ax.plot(self.flight_path_x, self.flight_path_y,
                         color='#7b61ff', linewidth=2, zorder=5,
                         label="Phasing Loop")

            # Two burns: enter and exit the phasing orbit at the same point
            self._mark_burn(r1, 0, "Burn 1 + 2")

        # ── Reference orbit labels (small annotations at the right edge) ───────
        self.ax.annotate(
            f" h₁ = {alt1_km:,.0f} km",
            xy=(r1, 0), fontsize=8,
            color='#7b61ff', va='center', alpha=0.8
        )
        if maneuver != "Phasing Orbit":
            self.ax.annotate(
                f" h₂ = {alt2_km:,.0f} km",
                xy=(r2, 0), fontsize=8,
                color='#ff6b35', va='center', alpha=0.8
            )

        # ── Axis formatting ────────────────────────────────────────────────────
        # Equal aspect ratio is critical — without it, circular orbits look elliptical.
        self.ax.set_aspect('equal')
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_title(
            f"Orbital Diagram — {body}  [{maneuver}]",
            color='#5a6a80', fontsize=9, pad=8,
            fontfamily='monospace'
        )

        # Legend outside the plot area to avoid covering the orbits
        self.ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            facecolor='#0a121e',
            edgecolor=(0, 212/255, 255/255, 0.1),
            labelcolor='#9aacbc',
            fontsize=9,
            framealpha=0.9,
        )
        self.fig.subplots_adjust(right=0.72, left=0.04, top=0.94, bottom=0.04)
        self.draw()

        # ── Load the rocket sprite for animation ───────────────────────────────
        try:
            current_folder = os.path.dirname(__file__)
            image_path = os.path.join(current_folder, 'rocket.png')

            rocket_img = mpimg.imread(image_path)
            rocket_img = np.rot90(rocket_img, k=2)    # Rotate 180° (nose forward)
            self.imagebox = OffsetImage(rocket_img, zoom=0.09)

            self.rocket_marker = AnnotationBbox(
                self.imagebox, (0, 0), frameon=False, zorder=10
            )
            self.rocket_marker.set_visible(False)
            self.ax.add_artist(self.rocket_marker)
            self.using_image = True

        except FileNotFoundError:
            # If the image is missing, fall back to a simple white dot
            self.rocket_marker, = self.ax.plot(
                [], [], 'wo', markersize=7, zorder=10, label="Spacecraft"
            )
            self.using_image = False

        # Velocity vector arrow (prograde = cyan, retrograde = orange)
        # FancyArrowPatch draws an arrow between two (x, y) coordinate pairs.
        # mutation_scale controls the arrowhead size; zorder=20 puts it on top.
        self.vector_arrow = FancyArrowPatch(
            (0, 0), (0, 0),
            mutation_scale=12,
            color='#00d4ff',
            zorder=20,
            arrowstyle='->',
            lw=1.8
        )
        self.ax.add_patch(self.vector_arrow)
        self.vector_arrow.set_visible(False)

    def _mark_burn(self, x, y, label):
        """
        Draws a gold circle at a burn location and labels it.

        These markers show exactly where the engine fires — matching how ESA's
        GMAT and NASA's STK display mission events on trajectory plots.
        """
        self.ax.plot(x, y, 'o', color='#ffd700', markersize=7,
                     zorder=15, alpha=0.9)
        self.ax.annotate(
            f" {label}", xy=(x, y),
            fontsize=7.5, color='#ffd700',
            va='bottom', alpha=0.85, zorder=16
        )

    def update_rocket_position(self, x, y, dx=0, dy=0, vector_type="Prograde"):
        """
        Move the spacecraft sprite and velocity arrow to a new position.

        Called once per animation frame from main_window.animation_step().

        Inputs:
          x, y        — Current position in km (plot coordinates)
          dx, dy      — Velocity direction vector (unnormalised tangent)
          vector_type — "Prograde" (forward burn) or "Retrograde" (brake burn)
        """
        if self.using_image:
            self.rocket_marker.xybox = (x, y)
            self.rocket_marker.set_visible(True)
        else:
            self.rocket_marker.set_data([x], [y])

        # Draw the velocity vector arrow if we have a direction to point in
        if dx != 0 or dy != 0:
            # Scale the vector to a fixed visual length regardless of orbit size
            # 5000 km is a reasonable arrow length for most orbit diagrams
            magnitude = np.hypot(dx, dy)
            scale = 5000 / magnitude

            if vector_type == "Retrograde":
                # Retrograde = engine pointing forward, decelerating
                self.vector_arrow.set_color('#ff6b35')
                end_x = x - dx * scale
                end_y = y - dy * scale
            else:
                # Prograde = engine pointing backward, accelerating
                self.vector_arrow.set_color('#00d4ff')
                end_x = x + dx * scale
                end_y = y + dy * scale

            self.vector_arrow.set_positions((x, y), (end_x, end_y))
            self.vector_arrow.set_visible(True)
        else:
            # No direction = engines off; hide the arrow
            self.vector_arrow.set_visible(False)

        self.draw()
