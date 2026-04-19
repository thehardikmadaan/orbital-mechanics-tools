# ui/main_window.py
# The Mission Control Dashboard — the main interface for the Orbital Transfer System.
#
# This window ties together all three layers of the project:
#   1. Physics engine (core/) — the ground-truth calculations
#   2. AI surrogate (ml/)     — fast learned predictions for comparison
#   3. Visualisation (visualization/) — animated orbital diagrams
#
# Built with PySide6, which is the Qt6 Python binding. Qt is used in real
# aerospace software (e.g., ESAC ground station tools, KDE applications on
# spacecraft simulators) because of its reliability and cross-platform support.

import sys
import os
import pandas as pd
import joblib

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFrame, QGridLayout,
    QSlider, QComboBox, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer

# Our own physics and visualisation modules
from core.astrodynamics import (
    hohmann_transfer, hohmann_transfer_time,
    bi_elliptic_transfer, bi_elliptic_transfer_time,
    phasing_maneuver, orbital_period, plane_change_dv,
    circular_velocity, BODY_PARAMS
)
from core.rocket_math import (
    calculate_initial_mass, ENGINE_PRESETS,
    mass_fraction, payload_fraction
)
from visualization.plot_orbit import OrbitPlotter


# ─────────────────────────────────────────────────────────────────────────────
# CENTRAL BODY PARAMETERS
# These values drive the orbital diagram and physics engine.
# ESA/NASA mission planners use a fixed table of parameters in their tools.
# ─────────────────────────────────────────────────────────────────────────────
BODY_RADIUS_KM = {
    "Earth": 6371.0,
    "Moon":  1737.0,
    "Mars":  3389.5
}
BODY_COLOURS = {
    "Earth": "#00d4ff",    # Cyan
    "Moon":  "#e8edf5",    # Off-White
    "Mars":  "#ff6b35"     # Orange
}


# ─────────────────────────────────────────────────────────────────────────────
# TARGET ORBIT PRESETS
# These are the standard reference orbits used by ESA/NASA mission planners.
# They give the user a quick way to select a realistic destination without
# needing to remember the altitude numbers.
# ─────────────────────────────────────────────────────────────────────────────
TARGET_PRESETS = {
    "Custom":                   None,       # User types their own value
    "LEO — Low Earth (300 km)": 300,        # ISS, Hubble, most crewed missions
    "SSO — Sun-Sync (550 km)":  550,        # Earth observation satellites
    "MEO — GPS (20,200 km)":    20200,      # GPS, Galileo, GLONASS
    "GEO — Geostationary (35,786 km)": 35786,  # TV satellites, weather, comms
    "Lunar Distance (384,400 km)":     384400,  # Moon transfer orbit apogee
}


class OrbitalDashboard(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Orbital Transfer System — Mission Control")
        self.setMinimumSize(1100, 820)
        self.setGeometry(80, 80, 1150, 900)

        # ── GLOBAL STYLESHEET ─────────────────────────────────────────────────
        # Dark space theme: #020408 background, cyan (#00d4ff) for data readouts,
        # purple (#7b61ff) for system labels, orange (#ff6b35) for warnings/targets.
        # These colours were chosen to match ESA's mission control aesthetic and
        # provide sufficient contrast for long-duration readability.
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #020408;
            }
            QLabel {
                color: #e8edf5;
                font-family: 'Outfit', 'Segoe UI', sans-serif;
                font-size: 13px;
            }
            QLabel#Header {
                color: #00d4ff;
                font-family: 'Syne', 'Segoe UI', sans-serif;
                font-size: 26px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#Eyebrow {
                color: #7b61ff;
                font-family: 'Courier New', 'DejaVu Sans Mono', monospace;
                font-size: 11px;
                letter-spacing: 3px;
            }
            QLabel#SectionLabel {
                color: #7b61ff;
                font-family: 'Courier New', 'DejaVu Sans Mono', monospace;
                font-size: 10px;
                letter-spacing: 2px;
            }
            QFrame#Panel {
                background-color: #080d14;
                border: 1px solid rgba(0, 212, 255, 0.12);
                border-radius: 8px;
            }
            QFrame#ResultPanel {
                background-color: #080d14;
                border: 1px solid rgba(0, 212, 255, 0.2);
                border-radius: 8px;
            }
            QFrame#ResultCard {
                background-color: #0d1520;
                border: 1px solid rgba(0, 212, 255, 0.08);
                border-radius: 6px;
            }
            QLabel#CardTitle {
                color: #7b61ff;
                font-family: 'Courier New', 'DejaVu Sans Mono', monospace;
                font-size: 9px;
                letter-spacing: 2px;
            }
            QLabel#CardValue {
                color: #00d4ff;
                font-family: 'Courier New', 'DejaVu Sans Mono', monospace;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#CardValueAlt {
                color: #ff6b35;
                font-family: 'Courier New', 'DejaVu Sans Mono', monospace;
                font-size: 15px;
                font-weight: 700;
            }
            QLabel#StatusOnline {
                color: #00ff88;
                font-family: 'Courier New', 'DejaVu Sans Mono', monospace;
                font-size: 11px;
                font-weight: 700;
            }
            QLabel#StatusOffline {
                color: #ff6b35;
                font-family: 'Courier New', 'DejaVu Sans Mono', monospace;
                font-size: 11px;
                font-weight: 700;
            }
            QLineEdit {
                background-color: #020408;
                color: #00d4ff;
                border: 1px solid rgba(0, 212, 255, 0.3);
                border-radius: 4px;
                padding: 8px 10px;
                font-family: 'Courier New', 'DejaVu Sans Mono', monospace;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #00d4ff;
                background-color: rgba(0, 212, 255, 0.05);
            }
            QPushButton#CalcButton {
                background-color: transparent;
                color: #00d4ff;
                border: 1px solid #00d4ff;
                border-radius: 4px;
                padding: 12px 6px;
                font-family: 'Syne', 'Segoe UI', sans-serif;
                font-weight: 700;
                font-size: 11px;
                letter-spacing: 1px;
            }
            QPushButton#CalcButton:hover {
                background-color: #00d4ff;
                color: #020408;
            }
            QPushButton#ResetButton {
                background-color: transparent;
                color: #7b61ff;
                border: 1px solid #7b61ff;
                border-radius: 4px;
                padding: 12px 6px;
                font-family: 'Syne', 'Segoe UI', sans-serif;
                font-weight: 700;
                font-size: 11px;
                letter-spacing: 1px;
            }
            QPushButton#ResetButton:hover {
                background-color: #7b61ff;
                color: #020408;
            }
            QComboBox {
                background-color: #020408;
                color: #00d4ff;
                border: 1px solid rgba(0, 212, 255, 0.3);
                border-radius: 4px;
                padding: 7px 10px;
                font-family: 'Courier New', 'DejaVu Sans Mono', monospace;
                font-size: 12px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #0d1520;
                color: #e8edf5;
                border: 1px solid rgba(0, 212, 255, 0.2);
                selection-background-color: rgba(0, 212, 255, 0.15);
            }
            QSlider::groove:horizontal {
                border: 1px solid #1a2535;
                height: 6px;
                background: #0d1520;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ff6b35;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background: rgba(0, 212, 255, 0.3);
                border-radius: 3px;
            }
        """)

        # ── MAIN LAYOUT ───────────────────────────────────────────────────────
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(30, 30, 30, 20)
        root_layout.setSpacing(16)

        # ── HEADER ROW ────────────────────────────────────────────────────────
        header_row = QHBoxLayout()

        header_left = QVBoxLayout()
        eyebrow = QLabel("MISSION CONTROL  //  TRAJECTORY PLANNING")
        eyebrow.setObjectName("Eyebrow")
        self.header = QLabel("HOHMANN TRANSFER")
        self.header.setObjectName("Header")
        header_left.addWidget(eyebrow)
        header_left.addWidget(self.header)

        # AI model status badge in the top-right corner
        self.ai_badge = QLabel("● AI: LOADING")
        self.ai_badge.setObjectName("StatusOffline")
        self.ai_badge.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        header_row.addLayout(header_left)
        header_row.addStretch()
        header_row.addWidget(self.ai_badge)
        root_layout.addLayout(header_row)

        # ── TWO-COLUMN BODY (inputs left | plot right) ────────────────────────
        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)

        # ── LEFT COLUMN: Input Panel ──────────────────────────────────────────
        left_panel = QFrame()
        left_panel.setObjectName("Panel")
        left_panel.setFixedWidth(460)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(12)

        # Section label
        inputs_label = QLabel("MISSION PARAMETERS")
        inputs_label.setObjectName("SectionLabel")
        left_layout.addWidget(inputs_label)

        grid = QGridLayout()
        grid.setSpacing(8)
        grid.setColumnStretch(1, 1.2)

        row = 0

        # Central Body selector — changes μ and radius for the physics engine
        grid.addWidget(self._field_label("Central Body:"), row, 0)
        self.body_box = QComboBox()
        self.body_box.addItems(list(BODY_PARAMS.keys()))  # Earth, Moon, Mars
        self.body_box.setToolTip(
            "Changes the gravitational parameter (μ) and radius used in all calculations.\n"
            "Earth = 3.986×10¹⁴ m³/s²   Moon = 4.905×10¹² m³/s²   Mars = 4.283×10¹³ m³/s²"
        )
        grid.addWidget(self.body_box, row, 1)
        row += 1

        # Parking Orbit input
        grid.addWidget(self._field_label("Parking Orbit (h₁, km):"), row, 0)
        self.input_r1 = QLineEdit("300")
        self.input_r1.setToolTip(
            "Altitude above the surface in km.\n"
            "Earth LEO safe floor: 200 km (below this, drag decays orbit in days).\n"
            "ISS orbits at ~420 km. Hubble at ~540 km."
        )
        grid.addWidget(self.input_r1, row, 1)
        row += 1

        # Target Orbit row (label + input + preset combo)
        self.label_r2 = self._field_label("Target Orbit (h₂, km):")
        grid.addWidget(self.label_r2, row, 0)

        target_col = QHBoxLayout()
        target_col.setSpacing(4)
        self.input_r2 = QLineEdit("35786")
        self.input_r2.setToolTip(
            "Altitude of the destination orbit in km.\n"
            "GEO = 35,786 km (satellites appear stationary over one point on Earth).\n"
            "GPS orbits at 20,200 km. Lunar transfer apogee ≈ 384,400 km."
        )
        self.preset_box = QComboBox()
        self.preset_box.addItems(list(TARGET_PRESETS.keys()))
        self.preset_box.setToolTip("Quick-fill the target altitude with a standard reference orbit.")
        self.preset_box.setFixedWidth(42)  # Narrow — acts as a dropdown trigger
        self.preset_box.setStyleSheet("QComboBox { padding: 7px 0px; font-size: 10px; }")
        target_col.addWidget(self.input_r2)
        target_col.addWidget(self.preset_box)
        grid.addLayout(target_col, row, 1)
        row += 1

        # Dry Mass input
        grid.addWidget(self._field_label("Dry Mass (m_dry, kg):"), row, 0)
        self.input_mass = QLineEdit("2000")
        self.input_mass.setToolTip(
            "Mass of the spacecraft WITHOUT propellant.\n"
            "Includes the payload, structure, and avionics — everything but fuel.\n"
            "Typical GEO comsat: 1,500–3,500 kg dry mass."
        )
        grid.addWidget(self.input_mass, row, 1)
        row += 1

        # Engine / Propellant type selector — drives the Isp value
        grid.addWidget(self._field_label("Engine Type:"), row, 0)
        self.engine_box = QComboBox()
        self.engine_box.addItems(list(ENGINE_PRESETS.keys()))
        self.engine_box.setToolTip(
            "Selects the propulsion system and its specific impulse (Isp).\n"
            "Higher Isp = more Δv per kg of fuel. Ion drives are very efficient but\n"
            "provide tiny thrust — they take months to manoeuvre instead of hours."
        )
        grid.addWidget(self.engine_box, row, 1)
        row += 1

        # Maneuver type selector
        grid.addWidget(self._field_label("Maneuver Profile:"), row, 0)
        self.maneuver_box = QComboBox()
        self.maneuver_box.addItems([
            "Hohmann Transfer",
            "Bi-Elliptic Transfer",
            "Phasing Orbit"
        ])
        self.maneuver_box.setToolTip(
            "Hohmann: 2-burn, most fuel-efficient for moderate orbit changes.\n"
            "Bi-Elliptic: 3-burn, more efficient when target is >12× initial radius.\n"
            "Phasing: 2-burn, catches up to or falls behind a target in the same orbit."
        )
        grid.addWidget(self.maneuver_box, row, 1)
        row += 1

        # ── Dynamic fields (shown/hidden by maneuver type) ─────────────────────

        # Bi-Elliptic only: deep-space intermediate apogee
        self.label_rb = self._field_label("Deep-Space Apogee (r_b, km):")
        self.input_rb = QLineEdit("100000")
        self.input_rb.setToolTip(
            "The intermediate apogee for the Bi-Elliptic maneuver.\n"
            "MUST be larger than both parking and target orbits.\n"
            "Higher rb → less fuel used (but much longer flight time)."
        )
        grid.addWidget(self.label_rb, row, 0)
        grid.addWidget(self.input_rb, row, 1)
        self.label_rb.hide()
        self.input_rb.hide()
        row += 1

        # Phasing only: phase angle
        self.label_phase = self._field_label("Phase Angle (degrees):")
        self.input_phase = QLineEdit("45")
        self.input_phase.setToolTip(
            "How far ahead the target spacecraft is in the same orbit.\n"
            "0° = same position (no maneuver needed). 180° = directly opposite.\n"
            "ISS rendezvous approaches use phase angles of 2–10° in the final burn."
        )
        grid.addWidget(self.label_phase, row, 0)
        grid.addWidget(self.input_phase, row, 1)
        self.label_phase.hide()
        self.input_phase.hide()
        row += 1

        # ── Inclination change option ──────────────────────────────────────────
        grid.addWidget(self._field_label("Inclination Change (°):"), row, 0)
        self.input_incl = QLineEdit("0")
        self.input_incl.setToolTip(
            "Add a plane-change maneuver on top of the orbit raise.\n"
            "Very expensive! A 28° plane change at LEO costs ~1,500 m/s.\n"
            "Set to 0 for a pure coplanar transfer."
        )
        grid.addWidget(self.input_incl, row, 1)
        row += 1

        left_layout.addLayout(grid)

        # Animation speed slider
        speed_section = QLabel("SIMULATION SPEED")
        speed_section.setObjectName("SectionLabel")
        left_layout.addWidget(speed_section)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_slider.setToolTip(
            "Controls how fast the animation plays.\n"
            "Low = slow (good for studying the trajectory).\n"
            "High = fast (good for quick iteration)."
        )
        left_layout.addWidget(self.speed_slider)

        left_layout.addStretch()

        # Action buttons
        btn_row = QHBoxLayout()
        self.calc_button = QPushButton("INITIATE CALCULATION")
        self.calc_button.setObjectName("CalcButton")
        self.reset_button = QPushButton("RESET")
        self.reset_button.setObjectName("ResetButton")
        btn_row.addWidget(self.calc_button, 4)
        btn_row.addWidget(self.reset_button, 1)
        left_layout.addLayout(btn_row)

        body_layout.addWidget(left_panel)

        # ── RIGHT COLUMN: Plot ─────────────────────────────────────────────────
        self.plotter = OrbitPlotter(self)
        self.plotter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        body_layout.addWidget(self.plotter, 1)

        root_layout.addLayout(body_layout, 1)

        # ── RESULTS PANEL (below the two columns) ─────────────────────────────
        results_panel = QFrame()
        results_panel.setObjectName("ResultPanel")
        results_row = QHBoxLayout(results_panel)
        results_row.setContentsMargins(16, 12, 16, 12)
        results_row.setSpacing(12)

        # We build 6 result "cards" in a row.
        # Each card shows a title label and a value label.
        self.card_dv       = self._result_card("DELTA-V",          "— m/s",     "CardValue")
        self.card_time     = self._result_card("TRANSFER TIME",     "—",         "CardValue")
        self.card_phys     = self._result_card("PROPELLANT (PHYS)", "— kg",      "CardValue")
        self.card_ai       = self._result_card("PROPELLANT (AI)",   "— kg",      "CardValueAlt")
        self.card_period1  = self._result_card("INITIAL PERIOD",    "—",         "CardValue")
        self.card_massfrac = self._result_card("MASS FRACTION",     "—",         "CardValue")

        for card in [self.card_dv, self.card_time, self.card_phys,
                     self.card_ai, self.card_period1, self.card_massfrac]:
            results_row.addWidget(card)

        root_layout.addWidget(results_panel)

        # ── STATUS BAR ────────────────────────────────────────────────────────
        self.status_label = QLabel("SYSTEM STANDBY.  CONFIGURE MISSION PARAMETERS AND PRESS INITIATE.")
        self.status_label.setStyleSheet(
            "color: #5a6a80; font-family: 'Courier New', 'DejaVu Sans Mono', monospace; font-size: 11px;"
        )
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root_layout.addWidget(self.status_label)

        # ── ANIMATION ENGINE ──────────────────────────────────────────────────
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animation_step)
        self.current_frame = 0

        # ── SIGNALS & SLOTS ───────────────────────────────────────────────────
        # Qt's signal/slot mechanism: when a widget changes, it emits a 'signal'
        # that triggers a connected 'slot' (a method). This keeps UI and logic
        # cleanly separated.
        self.calc_button.clicked.connect(self.calculate_mission)
        self.reset_button.clicked.connect(self.reset_dashboard)
        self.maneuver_box.currentTextChanged.connect(self.update_ui_for_maneuver)
        self.preset_box.currentTextChanged.connect(self.apply_preset)

        # ── LOAD AI SURROGATE MODEL ───────────────────────────────────────────
        # The model files are in ml/ relative to the project root.
        # We find the project root by going one directory up from this file.
        try:
            current_folder = os.path.dirname(__file__)
            project_root   = os.path.abspath(os.path.join(current_folder, '..'))
            model_path     = os.path.join(project_root, 'ml', 'surrogate_model.pkl')
            columns_path   = os.path.join(project_root, 'ml', 'model_columns.pkl')

            self.ai_model      = joblib.load(model_path)
            self.model_columns = joblib.load(columns_path)
            self.ai_status     = "ONLINE"
            self.ai_badge.setText("● AI MODEL: ONLINE")
            self.ai_badge.setObjectName("StatusOnline")
        except FileNotFoundError:
            # If the model hasn't been trained yet, the dashboard still works
            # on pure physics. Run ml/train_model.py to create the model.
            self.ai_model  = None
            self.ai_status = "OFFLINE"
            self.ai_badge.setText("● AI MODEL: OFFLINE")
            self.ai_badge.setObjectName("StatusOffline")

    # ── HELPER: Create a labelled input field label ───────────────────────────
    @staticmethod
    def _field_label(text):
        """Creates a consistently-styled input field label."""
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #8899aa; font-family: 'DM Mono', Courier; font-size: 11px;"
        )
        return lbl

    # ── HELPER: Create a result display card ─────────────────────────────────
    @staticmethod
    def _result_card(title_text, value_text, value_style):
        """
        Builds a small card widget with a category title and a data value.
        Returns the card QFrame so it can be added to the layout.
        The value label is stored on the card as card.value_label for updates.
        """
        card = QFrame()
        card.setObjectName("ResultCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(4)

        title = QLabel(title_text)
        title.setObjectName("CardTitle")
        card_layout.addWidget(title)

        value = QLabel(value_text)
        value.setObjectName(value_style)
        card_layout.addWidget(value)

        # Let's attach the specific label for direct text updating
        setattr(card, 'value_label', value)
        return card

    # ── SLOT: Apply target orbit preset ───────────────────────────────────────
    def apply_preset(self, preset_name):
        """Fills in the target orbit field when the user picks a preset."""
        altitude = TARGET_PRESETS.get(preset_name)
        if altitude is not None:
            self.input_r2.setText(str(altitude))

    # ── SLOT: Update visible fields based on selected maneuver ────────────────
    def update_ui_for_maneuver(self, text):
        """
        Shows and hides input fields depending on which maneuver is selected.
        Each maneuver needs different parameters from the user.
        """
        self.header.setText(text.upper())

        # Start by showing the standard fields
        self.label_r2.show()
        self.input_r2.show()
        self.preset_box.show()
        self.label_rb.hide()
        self.input_rb.hide()
        self.label_phase.hide()
        self.input_phase.hide()

        if text == "Bi-Elliptic Transfer":
            # Bi-Elliptic needs the deep-space apogee in addition to r2
            self.label_rb.show()
            self.input_rb.show()

        elif text == "Phasing Orbit":
            # Phasing stays in the same orbit — no target orbit needed
            self.label_r2.hide()
            self.input_r2.hide()
            self.preset_box.hide()
            self.label_phase.show()
            self.input_phase.show()

    # ── SLOT: Reset to default state ──────────────────────────────────────────
    def reset_dashboard(self):
        """Clears all results and resets inputs to Earth LEO → GEO defaults."""
        self.input_r1.setText("300")
        self.input_r2.setText("35786")
        self.input_mass.setText("2000")
        self.input_rb.setText("100000")
        self.input_phase.setText("45")
        self.input_incl.setText("0")
        self.body_box.setCurrentText("Earth")
        self.engine_box.setCurrentIndex(0)
        self.maneuver_box.setCurrentIndex(0)
        self.speed_slider.setValue(50)
        self.preset_box.setCurrentIndex(0)

        # Clear result cards
        for card, default in [
            (self.card_dv, "— m/s"),
            (self.card_time, "—"),
            (self.card_phys, "— kg"),
            (self.card_ai, "— kg"),
            (self.card_period1, "—"),
            (self.card_massfrac, "—"),
        ]:
            getattr(card, 'value_label').setText(default)

        self.status_label.setText(
            "SYSTEM RESET.  CONFIGURE MISSION PARAMETERS AND PRESS INITIATE."
        )
        self.plotter.draw_orbits(300, 35786, body="Earth")

    # ── MAIN CALCULATION SLOT ─────────────────────────────────────────────────
    def calculate_mission(self):
        """
        Runs when the user presses INITIATE CALCULATION.

        Sequence:
          1. Read and validate all input fields
          2. Run the physics engine for the selected maneuver
          3. Apply Tsiolkovsky rocket equation to get propellant mass
          4. Query the AI surrogate for a parallel prediction
          5. Update all result cards
          6. Trigger the orbital visualisation and animation
        """
        try:
            # ── 1. Read inputs ─────────────────────────────────────────────────
            alt1_km    = float(self.input_r1.text())
            final_mass = float(self.input_mass.text())
            incl_deg   = float(self.input_incl.text())

            # Look up selected body parameters (μ and radius)
            body_name  = self.body_box.currentText()
            body       = BODY_PARAMS[body_name]
            mu         = body["mu"]
            r_body_km  = body["radius_km"]
            min_alt    = body["min_alt_km"]
            max_alt    = body["max_alt_km"]

            # Look up the engine's Isp (specific impulse) from the preset dict
            engine_name = self.engine_box.currentText()
            isp         = ENGINE_PRESETS[engine_name]["isp"]

            # Convert altitudes to absolute radii (from body centre, in metres)
            r1 = (alt1_km * 1000) + (r_body_km * 1000)

            maneuver_type = self.maneuver_box.currentText()

            # ── 2. Input validation ────────────────────────────────────────────
            if alt1_km < min_alt or alt1_km > max_alt:
                self._set_error(
                    f"PARKING ORBIT OUT OF RANGE FOR {body_name.upper()} "
                    f"({min_alt}–{max_alt:,} km)."
                )
                return

            if maneuver_type != "Phasing Orbit":
                alt2_km = float(self.input_r2.text())
                r2 = (alt2_km * 1000) + (r_body_km * 1000)
                if alt2_km < min_alt or alt2_km > max_alt:
                    self._set_error(
                        f"TARGET ORBIT OUT OF RANGE FOR {body_name.upper()} "
                        f"({min_alt}–{max_alt:,} km)."
                    )
                    return
            else:
                alt2_km = alt1_km   # Phasing stays at r1
                r2 = r1

            # ── 3. Route to the correct physics function ───────────────────────
            rb_km = 0.0   # Default for maneuvers that don't use rb
            phase_angle = 0.0
            
            delta_v = 0.0
            transfer_time_s = 0.0

            if maneuver_type == "Hohmann Transfer":
                delta_v = hohmann_transfer(mu, r1, r2)
                transfer_time_s = hohmann_transfer_time(mu, r1, r2)

            elif maneuver_type == "Bi-Elliptic Transfer":
                rb_km = float(self.input_rb.text())
                rb = (rb_km * 1000) + (r_body_km * 1000)

                # rb must be outside both orbits for the geometry to work
                if rb <= r1 or rb <= r2:
                    self._set_error(
                        "DEEP-SPACE APOGEE (r_b) MUST BE LARGER THAN BOTH ORBITS."
                    )
                    return

                delta_v = bi_elliptic_transfer(mu, r1, r2, rb)
                transfer_time_s = bi_elliptic_transfer_time(mu, r1, r2, rb)

            elif maneuver_type == "Phasing Orbit":
                phase_angle = float(self.input_phase.text())
                if not (0 < phase_angle <= 180):
                    self._set_error("PHASE ANGLE MUST BE BETWEEN 1° AND 180°.")
                    return
                delta_v = phasing_maneuver(mu, r1, phase_angle)
                # Phasing time = one phasing orbit period (approximately)
                t_initial = orbital_period(mu, r1)
                transfer_time_s = t_initial - (phase_angle / 360.0) * t_initial

            # ── Optional inclination change Δv added on top ───────────────────
            # Plane changes are done most efficiently at the transfer point
            # where velocity is lowest (apogee), but here we add it as a simple
            # budget addition — standard practice in preliminary mission design.
            if incl_deg != 0:
                # Use the average of initial and final circular velocities as
                # a rough estimate for where the plane change happens
                v_avg = (circular_velocity(mu, r1) + circular_velocity(mu, r2)) / 2
                delta_v += plane_change_dv(v_avg, incl_deg)

            # ── 4. Propellant mass (Tsiolkovsky Rocket Equation) ──────────────
            wet_mass  = calculate_initial_mass(delta_v, isp, final_mass)
            propellant = wet_mass - final_mass

            # Mass fraction: what percentage of the vehicle's total mass is fuel?
            mf_pct = mass_fraction(propellant, wet_mass)

            # Orbital periods of the initial and final orbits (Kepler's 3rd Law)
            period1_s = orbital_period(mu, r1)
            period2_s = orbital_period(mu, r2)

            # ── 5. AI Surrogate Prediction ────────────────────────────────────
            # The model uses circular orbital velocities as features because
            # every ΔV formula ultimately depends on  v_c = sqrt(μ/r) — not
            # on μ and r independently.  For a Hohmann transfer:
            #   ΔV = f(v_c1, v_c2)   — no other body/orbit parameters needed
            # In log space this mapping is nearly linear, so the MLP fits it
            # with very low error across Earth, Moon, and Mars.
            import math as _math
            ai_text = "OFFLINE"
            if self.ai_model is not None:
                try:
                    r_body_m = r_body_km * 1000

                    # Total orbital radii from body centre in metres
                    r1_m_ai = alt1_km * 1000 + r_body_m
                    r2_m_ai = (alt2_km * 1000 + r_body_m
                                if maneuver_type != "Phasing Orbit" else 0.0)
                    rb_m_ai = (rb_km  * 1000 + r_body_m
                                if maneuver_type == "Bi-Elliptic Transfer" else 0.0)

                    # Circular orbital velocities — the exact features the model
                    # was trained on (matching generate_data.py):
                    #   vc1 — always valid (parking orbit)
                    #   vc2 — 0 for Phasing (no separate target orbit)
                    #   vcb — 0 for Hohmann/Phasing (no intermediate apogee)
                    vc1 = _math.sqrt(mu / r1_m_ai)
                    vc2 = _math.sqrt(mu / r2_m_ai) if r2_m_ai > 0 else 0.0
                    vcb = _math.sqrt(mu / rb_m_ai) if rb_m_ai > 0 else 0.0

                    log_vc1 = _math.log(vc1)
                    log_vc2 = _math.log(vc2) if vc2 > 0 else 0.0
                    log_vcb = _math.log(vcb) if vcb > 0 else 0.0

                    # Body one-hot: forces body-specific DV mapping.
                    # Preferred over log(μ) because Mars (μ between Earth and
                    # Moon) would otherwise be interpolated incorrectly.
                    input_data = {
                        'Body_Earth':                [1 if body_name == "Earth" else 0],
                        'Body_Mars':                 [1 if body_name == "Mars"  else 0],
                        'Body_Moon':                 [1 if body_name == "Moon"  else 0],
                        'log_vc1':                   [log_vc1],
                        'log_vc2':                   [log_vc2],
                        'log_vcb':                   [log_vcb],
                        'Phase_Angle':               [phase_angle],
                        'Maneuver_Type_Bi-Elliptic': [1 if maneuver_type == "Bi-Elliptic Transfer" else 0],
                        'Maneuver_Type_Hohmann':     [1 if maneuver_type == "Hohmann Transfer" else 0],
                        'Maneuver_Type_Phasing':     [1 if maneuver_type == "Phasing Orbit" else 0],
                    }
                    input_df = pd.DataFrame(input_data).reindex(
                        columns=self.model_columns, fill_value=0
                    )
                    ai_dv = max(0.0, self.ai_model.predict(input_df)[0])

                    # If physics found zero delta-V (identical orbits), trust
                    # it exactly — the model's near-zero output carries a small
                    # residual from the log1p/expm1 round-trip.
                    if delta_v < 0.01:
                        ai_propellant = 0.0
                    else:
                        ai_wet        = calculate_initial_mass(ai_dv, isp, final_mass)
                        ai_propellant = max(0.0, ai_wet - final_mass)
                    ai_text = f"{ai_propellant:,.1f} kg"
                except Exception:
                    ai_text = "PREDICT ERR"

            # ── 6. Update result cards ─────────────────────────────────────────
            getattr(self.card_dv, 'value_label').setText(f"{delta_v:,.1f} m/s")
            getattr(self.card_time, 'value_label').setText(self._format_time(transfer_time_s))
            getattr(self.card_phys, 'value_label').setText(f"{propellant:,.1f} kg")
            getattr(self.card_ai, 'value_label').setText(ai_text)
            getattr(self.card_period1, 'value_label').setText(self._format_time(period1_s))
            getattr(self.card_massfrac, 'value_label').setText(f"{mf_pct:.1f} %")

            # Build a concise status message for the bottom bar
            p2_str = self._format_time(period2_s)
            self.status_label.setText(
                f"MISSION SUCCESS  |  {body_name}  |  Engine: {engine_name}  |  "
                f"Target Orbit Period: {p2_str}  |  "
                f"Payload Fraction: {payload_fraction(final_mass, wet_mass):.1f}%  |  "
                f"Isp: {isp} s"
            )
            self.status_label.setStyleSheet(
                "color: #00d4ff; font-family: 'DM Mono', Courier; font-size: 11px;"
            )

            # ── 7. Draw orbits and start animation ─────────────────────────────
            if maneuver_type == "Bi-Elliptic Transfer":
                self.plotter.draw_orbits(
                    alt1_km, alt2_km, maneuver=maneuver_type,
                    rb_km=rb_km, body=body_name
                )
            elif maneuver_type == "Phasing Orbit":
                self.plotter.draw_orbits(
                    alt1_km, alt1_km, maneuver=maneuver_type, body=body_name
                )
            else:
                self.plotter.draw_orbits(
                    alt1_km, alt2_km, maneuver=maneuver_type, body=body_name
                )

            # Restart animation from the beginning
            self.current_frame = 0
            delay_ms = max(10, int(1000 / self.speed_slider.value()))
            self.animation_timer.start(delay_ms)

        except ValueError:
            # If the user typed a letter instead of a number, catch it cleanly
            self._set_error("INVALID INPUT — NUMERIC VALUES ONLY.")

    # ── SLOT: Single animation frame tick ─────────────────────────────────────
    def animation_step(self):
        """
        Fires every few milliseconds (controlled by speed slider) to move the
        spacecraft one step along the pre-computed flight path.
        """
        if len(self.plotter.flight_path_x) == 0:
            self.animation_timer.stop()
            return

        x = self.plotter.flight_path_x[self.current_frame]
        y = self.plotter.flight_path_y[self.current_frame]

        # Compute the instantaneous velocity direction (tangent to the path)
        # by looking at the next frame's position. This drives the thrust arrow.
        if self.current_frame < len(self.plotter.flight_path_x) - 1:
            dx = self.plotter.flight_path_x[self.current_frame + 1] - x
            dy = self.plotter.flight_path_y[self.current_frame + 1] - y
        else:
            dx, dy = 0, 0

        self.plotter.update_rocket_position(x, y, dx, dy, "Prograde")
        self.current_frame += 1

        if self.current_frame >= len(self.plotter.flight_path_x):
            self.animation_timer.stop()
            # Hide the thrust arrow on arrival — engines off
            self.plotter.update_rocket_position(x, y, 0, 0)

    # ── HELPER: Format seconds into a readable time string ────────────────────
    @staticmethod
    def _format_time(seconds):
        """
        Converts a duration in seconds to a human-readable string.
        Uses minutes for short transfers (< 2 hours), hours for medium,
        and days for long ones — matching how ESA/NASA present transfer windows.
        """
        if seconds < 120:
            return f"{seconds:.0f} s"
        elif seconds < 7200:
            return f"{seconds / 60:.1f} min"
        elif seconds < 172800:
            return f"{seconds / 3600:.2f} hr"
        else:
            return f"{seconds / 86400:.2f} days"

    # ── HELPER: Show a red error message ──────────────────────────────────────
    def _set_error(self, message):
        """Displays an error message in the status bar with warning styling."""
        self.status_label.setText(f"⚠  {message}")
        self.status_label.setStyleSheet(
            "color: #ff6b35; font-family: 'DM Mono', Courier; font-size: 11px;"
        )


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_app():
    """Initialise the Qt application and show the dashboard."""
    # Enable high-DPI scaling for Retina / 4K displays BEFORE creating QApplication
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    window = OrbitalDashboard()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()