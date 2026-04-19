import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QGroupBox, QGridLayout,
    QProgressBar, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon

# Import our custom mathematical engines and visualizers
from mechanics.hohmann import hohmann_transfer
from mechanics.bi_elliptic import bi_elliptic_transfer
from mechanics.phasing import phasing_orbit
from visualization.plot_orbit import OrbitPlotter


class OrbitalDashboard(QMainWindow):
    """
    The main user interface for the Orbital Transfer System HMI.
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Orbital Transfer System HMI v2.1.4")
        self.setGeometry(100, 100, 1100, 650)
        self.setMinimumSize(900, 600)

        # Set dark theme styling for the main window
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d141e; /* Deep space navy */
            }
            QLabel {
                color: #e0e6ed;
                font-family: 'Inter', -apple-system, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                background-color: #151f2e;
                border: 1px solid #2a3a50;
                border-radius: 6px;
                margin-top: 1.5ex;
                padding-top: 15px;
                color: #00d4ff; /* Cyan accent for group headers */
                font-weight: 600;
                font-size: 12px;
                letter-spacing: 1px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                left: 10px;
                top: 5px;
            }
            QLineEdit, QComboBox {
                background-color: #0a0f18;
                border: 1px solid #2a3a50;
                border-radius: 4px;
                padding: 6px;
                color: #ffffff;
                font-family: 'DM Mono', Courier, monospace;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #00d4ff; /* Focus ring */
            }
            QPushButton {
                background-color: #00d4ff;
                color: #080d14;
                border: none;
                border-radius: 4px;
                padding: 10px 15px;
                font-weight: bold;
                font-size: 13px;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                background-color: #33ddff;
            }
            QPushButton:pressed {
                background-color: #009ebf;
            }
            QPushButton#calcButton {
                background-color: #7b61ff; /* Purple for calculation */
                color: white;
            }
            QPushButton#calcButton:hover {
                background-color: #927bff;
            }
        """)

        # Main central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # ── Left Panel (Controls & Telemetry) ──────────────────────────────────
        left_panel = QWidget()
        left_panel.setFixedWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)

        # 1. Mission Configuration Box
        config_group = QGroupBox("MISSION PARAMETERS")
        config_layout = QGridLayout()
        config_layout.setSpacing(12)

        # Central Body Selection
        config_layout.addWidget(QLabel("Central Body:"), 0, 0)
        self.body_combo = QComboBox()
        self.body_combo.addItems(["Earth", "Moon", "Mars"])
        self.body_combo.currentTextChanged.connect(self._on_body_changed)
        config_layout.addWidget(self.body_combo, 0, 1)

        # Maneuver Type Selection
        config_layout.addWidget(QLabel("Maneuver Type:"), 1, 0)
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "Hohmann Transfer",
            "Bi-Elliptic Transfer",
            "Phasing Orbit"
        ])
        self.type_combo.currentTextChanged.connect(self._toggle_inputs)
        config_layout.addWidget(self.type_combo, 1, 1)

        # Parking Orbit Altitude
        config_layout.addWidget(QLabel("Parking Altitude (km):"), 2, 0)
        self.alt1_input = QLineEdit("300")
        config_layout.addWidget(self.alt1_input, 2, 1)

        # Target Orbit Altitude
        self.alt2_label = QLabel("Target Altitude (km):")
        config_layout.addWidget(self.alt2_label, 3, 0)
        self.alt2_input = QLineEdit("35786")
        config_layout.addWidget(self.alt2_input, 3, 1)

        # Bi-Elliptic Intermediate Altitude (Hidden by default)
        self.rb_label = QLabel("Intermediate Alt. (km):")
        config_layout.addWidget(self.rb_label, 4, 0)
        self.rb_input = QLineEdit("100000")
        config_layout.addWidget(self.rb_input, 4, 1)

        # Phasing Angle (Hidden by default)
        self.angle_label = QLabel("Catch-up Angle (deg):")
        config_layout.addWidget(self.angle_label, 5, 0)
        self.angle_input = QLineEdit("30")
        config_layout.addWidget(self.angle_input, 5, 1)

        config_group.setLayout(config_layout)
        left_layout.addWidget(config_group)

        # Calculate Button
        self.calc_btn = QPushButton("EXECUTE TRAJECTORY CALCULATION")
        self.calc_btn.setObjectName("calcButton")
        self.calc_btn.clicked.connect(self.calculate_maneuver)
        left_layout.addWidget(self.calc_btn)

        # 2. Results / Telemetry Box
        results_group = QGroupBox("COMPUTED TELEMETRY")
        results_layout = QGridLayout()
        results_layout.setSpacing(10)

        # Styles for telemetry labels (monospace numbers, right aligned)
        val_style = "color: #00d4ff; font-family: 'DM Mono', Courier; font-weight: bold;"
        
        self.res_dv1 = QLabel("--")
        self.res_dv1.setStyleSheet(val_style)
        self.res_dv1.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.res_dv2 = QLabel("--")
        self.res_dv2.setStyleSheet(val_style)
        self.res_dv2.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.res_dv_total = QLabel("--")
        self.res_dv_total.setStyleSheet("color: #ff6b35; font-family: 'DM Mono'; font-weight: bold; font-size: 14px;")
        self.res_dv_total.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        self.res_tof = QLabel("--")
        self.res_tof.setStyleSheet(val_style)
        self.res_tof.setAlignment(Qt.AlignmentFlag.AlignRight)

        results_layout.addWidget(QLabel("ΔV₁ (Burn 1):"), 0, 0)
        results_layout.addWidget(self.res_dv1, 0, 1)
        
        results_layout.addWidget(QLabel("ΔV₂ (Burn 2):"), 1, 0)
        results_layout.addWidget(self.res_dv2, 1, 1)
        
        results_layout.addWidget(QLabel("Total ΔV Required:"), 2, 0)
        results_layout.addWidget(self.res_dv_total, 2, 1)
        
        results_layout.addWidget(QLabel("Time of Flight (TOF):"), 3, 0)
        results_layout.addWidget(self.res_tof, 3, 1)

        results_group.setLayout(results_layout)
        left_layout.addWidget(results_group)

        # Status Bar
        self.status_label = QLabel("SYSTEM IDLE. AWAITING COMMAND.")
        self.status_label.setStyleSheet("color: #5a6a80; font-family: 'DM Mono', Courier; font-size: 11px;")
        left_layout.addWidget(self.status_label)

        # Push everything to the top
        left_layout.addStretch()

        # ── Right Panel (Visualizer) ───────────────────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Plotter Canvas
        self.plotter = OrbitPlotter(self)
        
        # Add canvas to a container with a border
        plot_container = QWidget()
        plot_container.setStyleSheet("""
            QWidget {
                background-color: #080d14;
                border: 1px solid #2a3a50;
                border-radius: 6px;
            }
        """)
        pc_layout = QVBoxLayout(plot_container)
        pc_layout.setContentsMargins(2, 2, 2, 2)
        pc_layout.addWidget(self.plotter)
        
        right_layout.addWidget(plot_container)

        # Animation Controls
        anim_layout = QHBoxLayout()
        
        self.anim_btn = QPushButton("PLAY SIMULATION")
        self.anim_btn.setFixedWidth(180)
        self.anim_btn.clicked.connect(self.toggle_animation)
        anim_layout.addWidget(self.anim_btn)
        
        self.anim_progress = QProgressBar()
        self.anim_progress.setTextVisible(False)
        self.anim_progress.setFixedHeight(8)
        self.anim_progress.setStyleSheet("""
            QProgressBar {
                background-color: #151f2e;
                border: 1px solid #2a3a50;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #00d4ff;
                border-radius: 3px;
            }
        """)
        anim_layout.addWidget(self.anim_progress)
        
        right_layout.addLayout(anim_layout)

        # Add left and right panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, stretch=1) # Right panel takes remaining space

        # ── Animation Timer Setup ──────────────────────────────────────────────
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.animation_step)
        self.anim_frame = 0
        self.is_animating = False

        # Set initial UI state
        self._toggle_inputs()
        
    def _on_body_changed(self, body_name):
        """Update defaults when central body changes to prevent absurd values."""
        if body_name == "Earth":
            self.alt1_input.setText("300")
            self.alt2_input.setText("35786")
        elif body_name == "Moon":
            self.alt1_input.setText("50")
            self.alt2_input.setText("2000")
        elif body_name == "Mars":
            self.alt1_input.setText("200")
            self.alt2_input.setText("17000")
            
        self.status_label.setText(f"CENTRAL BODY RE-TARGETED: {body_name.upper()}")

    def _toggle_inputs(self):
        """Show or hide input fields based on the selected maneuver type."""
        m_type = self.type_combo.currentText()
        
        # Reset visibility
        self.alt2_label.setVisible(True)
        self.alt2_input.setVisible(True)
        self.rb_label.setVisible(False)
        self.rb_input.setVisible(False)
        self.angle_label.setVisible(False)
        self.angle_input.setVisible(False)
        
        if m_type == "Bi-Elliptic Transfer":
            self.rb_label.setVisible(True)
            self.rb_input.setVisible(True)
            
        elif m_type == "Phasing Orbit":
            self.alt2_label.setVisible(False)
            self.alt2_input.setVisible(False)
            self.angle_label.setVisible(True)
            self.angle_input.setVisible(True)

    def calculate_maneuver(self):
        """
        Gathers inputs, calls the appropriate physics engine, updates telemetry,
        and redraws the orbital diagram.
        """
        # Stop animation if running
        if self.is_animating:
            self.toggle_animation()

        try:
            # Parse common inputs
            body = self.body_combo.currentText()
            alt1 = float(self.alt1_input.text())
            m_type = self.type_combo.currentText()

            # Input Validation
            if alt1 <= 0:
                self._set_error("PARKING ALTITUDE MUST BE POSITIVE")
                return

            if m_type == "Hohmann Transfer":
                alt2 = float(self.alt2_input.text())
                if alt2 <= alt1:
                    self._set_error("TARGET ALTITUDE MUST EXCEED PARKING ALTITUDE")
                    return
                    
                dv1, dv2, tof = hohmann_transfer(alt1, alt2, body)
                
                self.res_dv1.setText(f"{dv1:.3f} km/s")
                self.res_dv2.setText(f"{dv2:.3f} km/s")
                self.res_dv_total.setText(f"{(dv1 + dv2):.3f} km/s")
                self.res_tof.setText(self._format_time(tof))
                
                self.plotter.draw_orbits(alt1, alt2, m_type, body=body)

            elif m_type == "Bi-Elliptic Transfer":
                alt2 = float(self.alt2_input.text())
                rb = float(self.rb_input.text())
                
                if alt2 <= alt1:
                    self._set_error("TARGET ALTITUDE MUST EXCEED PARKING ALTITUDE")
                    return
                if rb <= alt2:
                    self._set_error("INTERMEDIATE ALTITUDE MUST EXCEED TARGET ALTITUDE")
                    return
                    
                dv1, dv2, dv3, tof = bi_elliptic_transfer(alt1, alt2, rb, body)
                
                self.res_dv1.setText(f"{dv1:.3f} km/s")
                # Bi-elliptic has 3 burns, combine 2 and 3 for display simplicity
                self.res_dv2.setText(f"{(dv2 + dv3):.3f} km/s (B2+B3)")
                self.res_dv_total.setText(f"{(dv1 + dv2 + dv3):.3f} km/s")
                self.res_tof.setText(self._format_time(tof))
                
                self.plotter.draw_orbits(alt1, alt2, m_type, rb_km=rb, body=body)

            elif m_type == "Phasing Orbit":
                catch_up_deg = float(self.angle_input.text())
                
                if not (0 < catch_up_deg < 360):
                    self._set_error("CATCH-UP ANGLE MUST BE BETWEEN 0 AND 360")
                    return
                    
                dv_entry, dv_exit, tof = phasing_orbit(alt1, catch_up_deg, body)
                
                self.res_dv1.setText(f"{dv_entry:.3f} km/s")
                self.res_dv2.setText(f"{dv_exit:.3f} km/s")
                self.res_dv_total.setText(f"{(dv_entry + dv_exit):.3f} km/s")
                self.res_tof.setText(self._format_time(tof))
                
                self.plotter.draw_orbits(alt1, alt1, m_type, body=body)

            # Success state
            self.status_label.setText("TRAJECTORY COMPUTED SUCCESSFULLY.")
            self.status_label.setStyleSheet("color: #00d4ff; font-family: 'DM Mono', Courier; font-size: 11px;")
            
            # Reset animation state ready for play
            self.anim_frame = 0
            self.anim_progress.setValue(0)
            self.anim_btn.setText("PLAY SIMULATION")

        except ValueError:
            self._set_error("INPUT ERROR: NON-NUMERIC CHARACTERS DETECTED")
        except Exception as e:
            self._set_error(f"SYSTEM FAULT: {str(e).upper()}")

    # ── ANIMATION HANDLING ─────────────────────────────────────────────────────
    def toggle_animation(self):
        """Starts or stops the rocket animation across the transfer trajectory."""
        if len(self.plotter.flight_path_x) == 0:
            self._set_error("NO TRAJECTORY CALCULATED. EXECUTE CALCULATION FIRST.")
            return

        if self.is_animating:
            self.anim_timer.stop()
            self.anim_btn.setText("RESUME SIMULATION")
            self.is_animating = False
        else:
            # If we reached the end, reset to the start
            if self.anim_frame >= len(self.plotter.flight_path_x) - 1:
                self.anim_frame = 0
            
            self.anim_timer.start(40)  # 40ms = 25 frames per second
            self.anim_btn.setText("PAUSE SIMULATION")
            self.is_animating = True

    def animation_step(self):
        """Called every 40ms by QTimer to move the rocket one frame forward."""
        path_x = self.plotter.flight_path_x
        path_y = self.plotter.flight_path_y
        
        if self.anim_frame >= len(path_x) - 1:
            # Animation finished
            self.anim_timer.stop()
            self.is_animating = False
            self.anim_btn.setText("REPLAY SIMULATION")
            self.anim_progress.setValue(100)
            
            # Hide velocity vector
            self.plotter.update_rocket_position(
                path_x[-1], path_y[-1], 0, 0
            )
            return

        # Current position
        x = path_x[self.anim_frame]
        y = path_y[self.anim_frame]
        
        # Calculate velocity vector direction (tangent to path)
        # We look ahead one frame to find the direction of travel
        next_x = path_x[self.anim_frame + 1]
        next_y = path_y[self.anim_frame + 1]
        dx = next_x - x
        dy = next_y - y

        # Determine if we are accelerating (prograde) or decelerating (retrograde)
        # Simplification for visuals: first frame is always prograde burn, 
        # last frame is always prograde burn (circularising) for Hohmann.
        v_type = "Prograde"
        
        # Tell the plotter to redraw the rocket and vector at the new position
        self.plotter.update_rocket_position(x, y, dx, dy, v_type)
        
        # Update progress bar
        progress = int((self.anim_frame / len(path_x)) * 100)
        self.anim_progress.setValue(progress)
        
        self.anim_frame += 1

    # ── HELPER: Format seconds into readable time ──────────────────────────────
    def _format_time(self, seconds):
        if seconds < 3600:
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
