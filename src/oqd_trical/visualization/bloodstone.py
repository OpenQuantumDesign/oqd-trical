# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as scipy_cst
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import RadioButtons, TextBox
from scipy.optimize import minimize

# Import the exact solver from your notebook
try:
    import oqd_trical
except ImportError:
    oqd_trical = None


def main():
    # ==========================================
    # 1. Initial Parameters & State Caching
    # ==========================================
    init_N = "20"
    init_mode = "1"
    init_wx = "3.00"  # MHz (Radial X frequency)
    init_wy = "3.00"  # MHz (Radial Y frequency)
    init_wz = "0.200"  # MHz (Axial Z frequency)

    # Physical Constants for Lamb-Dicke Calculation
    MASS_ION = 171 * scipy_cst.m_u  # Ytterbium-171 mass
    LAMBDA_LASER = 355e-9  # 355 nm Raman beams
    DELTA_K = 4 * np.pi / LAMBDA_LASER

    trical_cache = {
        "N": None,
        "wx": None,
        "wy": None,
        "wz": None,
        "freqs_x": None,
        "freqs_y": None,
        "freqs_z": None,
        "evecs_x": None,
        "evecs_y": None,
        "evecs_z": None,
        "x_eq": None,
        "raw_z_eq": None,
        "trical_stable": True,
        "steane_stable": True,
        "w_crit": 0.0,
        "table_drawn_for": None,
    }

    # ==========================================
    # 2. Physics Engine (OQD_TRICAL + Bare-Metal Fallback)
    # ==========================================
    def solve_trical(N, wx_mhz, wy_mhz, wz_mhz):
        if (
            trical_cache["N"] == N
            and trical_cache["wx"] == wx_mhz
            and trical_cache["wy"] == wy_mhz
            and trical_cache["wz"] == wz_mhz
        ):
            return

        w_rad_min = min(wx_mhz, wy_mhz)
        beta_c = 0.73 * (N**0.86)
        omega_r_critical = beta_c * wz_mhz
        steane_stable = w_rad_min > omega_r_critical

        trical_cache["steane_stable"] = steane_stable
        trical_cache["w_crit"] = omega_r_critical

        try:
            if oqd_trical is None:
                raise Exception("oqd_trical not installed")

            omega_x = 2 * np.pi * wx_mhz * 1e6
            omega_y = 2 * np.pi * wy_mhz * 1e6
            omega_z = 2 * np.pi * wz_mhz * 1e6

            alpha = np.zeros((3, 3, 3))
            alpha[2, 0, 0] = MASS_ION * (omega_x) ** 2 / 2
            alpha[0, 2, 0] = MASS_ION * (omega_y) ** 2 / 2
            alpha[0, 0, 2] = MASS_ION * (omega_z) ** 2 / 2

            pp = oqd_trical.mechanical.PolynomialPotential(alpha, N=N)
            ti = oqd_trical.mechanical.TrappedIons(N, pp, m=MASS_ION)

            ti.equilibrium_position()
            ti.normal_modes()

            if not hasattr(ti, "mode_vectors") and not hasattr(ti, "eigenvectors"):
                raise Exception("Trical missing eigenvector attributes.")

        except Exception as e:
            trical_cache["trical_stable"] = False

            # --- CUSTOM BARE-METAL EXACT NUMERICAL SOLVER ---
            def coulomb_potential(u):
                axial_confinement = 0.5 * np.sum(u**2)
                coulomb_repulsion = 0.0
                for i in range(N):
                    for j in range(i + 1, N):
                        coulomb_repulsion += 1.0 / np.abs(u[i] - u[j])
                return axial_confinement + coulomb_repulsion

            u_guess = np.linspace(-N / 2, N / 2, N)
            res = minimize(coulomb_potential, u_guess, method="BFGS")
            u_eq = np.sort(res.x)

            # Build Hessian Matrices (CORRECTED SIGNS)
            H_z, H_x, H_y = np.zeros((N, N)), np.zeros((N, N)), np.zeros((N, N))
            alpha_x = (wx_mhz / wz_mhz) ** 2
            alpha_y = (wy_mhz / wz_mhz) ** 2

            for i in range(N):
                for j in range(N):
                    if i != j:
                        inv_dist_cubed = 1.0 / np.abs(u_eq[i] - u_eq[j]) ** 3
                        # Axial off-diagonals are negative (-2/d^3)
                        H_z[i, j] = -2.0 * inv_dist_cubed
                        # Radial off-diagonals are positive (+1/d^3)
                        H_x[i, j] = 1.0 * inv_dist_cubed
                        H_y[i, j] = 1.0 * inv_dist_cubed

            for i in range(N):
                # Diagonal elements derived from the COM eigenvalue conditions
                H_z[i, i] = 1.0 - np.sum(H_z[i, :]) + H_z[i, i]
                H_x[i, i] = alpha_x - np.sum(H_x[i, :]) + H_x[i, i]
                H_y[i, i] = alpha_y - np.sum(H_y[i, :]) + H_y[i, i]

            def solve_modes(H, omega_base):
                evals, evecs = np.linalg.eigh(H)
                freqs = np.sqrt(np.maximum(0, evals)) * omega_base
                return freqs, evecs

            f_z, e_z = solve_modes(H_z, wz_mhz)
            f_x, e_x = solve_modes(H_x, wz_mhz)
            f_y, e_y = solve_modes(H_y, wz_mhz)

            idx_z = np.argsort(f_z)
            trical_cache["freqs_z"] = f_z[idx_z]
            trical_cache["evecs_z"] = e_z[:, idx_z]

            idx_x = np.argsort(f_x)[::-1]
            trical_cache["freqs_x"] = f_x[idx_x]
            trical_cache["evecs_x"] = e_x[:, idx_x]

            idx_y = np.argsort(f_y)[::-1]
            trical_cache["freqs_y"] = f_y[idx_y]
            trical_cache["evecs_y"] = e_y[:, idx_y]

            omega_z_si = 2 * np.pi * wz_mhz * 1e6
            l_0 = (
                (scipy_cst.e**2 / (4 * np.pi * scipy_cst.epsilon_0))
                / (MASS_ION * omega_z_si**2)
            ) ** (1 / 3)

            trical_cache["raw_z_eq"] = u_eq * l_0
            trical_cache["x_eq"] = u_eq

        trical_cache["N"] = N
        trical_cache["wx"] = wx_mhz
        trical_cache["wy"] = wy_mhz
        trical_cache["wz"] = wz_mhz

    # ==========================================
    # 3. Figure and UI Setup (OPTIMIZED LAYOUT)
    # ==========================================
    fig, (ax_x, ax_y, ax_z) = plt.subplots(3, 1, figsize=(16, 9))

    # Gives maximum vertical height to the three spectrum plots on the left
    plt.subplots_adjust(top=0.94, bottom=0.22, left=0.06, right=0.52, hspace=0.45)

    # --- RIGHT SIDE PLOTS ---
    # Main modes table (Top Right)
    ax_table = plt.axes([0.56, 0.52, 0.42, 0.42])
    ax_table.axis("off")

    # Eigenvector Plot (Bottom Right-Middle)
    ax_evec = plt.axes([0.56, 0.05, 0.24, 0.40])

    # Lamb-Dicke Table (Bottom Right Edge - Maximized for Height)
    ax_eta = plt.axes([0.82, 0.05, 0.16, 0.40])
    ax_eta.axis("off")

    # --- Spectrum Plots Setup ---
    axes = [ax_x, ax_y, ax_z]
    labels = ["Radial X", "Radial Y", "Axial Z"]
    colors = ["tab:red", "tab:blue", "tab:green"]

    lines = {}
    highlights = {}

    for ax, label, color in zip(axes, labels, colors):
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        ax.set_ylabel(label, fontsize=11, fontweight="bold")
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(axis="x", which="minor", linestyle=":", alpha=0.4)

        lines[label] = ax.plot(
            [], [], "|", color=color, markersize=30, markeredgewidth=1.5
        )[0]
        highlights[label] = ax.plot(
            [], [], "o", color="gold", markeredgecolor="black", markersize=12, zorder=10
        )[0]

    ax_z.set_xlabel(r"Frequency $\omega/2\pi$ (MHz)", fontsize=12)

    # --- TEXT BOX UI ELEMENTS (GRID LAYOUT) ---
    # Column 1
    ax_N = plt.axes([0.08, 0.13, 0.05, 0.04])
    ax_mode = plt.axes([0.08, 0.06, 0.05, 0.04])

    # Column 2
    ax_wx = plt.axes([0.22, 0.13, 0.05, 0.04])
    ax_wy = plt.axes([0.22, 0.06, 0.05, 0.04])

    # Column 3
    ax_wz = plt.axes([0.35, 0.13, 0.05, 0.04])

    # Column 4 (Radio Buttons)
    ax_radio = plt.axes([0.43, 0.06, 0.08, 0.11], facecolor="lightgray")

    # Status Bar (Bottom spanning)
    ax_status = plt.axes([0.06, 0.01, 0.45, 0.03])
    ax_status.axis("off")
    status_text = ax_status.text(
        0.0,
        0.5,
        "STATUS: INITIALIZING...",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="gray",
    )

    # Text Boxes with Compact Labels
    text_N = TextBox(ax_N, "N: ", initial=init_N)
    text_mode = TextBox(ax_mode, "Mode: ", initial=init_mode)
    text_wx = TextBox(ax_wx, r"$\omega_x$ (MHz): ", initial=init_wx)
    text_wy = TextBox(ax_wy, r"$\omega_y$ (MHz): ", initial=init_wy)
    text_wz = TextBox(ax_wz, r"$\omega_z$ (MHz): ", initial=init_wz)
    radio_dir = RadioButtons(ax_radio, ("Radial X", "Radial Y", "Axial Z"))

    def get_val(text_box, default, dtype=float):
        try:
            return dtype(text_box.text)
        except ValueError:
            return default

    # ==========================================
    # 4. Unified Animation Engine
    # ==========================================
    def init():
        return list(lines.values()) + list(highlights.values()) + [status_text]

    def animate(frame):
        try:
            N = get_val(text_N, int(init_N), int)
            mode = get_val(text_mode, int(init_mode), int)
            wx = get_val(text_wx, float(init_wx))
            wy = get_val(text_wy, float(init_wy))
            wz = get_val(text_wz, float(init_wz))
            direction = radio_dir.value_selected

            if N < 2:
                N = 2
            if mode < 1:
                mode = 1
            if mode > N:
                mode = N

            solve_trical(N, wx, wy, wz)

            is_stable = trical_cache["steane_stable"]
            w_crit = trical_cache["w_crit"]
            raw_z = trical_cache["raw_z_eq"]

            freqs_z, freqs_x, freqs_y = (
                trical_cache["freqs_z"],
                trical_cache["freqs_x"],
                trical_cache["freqs_y"],
            )

            # --- UPDATE MAIN TABLE ---
            current_state = (N, wx, wy, wz, mode, is_stable)
            if trical_cache["table_drawn_for"] != current_state:
                ax_table.clear()
                ax_table.axis("off")
                ax_table.set_title(
                    "Normal Modes & Z-Positions", fontweight="bold", pad=10
                )

                if N <= 15:
                    col_labels = ["m", "Axial Z", "Radial X", "Radial Y", "Z Pos (μm)"]
                    cell_text = []
                    for i in range(N):
                        pos_str = (
                            f"{raw_z[i] * 1e6:.3f}"
                            if not trical_cache["trical_stable"]
                            else f"{raw_z[i] * 1e6:.3f}"
                        )
                        cell_text.append(
                            [
                                f"{i + 1}",
                                f"{freqs_z[i]:.4f}",
                                f"{freqs_x[i]:.4f}",
                                f"{freqs_y[i]:.4f}",
                                pos_str,
                            ]
                        )

                    table = ax_table.table(
                        cellText=cell_text,
                        colLabels=col_labels,
                        loc="center",
                        cellLoc="center",
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(9)
                    table.scale(1, 1.3)

                    active_row = mode
                    for j in range(5):
                        table[(active_row, j)].set_facecolor("gold")

                else:
                    col_labels = [
                        "m",
                        "Z",
                        "X",
                        "Y",
                        "Z(μm)",
                        "m",
                        "Z",
                        "X",
                        "Y",
                        "Z(μm)",
                    ]
                    cell_text = []
                    half = math.ceil(N / 2)
                    for i in range(half):
                        pos_str1 = f"{raw_z[i] * 1e6:.2f}"
                        row = [
                            f"{i + 1}",
                            f"{freqs_z[i]:.2f}",
                            f"{freqs_x[i]:.2f}",
                            f"{freqs_y[i]:.2f}",
                            pos_str1,
                        ]

                        idx2 = i + half
                        if idx2 < N:
                            pos_str2 = f"{raw_z[idx2] * 1e6:.2f}"
                            row.extend(
                                [
                                    f"{idx2 + 1}",
                                    f"{freqs_z[idx2]:.2f}",
                                    f"{freqs_x[idx2]:.2f}",
                                    f"{freqs_y[idx2]:.2f}",
                                    pos_str2,
                                ]
                            )
                        else:
                            row.extend(["", "", "", "", ""])
                        cell_text.append(row)

                    table = ax_table.table(
                        cellText=cell_text,
                        colLabels=col_labels,
                        loc="center",
                        cellLoc="center",
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(7)
                    table.scale(1, 1.1)

                    if mode <= half:
                        active_row = mode
                        for j in range(5):
                            table[(active_row, j)].set_facecolor("gold")
                    else:
                        active_row = mode - half
                        for j in range(5, 10):
                            table[(active_row, j)].set_facecolor("gold")

                trical_cache["table_drawn_for"] = current_state

            # --- UPDATE STATUS BAR ---
            if is_stable:
                status_text.set_text(
                    f"STATUS: STABLE (1D Linear Chain) ✓ | Critical Radial w_r > {w_crit:.3f} MHz"
                )
                status_text.set_color("darkgreen")
            else:
                status_text.set_text(
                    f"STATUS: UNSTABLE (Buckles into 2D Zigzag) ✗ | Critical Radial w_r > {w_crit:.3f} MHz"
                )
                status_text.set_color("red")

            # --- UPDATE STACKED SPECTRUM PLOTS ---
            lines["Radial X"].set_data(freqs_x, np.zeros_like(freqs_x))
            lines["Radial Y"].set_data(freqs_y, np.zeros_like(freqs_y))
            lines["Axial Z"].set_data(freqs_z, np.zeros_like(freqs_z))

            for hl in highlights.values():
                hl.set_data([], [])

            if direction == "Radial X":
                active_freq = freqs_x[mode - 1] if mode <= len(freqs_x) else 0.0
                highlights["Radial X"].set_data([active_freq], [0])
                active_ax = ax_x
                evecs = trical_cache["evecs_x"]
            elif direction == "Radial Y":
                active_freq = freqs_y[mode - 1] if mode <= len(freqs_y) else 0.0
                highlights["Radial Y"].set_data([active_freq], [0])
                active_ax = ax_y
                evecs = trical_cache["evecs_y"]
            else:
                active_freq = freqs_z[mode - 1] if mode <= len(freqs_z) else 0.0
                highlights["Axial Z"].set_data([active_freq], [0])
                active_ax = ax_z
                evecs = trical_cache["evecs_z"]

            for ax, freqs in zip(axes, [freqs_x, freqs_y, freqs_z]):
                if len(freqs) > 0 and np.max(freqs) > 0:
                    ax.set_xlim(np.min(freqs) * 0.9 - 0.05, np.max(freqs) * 1.1 + 0.05)
                else:
                    ax.set_xlim(0, 1)

            for ax in axes:
                ax.set_title("")

            if not is_stable:
                fig.suptitle(
                    f"2D Zigzag Threshold Crossed (Modes Invalid)",
                    color="red",
                    fontsize=14,
                    y=0.98,
                )
            else:
                title_str = f"Yb: Bare-Metal {direction} Spectrum (Active: {active_freq:.6f} MHz)"
                active_ax.set_title(title_str, color="black", fontsize=11)
                fig.suptitle("", y=0.98)

            # --- UPDATE EIGENVECTOR & LAMBD-DICKE PLOT ---
            ax_evec.clear()
            ax_eta.clear()
            ax_eta.axis("off")

            if evecs is not None and mode <= evecs.shape[1]:
                vec = evecs[:, mode - 1].copy()

                # Phase correction
                if vec[np.argmax(np.abs(vec))] < 0:
                    vec *= -1

                x_pos = np.arange(1, N + 1)
                ax_evec.stem(x_pos, vec, basefmt="C3-", linefmt="C0-", markerfmt="C0o")
                ax_evec.set_ylim(-1.0, 1.0)
                ax_evec.set_title(
                    f"Eigenvector $b_m^{{(j)}}$", fontweight="bold", fontsize=11
                )

                # Add numerical values beside each mode marker
                if N <= 15:
                    for i, val in enumerate(vec):
                        y_offset = 0.12 if val >= 0 else -0.15
                        ax_evec.text(
                            x_pos[i],
                            val + y_offset,
                            f"{val:.3f}",
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="black",
                            fontweight="bold",
                        )

                if N <= 10:
                    ax_evec.set_xticks(x_pos)
                    ax_evec.set_xticklabels([f"{i}" for i in x_pos])
                else:
                    ax_evec.set_xticks([1, N // 2, N])
                    ax_evec.set_xticklabels(["1", f"{N // 2}", f"{N}"])

                ax_evec.set_xlabel("Ion Index (j)")
                ax_evec.grid(True, linestyle=":", alpha=0.6)

                # --- Lamb-Dicke Calculation ---
                if active_freq > 1e-6:
                    omega_m = 2 * np.pi * active_freq * 1e6
                    zero_point_spread = np.sqrt(
                        scipy_cst.hbar / (2 * MASS_ION * omega_m)
                    )
                    eta_vec = vec * DELTA_K * zero_point_spread
                else:
                    eta_vec = np.zeros_like(vec)

                # --- Rendering Lamb-Dicke Table ---
                ax_eta.set_title(
                    r"Lamb-Dicke $\eta_m^{(j)}$", fontweight="bold", fontsize=11
                )
                col_labels = ["Ion", r"$b_m^{(j)}$", r"$\eta_m^{(j)}$"]
                cell_text = []

                display_N = min(N, 30)
                for j in range(display_N):
                    cell_text.append([f"{j + 1}", f"{vec[j]:.3f}", f"{eta_vec[j]:.4f}"])

                if N > 30:
                    cell_text.append(["...", "...", "..."])

                table_eta = ax_eta.table(
                    cellText=cell_text,
                    colLabels=col_labels,
                    loc="center",
                    cellLoc="center",
                )
                table_eta.auto_set_font_size(False)

                # Dynamically shrink table row height slightly as N grows to prevent overflow
                scale_factor = 1.0 if N <= 20 else 0.85
                table_eta.scale(1, scale_factor)
                table_eta.set_fontsize(7)

            return list(lines.values()) + list(highlights.values()) + [status_text]

        except Exception as e:
            print(f"Update Error: {e}")
            status_text.set_text(f"STATUS: ERROR ({e})")
            status_text.set_color("red")
            return list(lines.values()) + list(highlights.values()) + [status_text]

    # ==========================================
    # 5. Execute
    # ==========================================
    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=1000, interval=100, blit=False
    )
    plt.show()
