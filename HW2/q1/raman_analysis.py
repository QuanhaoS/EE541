#!/usr/bin/env python3
"""
Raman spectroscopy analysis
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

DATA_FILE = "raman.txt"  


def load_raman(path: str):
    """Load wavenumber and intensity."""
    x_list, y_list = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                xi = float(parts[0])
                yi = float(parts[1])
            except ValueError:
                continue
            x_list.append(xi)
            y_list.append(yi)
    return np.array(x_list), np.array(y_list)

def detect_peaks(x: np.ndarray, y: np.ndarray, prominence=None, distance=None):
    """Find peak indices in intensity."""
    dx = np.median(np.diff(x))
    if prominence is None:
        prominence = 0.05 * (np.max(y) - np.min(y))
    if distance is None:
        distance = max(1, int(5.0 / (dx if dx > 0 else 1)))
    peaks, props = find_peaks(y, prominence=prominence, distance=distance)
    return peaks, props


def good_widths(peak_x: np.ndarray, n_min: float = 2.0, n_max: float = 12.0, frac: float = 0.4):
    n_peaks = len(peak_x)
    widths = np.full(n_peaks, n_max)
    for i in range(n_peaks):
        dists = np.abs(peak_x - peak_x[i])
        dists[i] = np.inf
        nearest = np.min(dists)
        w = max(n_min, min(n_max, frac * nearest))
        widths[i] = w
    return widths


def refine_peak_wavenumber(x: np.ndarray, y: np.ndarray, x_peak: float, half_width: float):
    mask = (x >= x_peak - half_width) & (x <= x_peak + half_width)
    x_roi = x[mask]
    y_roi = y[mask]
    if len(x_roi) < 4:
        return x_peak
    cs = CubicSpline(x_roi, y_roi)
    x_fine = np.linspace(x_roi.min(), x_roi.max(), 200)
    d1 = cs(x_fine, 1)
    sign_changes = np.diff(np.sign(d1))
    idx = np.where(sign_changes < 0)[0]
    if len(idx) == 0:
        return x_peak
    cand = (x_fine[idx] + x_fine[idx + 1]) / 2
    best = np.argmin(np.abs(cand - x_peak))
    return float(cand[best])


def analyze_spectrum(x: np.ndarray, y: np.ndarray):
    peak_idx, props = detect_peaks(x, y)
    peak_x = x[peak_idx]
    peak_y = y[peak_idx]
    widths = good_widths(peak_x)
    refined_x = np.array([
        refine_peak_wavenumber(x, y, peak_x[i], widths[i])
        for i in range(len(peak_x))
    ])
    refined_y = np.interp(refined_x, x, y)
    return list(zip(refined_x, refined_y))

def main():
    x, y = load_raman(DATA_FILE)
    peaks = analyze_spectrum(x, y)
    # Sort by intensity
    peaks_sorted = sorted(peaks, key=lambda p: p[1], reverse=True)

    # (a) 
    print("Wavenumber estimates for the 8 largest spectral peaks (cm^-1):")
    for i, (wx, wy) in enumerate(peaks_sorted[:8], 1):
        print(f"  {i}. {wx:.4f}  (intensity {wy:.4f})")

    # (b) 
    fig_b, ax_b = plt.subplots(1, 1, figsize=(10, 4))
    ax_b.plot(x, y, "b-", linewidth=0.6, label="Raman spectrum")
    all_x = [p[0] for p in peaks_sorted]
    all_y = [p[1] for p in peaks_sorted]
    ax_b.scatter(all_x, all_y, color="red", s=25, zorder=5, label="Peak maxima")
    ax_b.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax_b.set_ylabel("Intensity")
    ax_b.set_title("Raman spectrum with detected peak maxima")
    ax_b.legend(loc="upper right")
    ax_b.grid(True, alpha=0.3)
    fig_b.tight_layout()
    fig_b.savefig("raman_full_spectrum.png", dpi=150)
    plt.close(fig_b)
    print("\nSaved: raman_full_spectrum.png")

    # (c) 
    top4 = peaks_sorted[:4]
    all_x_peaks = np.array([p[0] for p in peaks_sorted])
    widths = good_widths(all_x_peaks)
    widths_top4 = widths[:4]
    fig_c, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for idx, ((wx, wy), half_width) in enumerate(zip(top4, widths_top4)):
        ax = axes[idx]
        mask = (x >= wx - half_width) & (x <= wx + half_width)
        x_roi = x[mask]
        y_roi = y[mask]
        ax.plot(x_roi, y_roi, "b.-", markersize=4, label="Raw data")
        if len(x_roi) >= 4:
            cs = CubicSpline(x_roi, y_roi)
            x_fine = np.linspace(x_roi.min(), x_roi.max(), 300)
            ax.plot(x_fine, cs(x_fine), "g-", linewidth=1.5, label="Spline")
        ax.scatter([wx], [wy], color="red", s=80, zorder=5, marker="v", label="Max intensity")
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Peak {idx + 1}: {wx:.2f} cm$^{{-1}}$")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig_c.suptitle("Regions of interest: 4 largest peaks", fontsize=12)
    fig_c.tight_layout()
    fig_c.savefig("raman_zoomed_peaks.png", dpi=150)
    plt.close(fig_c)
    print("Saved: raman_zoomed_peaks.png")


if __name__ == "__main__":
    main()
