"""
CMP In-Situ Optical Endpoint Enhanced Demo
=============================================
Adds:
- Multi-wavelength signals (3 channels)
- Adaptive baseline
- Causal Kalman filter (level + slope)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# ----------------------------------------------------------------------
# 1. Multi-wavelength signal simulation
# ----------------------------------------------------------------------
def simulate_multiwavelength_signal(
    duration_s=60.0,
    sample_rate_hz=50.0,
    true_endpoint_s=42.0,
    seed=None,
):
    """
    Simulate 3 optical channels (e.g. 500 nm, 650 nm, 800 nm).
    Each has slightly different transition shape, noise and a small
    timing offset typical of real multi-wavelength endpoint heads.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1.0 / sample_rate_hz)
    n = len(t)

    channels = {}
    # Channel parameters: (amplitude, transition_width, time_offset, noise_std)
    specs = {
        "ch1_500nm": (0.48, 2.6, -0.3, 0.045),
        "ch2_650nm": (0.45, 2.9,  0.0, 0.038),
        "ch3_800nm": (0.42, 3.2,  0.4, 0.050),
    }

    for name, (amp, width, t_off, nstd) in specs.items():
        ep = true_endpoint_s + t_off
        ideal = 0.88 - amp / (1.0 + np.exp(-(t - ep) / (width / 4)))
        ideal += -0.0012 * np.clip(t - 5, 0, ep)
        ideal = np.clip(ideal, 0.22, 0.95)

        drift = 0.025 * np.sin(2 * np.pi * t / 33) + 0.012 * np.sin(2 * np.pi * t / 11)
        noise = rng.normal(0, nstd, n)
        spikes = np.zeros(n)
        sp = rng.random(n) < 0.007
        spikes[sp] = rng.choice([-1, 1], size=sp.sum()) * rng.uniform(0.12, 0.30, sp.sum())

        channels[name] = ideal + drift + noise + spikes

    df = pd.DataFrame({"time_s": t, **channels})
    df["true_endpoint_s"] = true_endpoint_s
    return df


def generate_lot(n_wafers=8, base_endpoint=42.0, seed=42):
    rng = np.random.default_rng(seed)
    wafers = []
    for i in range(n_wafers):
        ep = base_endpoint + rng.normal(0, 1.7)
        if i == 5:
            ep += 4.3
        df = simulate_multiwavelength_signal(true_endpoint_s=ep, seed=seed + i)
        df["wafer_id"] = i + 1
        wafers.append(df)
    return wafers


# ----------------------------------------------------------------------
# 2. Adaptive baseline (causal)
# ----------------------------------------------------------------------
def adaptive_baseline(x, alpha=0.015):
    """
    Simple exponential adaptive baseline.
    Tracks the slowly varying pre-endpoint level.
    alpha small → slow adaptation (good for baseline).
    """
    baseline = np.zeros_like(x)
    baseline[0] = x[0]
    for i in range(1, len(x)):
        # Only adapt when the signal is relatively flat / high
        # (very simple gate – real systems use more sophisticated logic)
        if x[i] > baseline[i-1] - 0.08:
            baseline[i] = (1 - alpha) * baseline[i-1] + alpha * x[i]
        else:
            baseline[i] = baseline[i-1]
    return baseline


# ----------------------------------------------------------------------
# 3. Causal Kalman filter (level + slope)
# ----------------------------------------------------------------------
def kalman_level_slope(z, dt=0.02, q_level=1e-5, q_slope=3e-6, r=0.0025):
    """
    Simple 2-state Kalman filter:
      state = [level, slope]
    Fully causal.
    """
    n = len(z)
    # State transition
    F = np.array([[1.0, dt],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])          # we observe level only

    Q = np.array([[q_level, 0],
                  [0, q_slope]])
    R = np.array([[r]])

    x = np.array([z[0], 0.0])           # initial state
    P = np.eye(2) * 0.1

    level_est = np.zeros(n)
    slope_est = np.zeros(n)

    for i in range(n):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        # Update
        y = z[i] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + (K @ y).flatten()
        P = (np.eye(2) - K @ H) @ P

        level_est[i] = x[0]
        slope_est[i] = x[1]

    return level_est, slope_est


# ----------------------------------------------------------------------
# 4. Full processing pipeline
# ----------------------------------------------------------------------
def process_signal(df, sample_rate=50.0):
    t = df["time_s"].values
    dt = t[1] - t[0]

    # Use the most stable channel as primary (ch2) and keep the others for fusion
    primary = df["ch2_650nm"].values
    ch1 = df["ch1_500nm"].values
    ch3 = df["ch3_800nm"].values

    # Light spike cleaning
    primary = signal.medfilt(primary, kernel_size=5)
    ch1 = signal.medfilt(ch1, kernel_size=5)
    ch3 = signal.medfilt(ch3, kernel_size=5)

    # Adaptive baseline on primary channel
    baseline = adaptive_baseline(primary, alpha=0.012)

    # Kalman on primary channel
    level_k, slope_k = kalman_level_slope(primary, dt=dt)

    # Simple multi-channel fusion: average of the three Kalman levels
    # (in a real system this would be a proper weighted / PCA / ML fusion)
    _, slope1 = kalman_level_slope(ch1, dt=dt)
    _, slope3 = kalman_level_slope(ch3, dt=dt)
    slope_fused = (slope_k + 0.7 * slope1 + 0.7 * slope3) / 2.4

    out = df.copy()
    out["primary"] = primary
    out["baseline"] = baseline
    out["level_kalman"] = level_k
    out["slope_kalman"] = slope_k
    out["slope_fused"] = slope_fused
    return out


# ----------------------------------------------------------------------
# 5. Detection using Kalman slope + adaptive baseline
# ----------------------------------------------------------------------
def detect_endpoint(
    df,
    slope_threshold=-0.025,
    drop_fraction=0.35,          # fraction of expected drop from baseline
    min_time_s=15.0,
    confirm_s=0.5,
    sample_rate=50.0,
):
    t = df["time_s"].values
    slope = df["slope_fused"].values
    level = df["level_kalman"].values
    baseline = df["baseline"].values

    confirm = int(confirm_s * sample_rate)
    start = int(min_time_s * sample_rate)

    for i in range(start, len(slope) - confirm):
        if not np.all(slope[i:i+confirm] < slope_threshold):
            continue
        # Require a significant drop relative to the adaptive baseline
        drop = baseline[i] - level[i]
        if drop > drop_fraction * 0.45:          # 0.45 ≈ typical full drop
            return {
                "detected": True,
                "endpoint_time_s": float(t[i]),
                "true_endpoint_s": float(df["true_endpoint_s"].iloc[0]),
                "error_s": float(t[i] - df["true_endpoint_s"].iloc[0]),
            }

    return {
        "detected": False,
        "endpoint_time_s": np.nan,
        "true_endpoint_s": float(df["true_endpoint_s"].iloc[0]),
        "error_s": np.nan,
    }


# ----------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CMP Endpoint – Multi-λ + Adaptive Baseline + Kalman")
    print("=" * 70)

    wafers = generate_lot(n_wafers=8, seed=42)
    results = []
    processed = []

    for df in wafers:
        proc = process_signal(df)
        det = detect_endpoint(proc)
        results.append(det)
        processed.append(proc)

        status = "DETECTED" if det["detected"] else "MISSED"
        print(f"Wafer {df['wafer_id'].iloc[0]:2d} | "
              f"True: {det['true_endpoint_s']:5.1f}s | "
              f"Det: {det['endpoint_time_s']:5.1f}s | "
              f"Err: {det['error_s']:+5.2f}s | {status}")

    # -------------------- Plots (first 3 interesting wafers) --------------------
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    fig.suptitle("CMP Multi-Wavelength Endpoint\n"
                 "(Adaptive Baseline + Causal Kalman)", fontsize=13, fontweight="bold")

    for ax_idx, w_idx in enumerate([0, 2, 5]):
        df = processed[w_idx]
        det = results[w_idx]

        # Left: signals + baseline + Kalman level
        ax = axes[ax_idx, 0]
        ax.plot(df["time_s"], df["ch2_650nm"], color="lightgray", lw=0.7, label="Raw ch2")
        ax.plot(df["time_s"], df["level_kalman"], color="C0", lw=1.5, label="Kalman level")
        ax.plot(df["time_s"], df["baseline"], color="C2", lw=1.3, ls="--", label="Adaptive baseline")
        ax.axvline(det["true_endpoint_s"], color="green", ls="--", lw=1.4, label="True EP")
        if det["detected"]:
            ax.axvline(det["endpoint_time_s"], color="red", ls="-.", lw=1.4, label="Detected EP")
        ax.set_ylabel("Reflectance")
        ax.set_title(f"Wafer {w_idx+1} – Signal / Baseline / Kalman")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        # Right: fused Kalman slope
        ax2 = axes[ax_idx, 1]
        ax2.plot(df["time_s"], df["slope_fused"], color="C1", lw=1.4, label="Fused Kalman slope")
        ax2.axhline(-0.025, color="gray", ls=":", label="Threshold")
        ax2.axvline(det["true_endpoint_s"], color="green", ls="--", lw=1.4)
        if det["detected"]:
            ax2.axvline(det["endpoint_time_s"], color="red", ls="-.", lw=1.4)
        ax2.set_ylabel("Slope")
        ax2.set_title(f"Wafer {w_idx+1} – Fused Slope Feature")
        ax2.legend(loc="lower right", fontsize=7)
        ax2.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("endpoint_multi_kalman.png", dpi=150)
    print("\nSaved: endpoint_multi_kalman.png")
    print("Done.")


if __name__ == "__main__":
    main()