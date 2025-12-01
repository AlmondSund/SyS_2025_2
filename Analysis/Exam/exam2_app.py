"""
Streamlit dashboard that mirrors the Analysis/Exam/Exam2.ipynb notebook.
It covers AM modulation/demodulation and the mass–spring–damper to RLC study.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp
import sympy.physics.control as ctrl
from scipy.fft import fft, fftfreq
from scipy.io.wavfile import write as wav_write

import pydub
import yt_dlp
from pydub import AudioSegment
from yt_dlp import YoutubeDL

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
DEFAULT_AUDIO = CACHE_DIR / "DraDpi4bM-Y.webm"

st.set_page_config(page_title="Signals and Systems - Exam 2", layout="wide")
plt.style.use("seaborn-v0_8")


# ---------- Shared utilities ----------
def time_frequency_plot(signal: np.ndarray, f_s: int, labels: Optional[dict[str, str]] = None) -> plt.Figure:
    """Return a time/frequency matplotlib figure for a real-valued signal."""
    if labels is not None:
        expected_keys = {"title", "t_leyend", "f_leyend"}
        if not isinstance(labels, dict) or not expected_keys.issuperset(labels.keys()):
            raise ValueError("labels must be a dict with keys 'title', 't_leyend', 'f_leyend'")
    title = labels.get("title") if labels else None
    t_leyend = labels.get("t_leyend") if labels else None
    f_leyend = labels.get("f_leyend") if labels else None

    N = len(signal)
    if N == 0:
        raise ValueError("Signal is empty.")
    T_s = 1.0 / f_s
    t = np.linspace(0.0, N * T_s, N, endpoint=False)
    spectrum = fft(signal)
    freqs = fftfreq(N, T_s)[: N // 2]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)
    if title:
        fig.suptitle(title)

    time_plot_kwargs = {"color": "tab:blue"}
    if t_leyend:
        time_plot_kwargs["label"] = t_leyend

    axes[0].plot(t, signal, **time_plot_kwargs)
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)
    axes[0].set_title("Time domain")
    if t_leyend:
        axes[0].legend()

    freq_plot_kwargs = {"color": "tab:red"}
    if f_leyend:
        freq_plot_kwargs["label"] = f_leyend

    axes[1].plot(freqs, 2.0 / N * np.abs(spectrum[: N // 2]), **freq_plot_kwargs)
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_xlim(0, freqs[-1])
    axes[1].grid(True)
    axes[1].set_title("Frequency domain")
    if f_leyend:
        axes[1].legend()

    if title:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    return fig


def to_wav_bytes(signal: np.ndarray, f_s: int) -> bytes:
    """Encode a float32 signal into WAV bytes for st.audio."""
    buffer = io.BytesIO()
    wav_write(buffer, f_s, signal.astype(np.float32))
    buffer.seek(0)
    return buffer.read()


# ---------- AM modulation / demodulation ----------
def clear_cache(cache_dir: Optional[Path] = None) -> None:
    """Remove previously downloaded audio while keeping .gitignore intact."""
    cache_path = cache_dir if cache_dir else CACHE_DIR
    if not cache_path.exists() or not cache_path.is_dir():
        return
    for entry in cache_path.iterdir():
        if entry.name == ".gitignore":
            continue
        if entry.is_dir():
            for child in entry.iterdir():
                if child.is_dir():
                    continue
                child.unlink()
            entry.rmdir()
        else:
            entry.unlink()


def audiosegment_to_np(segment: AudioSegment, start_s: float, duration_s: float, sample_rate_hz: int) -> Tuple[np.ndarray, int]:
    """Slice, resample, and convert an AudioSegment to a normalized numpy array."""
    start_ms = max(int(start_s * 1000), 0)
    end_ms = start_ms + int(duration_s * 1000)
    trimmed = segment[start_ms:end_ms].set_channels(1).set_frame_rate(sample_rate_hz)
    raw = np.asarray(trimmed.get_array_of_samples())
    samples = raw.astype(np.float32) / np.iinfo(raw.dtype).max
    return samples, sample_rate_hz


def download_youtube_audio(url: str, start_s: float, duration_s: float, sample_rate_hz: int, clean_cache: bool = False) -> Tuple[np.ndarray, int, Path]:
    """Download audio from YouTube and return the processed samples."""
    if YoutubeDL is None or AudioSegment is None:
        raise ImportError("yt_dlp and pydub are required for YouTube downloads. Install them with `pip install yt-dlp pydub`.")

    cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    if clean_cache:
        clear_cache(cache_dir)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(cache_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        audio_path = cache_dir / f"{info['id']}.{info['ext']}"
    segment = AudioSegment.from_file(audio_path)
    samples, sr = audiosegment_to_np(segment, start_s, duration_s, sample_rate_hz)
    return samples, sr, audio_path


@st.cache_data(show_spinner=False)
def cached_youtube_audio(url: str, start_s: float, duration_s: float, sample_rate_hz: int, clean_cache: bool) -> Tuple[np.ndarray, int, Path]:
    return download_youtube_audio(url, start_s, duration_s, sample_rate_hz, clean_cache)


@st.cache_data(show_spinner=False)
def cached_file_audio(path: Path, start_s: float, duration_s: float, sample_rate_hz: int) -> Tuple[np.ndarray, int]:
    if AudioSegment is None:
        raise ImportError("pydub is required for local audio decoding. Install it with `pip install pydub`.")
    segment = AudioSegment.from_file(path)
    return audiosegment_to_np(segment, start_s, duration_s, sample_rate_hz)


@st.cache_data(show_spinner=False)
def cached_upload_audio(file_bytes: bytes, start_s: float, duration_s: float, sample_rate_hz: int) -> Tuple[np.ndarray, int]:
    if AudioSegment is None:
        raise ImportError("pydub is required for uploaded audio decoding. Install it with `pip install pydub`.")
    segment = AudioSegment.from_file(io.BytesIO(file_bytes))
    return audiosegment_to_np(segment, start_s, duration_s, sample_rate_hz)


def am_modulate(msg_t: np.ndarray, f_s: int, carrier_hz: float = 10_000.0, modulation_index: float = 1.0) -> np.ndarray:
    """DSB-SC modulation of a baseband message."""
    if modulation_index < 0:
        raise ValueError("modulation_index must be non-negative.")
    msg = np.asarray(msg_t, dtype=float)
    peak = float(np.max(np.abs(msg))) if msg.size else 0.0
    if peak == 0:
        return np.zeros_like(msg)
    norm_msg = msg / peak
    t = np.arange(len(norm_msg)) / f_s
    carrier = np.cos(2 * np.pi * carrier_hz * t)
    return modulation_index * norm_msg * carrier


def ideal_lowpass_fft(signal: np.ndarray, f_s: int, cutoff_hz: float) -> np.ndarray:
    freqs = np.fft.fftfreq(len(signal), d=1.0 / f_s)
    spectrum = np.fft.fft(signal)
    mask = np.abs(freqs) <= cutoff_hz
    return np.fft.ifft(spectrum * mask).real


def am_demodulate(am_t: np.ndarray, f_s: int, carrier_hz: float = 10_000.0, cutoff_hz: Optional[float] = None) -> np.ndarray:
    cutoff = cutoff_hz if cutoff_hz is not None else 0.45 * (f_s / 2.0)
    t = np.arange(len(am_t)) / f_s
    mixed = 2 * am_t * np.cos(2 * np.pi * carrier_hz * t)
    return ideal_lowpass_fft(mixed, f_s, cutoff)


def render_am_panel() -> None:
    st.header("1) AM modulation and coherent demodulation")
    st.caption("Ideal FFT low-pass filter used for envelope recovery; θ₀ = 0.")

    if (BASE_DIR / "demodulation-system.png").exists():
        st.image(str(BASE_DIR / "demodulation-system.png"), width="stretch")

    if AudioSegment is None:
        st.warning("Audio features need `pip install streamlit pydub yt-dlp` (ffmpeg is also required). The second tab works without them.")
        return

    col_left, col_right = st.columns([2, 1])
    with col_left:
        source = st.radio(
            "Message source",
            ["Cached snippet", "YouTube URL", "Upload file"],
            index=0,
            help="Cached snippet uses the sample already downloaded in cache/DraDpi4bM-Y.webm",
        )
    with col_right:
        start_s = st.number_input("Start time [s]", value=15.0, min_value=0.0, max_value=120.0, step=0.5)
        duration_s = st.slider("Duration [s]", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
        sample_rate = st.selectbox("Sample rate [Hz]", [22050, 32000, 44100, 48000], index=2)

    url = ""
    uploaded = None
    if source == "YouTube URL":
        url = st.text_input("YouTube link", value="https://www.youtube.com/watch?v=DraDpi4bM-Y")
    elif source == "Upload file":
        uploaded = st.file_uploader("Upload audio (any ffmpeg-compatible type)", type=None)

    load_button = st.button("Load/refresh message segment", type="primary")

    if "msg_payload" not in st.session_state:
        st.session_state.msg_payload = None
    if load_button or (st.session_state.msg_payload is None and source == "Cached snippet"):
        try:
            if source == "Cached snippet":
                samples, sr = cached_file_audio(DEFAULT_AUDIO, start_s, duration_s, sample_rate)
                label = "Cached snippet"
            elif source == "YouTube URL":
                if not url:
                    raise ValueError("Provide a valid YouTube URL.")
                samples, sr, audio_path = cached_youtube_audio(url, start_s, duration_s, sample_rate, clean_cache=False)
                label = f"YouTube ({audio_path.name})"
            else:
                if uploaded is None:
                    raise ValueError("Upload an audio file to proceed.")
                samples, sr = cached_upload_audio(uploaded.getvalue(), start_s, duration_s, sample_rate)
                label = f"Uploaded: {uploaded.name}"
            st.session_state.msg_payload = {"samples": samples, "fs": sr, "label": label}
            st.success(f"Loaded message from {label}")
        except Exception as exc:  # pragma: no cover - user feedback path
            st.session_state.msg_payload = None
            st.error(f"Audio load failed: {exc}")

    payload = st.session_state.msg_payload
    if not payload:
        st.info("Load audio to explore the AM chain.")
        return

    msg_t = payload["samples"]
    f_s = payload["fs"]

    st.subheader("Message")
    st.audio(to_wav_bytes(msg_t, f_s), format="audio/wav")
    st.pyplot(
        time_frequency_plot(
            msg_t,
            f_s,
            labels={"title": "Original message", "t_leyend": "$m(t)$", "f_leyend": "$M(f)$"},
        )
    )

    st.subheader("Modulation and demodulation")
    col1, col2 = st.columns(2)
    with col1:
        carrier = st.slider("Carrier frequency [Hz]", min_value=1_000, max_value=20_000, value=10_000, step=500)
        modulation_index = st.slider("Modulation index", min_value=0.0, max_value=2.5, value=1.0, step=0.1)
    with col2:
        cutoff = st.slider("LPF cutoff [Hz]", min_value=500, max_value=int(f_s // 2), value=int(0.45 * f_s / 2), step=100)
        clean_cache = st.checkbox("Clear cache before next YouTube fetch", value=False)
        if clean_cache:
            clear_cache()

    t_axis = np.arange(len(msg_t)) / f_s
    carrier_t = np.cos(2 * np.pi * carrier * t_axis)
    am_t = am_modulate(msg_t, f_s, carrier_hz=carrier, modulation_index=modulation_index)
    mixed_t = 2 * am_t * np.cos(2 * np.pi * carrier * t_axis)
    demod_t = am_demodulate(am_t, f_s, carrier_hz=carrier, cutoff_hz=float(cutoff))

    col_mod, col_demod = st.columns(2)
    with col_mod:
        st.markdown("**Carrier**")
        st.pyplot(
            time_frequency_plot(
                carrier_t,
                f_s,
                labels={"title": "Carrier Signal", "t_leyend": "$c(t)$", "f_leyend": "$C(f)$"},
            )
        )

        st.markdown("**AM (DSB-SC) signal**")
        st.audio(to_wav_bytes(am_t / (np.max(np.abs(am_t)) + 1e-9), f_s), format="audio/wav")
        st.pyplot(
            time_frequency_plot(
                am_t,
                f_s,
                labels={"title": "AM Modulated Signal", "t_leyend": "$s(t)$", "f_leyend": "$S(f)$"},
            )
        )
    with col_demod:
        st.markdown("**Mixer output (before LPF)**")
        st.pyplot(
            time_frequency_plot(
                mixed_t,
                f_s,
                labels={"title": "Mixer Output", "t_leyend": "$2s(t)\\cos(2\\pi f_c t)$", "f_leyend": "$|Mixed(f)|$"},
            )
        )

        st.markdown("**Demodulated (coherent + ideal LPF)**")
        st.audio(to_wav_bytes(demod_t / (np.max(np.abs(demod_t)) + 1e-9), f_s), format="audio/wav")
        st.pyplot(
            time_frequency_plot(
                demod_t,
                f_s,
                labels={"title": "Demodulated Message Signal", "t_leyend": "$\\hat{m}(t)$", "f_leyend": "$\\hat{M}(f)$"},
            )
        )

    st.markdown(
        """
The ideal spectra for DSB-SC are centered at ±f<sub>c</sub> with shapes matching M(f).
Coherent detection shifts the bands back to baseband, and the FFT mask removes doubled images,
leaving a scaled copy of the message.
        """,
        unsafe_allow_html=True,
    )


# ---------- Mechanical and RLC study ----------
s = sp.symbols("s")


def mechanical_tf(m: float, c: float, k: float) -> ctrl.TransferFunction:
    return ctrl.TransferFunction(1, sp.expand(m * s**2 + c * s + k), s)


def electrical_equivalent(m: float, c: float, k: float) -> dict:
    return {"R": float(c), "L": float(m), "C": float(1.0 / k)}


def denominator_coeffs(tf: ctrl.TransferFunction) -> Tuple[float, float, float]:
    den_poly = sp.Poly(sp.expand(tf.den), tf.var)
    coeffs = [float(sp.N(c)) for c in den_poly.all_coeffs()]
    if len(coeffs) != 3:
        raise ValueError("The model is not second order.")
    return coeffs[0], coeffs[1], coeffs[2]


def natural_quantities(a0: float, a1: float, a2: float) -> Tuple[float, float, float]:
    wn = float(np.sqrt(a2 / a0))
    zeta = float(a1 / (2 * np.sqrt(a0 * a2)))
    if abs(zeta - 1.0) < 1e-10:
        wd = 0.0
    elif zeta < 1.0:
        wd = float(wn * math.sqrt(1.0 - zeta**2))
    else:
        wd = float("nan")
    return wn, zeta, wd


def closed_loop_unity(tf: ctrl.TransferFunction) -> ctrl.TransferFunction:
    return ctrl.TransferFunction(tf.num, sp.simplify(tf.den + tf.num), tf.var)


def choose_time_window(wn: float, zeta: float) -> float:
    base = max(8.0 / max(wn, 1e-6), 0.4)
    if zeta < 0.4:
        base *= 2.0
    elif zeta > 1.2:
        base *= 0.8
    return float(base)


def step_time_metrics(tf: ctrl.TransferFunction, horizon: float, target: Optional[float] = None) -> Tuple[float, float, float, float]:
    t, y = ctrl.step_response_numerical_data(tf, upper_limit=horizon, adaptive=False, n=600)
    t_arr = np.asarray(t, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    final_val = float(target if target is not None else y_arr[-1])
    rise_time = math.nan
    idx10 = np.where(y_arr >= 0.1 * final_val)[0]
    idx90 = np.where(y_arr >= 0.9 * final_val)[0]
    if idx10.size and idx90.size:
        rise_time = float(t_arr[idx90[0]] - t_arr[idx10[0]])
    peak_idx = int(np.argmax(y_arr))
    peak_time = float(t_arr[peak_idx])
    band = 0.02 * max(abs(final_val), 1e-6)
    settling_time = math.nan
    for i in range(len(y_arr)):
        if np.all(np.abs(y_arr[i:] - final_val) <= band):
            settling_time = float(t_arr[i])
            break
    return rise_time, peak_time, settling_time, final_val


def collect_metrics(label: str, tf: ctrl.TransferFunction, loop_label: str, m: float, c: float, k: float, equivalents: dict):
    a0, a1, a2 = denominator_coeffs(tf)
    wn, zeta, wd = natural_quantities(a0, a1, a2)
    time_window = choose_time_window(wn, zeta)
    dc_gain = float(sp.N(tf.num.subs(tf.var, 0) / tf.den.subs(tf.var, 0)))
    rise_time, peak_time, settling_time, final_val = step_time_metrics(tf, time_window, target=dc_gain)
    metrics = {
        "Scenario": label,
        "Loop": loop_label,
        "m": m,
        "c": c,
        "k": k,
        "den_a0": a0,
        "den_a1": a1,
        "den_a2": a2,
        "zeta": zeta,
        "wn_rad_s": wn,
        "wd_rad_s": wd,
        "peak_time_s": peak_time,
        "rise_time_s": rise_time,
        "settling_time_s": settling_time,
        "dc_gain": dc_gain,
        "R": equivalents["R"],
        "L": equivalents["L"],
        "C": equivalents["C"],
    }
    return metrics, time_window


def frequency_response(tf: ctrl.TransferFunction, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H_expr = sp.simplify(tf.num / tf.den)
    H_fun = sp.lambdify(s, H_expr, "numpy")
    Hw = H_fun(1j * w)
    mag_db = 20.0 * np.log10(np.abs(Hw))
    phase_deg = np.unwrap(np.angle(Hw)) * 180.0 / np.pi
    return mag_db, phase_deg


def plot_case(tf: ctrl.TransferFunction, label: str, horizon: float, freq_exp: Tuple[float, float] = (-2, 3)) -> plt.Figure:
    w = np.logspace(freq_exp[0], freq_exp[1], 400)
    mag_db, phase_deg = frequency_response(tf, w)
    t_imp, imp = ctrl.impulse_response_numerical_data(tf, upper_limit=horizon, adaptive=False, n=400)
    t_step, step = ctrl.step_response_numerical_data(tf, upper_limit=horizon, adaptive=False, n=400)
    t_ramp, ramp = ctrl.ramp_response_numerical_data(tf, upper_limit=horizon, adaptive=False, n=400)
    poles = [complex(p) for p in tf.poles()]
    zeros = [complex(z) for z in tf.zeros()]

    fig, axs = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle(label, fontsize=13)

    log_formatter = mticker.FuncFormatter(lambda val, _: f"{val:g}")

    axs[0, 0].semilogx(w, mag_db, color="C0")
    axs[0, 0].set_ylabel("Magnitude [dB]")
    axs[0, 0].set_title("Bode magnitude")
    axs[0, 0].grid(True, which="both")
    axs[0, 0].set_xlim(w[0], w[-1])
    axs[0, 0].xaxis.set_major_formatter(log_formatter)

    axs[1, 0].semilogx(w, phase_deg, color="C1")
    axs[1, 0].set_ylabel("Phase [deg]")
    axs[1, 0].set_xlabel("Frequency [rad/s]")
    axs[1, 0].set_title("Bode phase")
    axs[1, 0].grid(True, which="both")
    axs[1, 0].set_xlim(w[0], w[-1])
    axs[1, 0].xaxis.set_major_formatter(log_formatter)

    ax_pz = axs[2, 0]
    if zeros:
        ax_pz.scatter([z.real for z in zeros], [z.imag for z in zeros], marker="o", facecolors="none", edgecolors="tab:orange", label="Zeros")
    if poles:
        ax_pz.scatter([p.real for p in poles], [p.imag for p in poles], marker="x", color="tab:blue", label="Poles")
    max_range = max(
        [abs(v) for v in ([z.real for z in zeros] + [z.imag for z in zeros] + [p.real for p in poles] + [p.imag for p in poles])],
        default=1.0,
    )
    limit = max(1.0, max_range * 1.2)
    ax_pz.axhline(0, color="grey", linewidth=0.8)
    ax_pz.axvline(0, color="grey", linewidth=0.8)
    ax_pz.set_xlim(-limit, limit)
    ax_pz.set_ylim(-limit, limit)
    ax_pz.set_title("Pole-zero map")
    ax_pz.set_xlabel("Re{s}")
    ax_pz.set_ylabel("Im{s}")
    ax_pz.grid(True)
    if zeros or poles:
        ax_pz.legend(loc="best")

    axs[0, 1].plot(t_imp, imp, color="C2")
    axs[0, 1].set_title("Impulse response")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].grid(True)

    axs[1, 1].plot(t_step, step, color="C3")
    axs[1, 1].set_title("Step response")
    axs[1, 1].set_ylabel("Amplitude")
    axs[1, 1].grid(True)

    axs[2, 1].plot(t_ramp, ramp, color="C4")
    axs[2, 1].set_title("Ramp response")
    axs[2, 1].set_ylabel("Amplitude")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].grid(True)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def render_mechanical_panel() -> None:
    st.header("2) Mass–spring–damper ⇄ RLC study")

    if (BASE_DIR / "spring-mass-damper.png").exists():
        st.image(str(BASE_DIR / "spring-mass-damper.png"), width=420)
    if (BASE_DIR / "rlc-system.png").exists():
        st.image(str(BASE_DIR / "rlc-system.png"), width=420)

    st.markdown(
        """
Transfer function: $G(s)=\\dfrac{1}{ms^2+cs+k}$ with the force–voltage mapping $L=m$, $R=c$, $C=1/k$.
Closed loop uses unity feedback. Use the controls below to explore damping regimes.
        """,
        unsafe_allow_html=True,
    )

    presets = {
        "Underdamped (ζ=0.5)": {"m": 1.0, "c": 10.0, "k": 100.0},
        "Critically damped (ζ=1)": {"m": 1.0, "c": 20.0, "k": 100.0},
        "Overdamped (ζ=2)": {"m": 1.0, "c": 40.0, "k": 100.0},
        "Custom": None,
    }
    preset_label = st.selectbox("Preset", list(presets.keys()), index=0)
    preset_vals = presets[preset_label]

    col_m, col_c, col_k = st.columns(3)
    with col_m:
        m_val = st.number_input("Mass m [kg]", value=preset_vals["m"] if preset_vals else 1.0, min_value=0.1, max_value=10.0, step=0.1)
    with col_c:
        c_val = st.number_input("Damping c [N·s/m]", value=preset_vals["c"] if preset_vals else 10.0, min_value=0.1, max_value=80.0, step=1.0)
    with col_k:
        k_val = st.number_input("Spring k [N/m]", value=preset_vals["k"] if preset_vals else 100.0, min_value=1.0, max_value=200.0, step=1.0)

    loop_modes = st.multiselect("Systems to display", ["Open-loop", "Closed-loop"], default=["Open-loop", "Closed-loop"])

    base_tf = mechanical_tf(m_val, c_val, k_val)
    eq_vals = electrical_equivalent(m_val, c_val, k_val)

    rows = []
    plot_entries = []
    if "Open-loop" in loop_modes:
        metrics, window = collect_metrics(preset_label, base_tf, "open-loop", m_val, c_val, k_val, eq_vals)
        rows.append(metrics)
        plot_entries.append(("Open-loop", base_tf, window))
    if "Closed-loop" in loop_modes:
        closed_tf = closed_loop_unity(base_tf)
        metrics, window = collect_metrics(preset_label, closed_tf, "closed-loop", m_val, c_val, k_val, eq_vals)
        rows.append(metrics)
        plot_entries.append(("Closed-loop", closed_tf, window))

    df = pd.DataFrame(rows)
    if not df.empty:
        st.dataframe(
            df[
                [
                    "Scenario",
                    "Loop",
                    "zeta",
                    "wn_rad_s",
                    "wd_rad_s",
                    "peak_time_s",
                    "rise_time_s",
                    "settling_time_s",
                    "dc_gain",
                    "den_a0",
                    "den_a1",
                    "den_a2",
                    "R",
                    "L",
                    "C",
                ]
            ].round(4),
            width="stretch",
        )

    for loop_label, tf_obj, window in plot_entries:
        st.divider()
        st.subheader(f"{loop_label} responses")
        st.pyplot(plot_case(tf_obj, f"{preset_label} – {loop_label}", window))


def main() -> None:
    st.title("Exam #2 Signals and Systems - 2025 2S")
    st.caption("Martín Ramírez Espinosa – Department of Electrical, Electronic and Computer Engineering – UNAL Manizales")

    tab_am, tab_mech = st.tabs(["AM demodulator", "Mass-spring / RLC"])
    with tab_am:
        render_am_panel()
    with tab_mech:
        render_mechanical_panel()


if __name__ == "__main__":
    main()
