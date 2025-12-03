"""
Streamlit dashboard that mirrors the Analysis/Exam/Exam2.ipynb notebook.
It covers AM modulation/demodulation and the mass–spring–damper to RLC study.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
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
        axes[0].legend(loc="upper right")

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
        axes[1].legend(loc="upper right")

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
            ["Cached snippet", "Upload file"],
            index=0,
            help="Cached snippet uses the sample already downloaded in cache/DraDpi4bM-Y.webm",
        )
    with col_right:
        start_s = st.number_input("Start time [s]", value=15.0, min_value=0.0, max_value=120.0, step=0.5)
        duration_s = st.slider("Duration [s]", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
        sample_rate = st.selectbox("Sample rate [Hz]", [22050, 32000, 44100, 48000], index=2)

    uploaded = None
    if source == "Upload file":
        uploaded = st.file_uploader("Upload audio (any ffmpeg-compatible type)", type=None)

    load_button = st.button("Load/refresh message segment", type="primary")

    if "msg_payload" not in st.session_state:
        st.session_state.msg_payload = None
    if load_button or (st.session_state.msg_payload is None and source == "Cached snippet"):
        try:
            if source == "Cached snippet":
                samples, sr = cached_file_audio(DEFAULT_AUDIO, start_s, duration_s, sample_rate)
                label = "Cached snippet"
            elif source == "Upload file":
                if uploaded is None:
                    raise ValueError("Upload an audio file to proceed.")
                samples, sr = cached_upload_audio(uploaded.getvalue(), start_s, duration_s, sample_rate)
                label = f"Uploaded: {uploaded.name}"
            else:
                raise ValueError("Unsupported source selection.")
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
        st.audio(to_wav_bytes(carrier_t.astype(np.float32), f_s), format="audio/wav")
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
        st.audio(to_wav_bytes(mixed_t / (np.max(np.abs(mixed_t)) + 1e-9), f_s), format="audio/wav")
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
s = sp.symbols("s", complex=True)
w_n, zeta, K = sp.symbols("w_n zeta K", real=True, positive=True)


def get_electric_params(mechanical_params: dict[str, float]) -> dict[str, float]:
    """Convert mass-spring-damper params to their RLC analog."""
    required = {"m", "c", "k"}
    missing = required.difference(mechanical_params)
    if missing:
        raise KeyError(f"Missing parameters: {', '.join(sorted(missing))}")

    m = float(mechanical_params["m"])
    c = float(mechanical_params["c"])
    k = float(mechanical_params["k"])
    if k == 0:
        raise ValueError("k must be non-zero to compute capacitance")

    return {"R": c, "L": m, "C": 1.0 / k}


def get_mechanical_params(electric_params: dict[str, float]) -> dict[str, float]:
    """Convert RLC params to their mass-spring-damper analog."""
    required = {"R", "L", "C"}
    missing = required.difference(electric_params)
    if missing:
        raise KeyError(f"Missing parameters: {', '.join(sorted(missing))}")

    R = float(electric_params["R"])
    L = float(electric_params["L"])
    C = float(electric_params["C"])
    if C == 0:
        raise ValueError("C must be non-zero to compute stiffness")

    return {"m": L, "c": R, "k": 1.0 / C}


def get_standard_params(params: dict[str, float]) -> dict[str, float]:
    """Return standard second-order parameters (w_n, zeta, K)."""
    mechanical_keys = {"m", "c", "k"}
    electrical_keys = {"R", "L", "C"}

    if mechanical_keys.issubset(params):
        mechanical = {key: float(params[key]) for key in ("m", "c", "k")}
    elif electrical_keys.issubset(params):
        mechanical = get_mechanical_params(params)
    else:
        raise KeyError("params must contain either {'m', 'c', 'k'} or {'R', 'L', 'C'}")

    m = mechanical["m"]
    c = mechanical["c"]
    k = mechanical["k"]
    if m == 0 or k == 0:
        raise ValueError("m and k must be non-zero to compute standard parameters")

    wn_val = float(np.sqrt(k / m))
    zeta_val = float(c / (2 * np.sqrt(m * k)))
    K_val = float(1.0 / k)
    return {"w_n": wn_val, "zeta": zeta_val, "K": K_val}


def classify_damping(standard_params: dict[str, float]) -> str:
    """Return the damping type based on zeta."""
    required = {"w_n", "zeta", "K"}
    missing = required.difference(standard_params)
    if missing:
        raise KeyError(f"Missing parameters: {', '.join(sorted(missing))}")

    zeta_val = float(standard_params["zeta"])
    eps = 1e-6
    if abs(zeta_val - 1.0) <= eps:
        return "critically damped"
    if zeta_val < 1.0:
        return "underdamped"
    return "overdamped"


def symbolic_transfer_function(closed_loop: bool = False) -> Tuple[sp.Expr, list[sp.Expr]]:
    """Return the symbolic transfer function and its poles."""
    H_open = (K * w_n**2) / (s**2 + 2 * zeta * w_n * s + w_n**2)
    H = sp.simplify(H_open / (1 + H_open)) if closed_loop else sp.simplify(H_open)

    denom = sp.denom(sp.together(H))
    poles = sp.solve(sp.Eq(denom, 0), s)
    return H, poles


def get_time_response_metrics(poles: list[sp.Expr], responses: dict[str, bool], standard_params: dict[str, float]) -> pd.DataFrame:
    """Return key time-domain metrics for the specified response type."""
    required_params = {"w_n", "zeta", "K"}
    missing_params = required_params.difference(standard_params)
    if missing_params:
        raise KeyError(f"Missing parameters: {', '.join(sorted(missing_params))}")

    expected_responses = {"impulse", "step", "ramp"}
    if set(responses.keys()) != expected_responses:
        raise KeyError("responses must contain the keys 'impulse', 'step', and 'ramp'")

    active = [k for k, v in responses.items() if v]
    if len(active) != 1:
        raise ValueError("Exactly one response type must be set to True")

    if len(poles) != 2:
        raise ValueError("poles must contain exactly two elements [p1, p2]")

    w_n_val = float(standard_params["w_n"])
    zeta_val = float(standard_params["zeta"])
    K_val = float(standard_params["K"])
    w_d_val = w_n_val * np.sqrt(max(1 - zeta_val**2, 0.0))

    data = {"w_n": w_n_val, "zeta": zeta_val, "w_d": w_d_val}

    if active[0] == "step" and zeta_val < 1.0:
        import cmath

        subs = {w_n: w_n_val, zeta: zeta_val, K: K_val}
        p1 = complex(sp.N(poles[0].subs(subs)))
        p2 = complex(sp.N(poles[1].subs(subs)))
        diff = p1 - p2
        T_r = np.inf if diff == 0 else (1.0 / diff) * cmath.log(p1 / p2, cmath.e).real

        T_p = np.inf if w_d_val == 0 else np.pi / w_d_val
        T_s = np.inf if zeta_val == 0 else 4.0 / (zeta_val * w_n_val)

        data.update({"T_p": T_p, "T_r": T_r, "T_s": T_s})

    return pd.DataFrame([data])


def choose_time_window(wn: float, zeta_val: float) -> float:
    base = max(8.0 / max(wn, 1e-6), 0.4)
    if zeta_val < 0.4:
        base *= 2.0
    elif zeta_val > 1.2:
        base *= 0.8
    return float(base)


def instantiate_transfer_function(standard_params: dict[str, float], closed_loop: bool) -> Tuple[ctrl.TransferFunction, list[sp.Expr], dict]:
    """Create a control.TransferFunction from the symbolic expression."""
    H_sym, poles_sym = symbolic_transfer_function(closed_loop=closed_loop)
    subs = {w_n: standard_params["w_n"], zeta: standard_params["zeta"], K: standard_params["K"]}
    H_eval = sp.simplify(H_sym.subs(subs))
    num_expr, den_expr = sp.fraction(sp.together(H_eval))
    tf_obj = ctrl.TransferFunction(num_expr, den_expr, s)
    return tf_obj, poles_sym, subs


def bode_figure(system: ctrl.TransferFunction, freq_decades: Tuple[int, int]) -> plt.Figure:
    """Use the control API to create a Bode plot figure."""
    plt.figure(figsize=(8, 6))
    ctrl.bode_plot(system, initial_exp=freq_decades[0], final_exp=freq_decades[1], grid=True, show_axes=False, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def pole_zero_figure(system: ctrl.TransferFunction) -> plt.Figure:
    """Use the control API to create a pole-zero plot figure."""
    plt.figure(figsize=(5.5, 5.5))
    ctrl.pole_zero_plot(system, grid=True, show_axes=True, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def time_response_figure(system: ctrl.TransferFunction, response: str, horizon: float) -> plt.Figure:
    """Plot impulse/step/ramp responses using control API numerical data."""
    response_key = response.lower()
    if response_key == "impulse":
        t, y = ctrl.impulse_response_numerical_data(system, upper_limit=horizon, adaptive=False, n=600)
        title = "Impulse response"
    elif response_key == "step":
        t, y = ctrl.step_response_numerical_data(system, upper_limit=horizon, adaptive=False, n=600)
        title = "Step response"
    else:
        t, y = ctrl.ramp_response_numerical_data(system, upper_limit=horizon, adaptive=False, n=600)
        title = "Ramp response"

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(t, y, color="C3")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, ls="--", lw=0.6)
    fig.tight_layout()
    return fig


def render_mechanical_panel() -> None:
    st.header("2) Mass–spring–damper ⇄ RLC study")

    images = []
    if (BASE_DIR / "spring-mass-damper.png").exists():
        images.append(str(BASE_DIR / "spring-mass-damper.png"))
    if (BASE_DIR / "rlc-system.png").exists():
        images.append(str(BASE_DIR / "rlc-system.png"))
    if images:
        st.image(images, width=400)

    st.markdown(
        """
Transfer function: $G(s)=\\dfrac{1}{ms^2+cs+k}$ with the force–voltage mapping $L=m$, $R=c$, $C=1/k$.
Closed loop uses unity feedback. Choose whether to type mechanical or electrical values; conversions, damping, and plots update automatically.
        """,
        unsafe_allow_html=True,
    )

    input_mode = st.select_slider(
        "Parameter domain",
        options=["Mechanical (m, c, k)", "Electrical (R, L, C)"],
        value="Mechanical (m, c, k)",
        help="Slide to decide if you want to enter mechanical or electrical values.",
    )

    default_mech = {"m": 1.0, "c": 10.0, "k": 100.0}
    default_elec = get_electric_params(default_mech)

    if input_mode.startswith("Mechanical"):
        col_m, col_c, col_k = st.columns(3)
        with col_m:
            m_val = st.number_input("Mass m [kg]", value=default_mech["m"], min_value=0.1, max_value=20.0, step=0.1)
        with col_c:
            c_val = st.number_input("Damping c [N·s/m]", value=default_mech["c"], min_value=0.1, max_value=120.0, step=0.5)
        with col_k:
            k_val = st.number_input("Spring k [N/m]", value=default_mech["k"], min_value=1.0, max_value=400.0, step=1.0)
        mechanical_params = {"m": m_val, "c": c_val, "k": k_val}
        electrical_params = get_electric_params(mechanical_params)
    else:
        col_R, col_L, col_C = st.columns(3)
        with col_R:
            R_val = st.number_input("Resistance R [Ω]", value=default_elec["R"], min_value=0.1, max_value=200.0, step=0.5)
        with col_L:
            L_val = st.number_input("Inductance L [H]", value=default_elec["L"], min_value=0.05, max_value=20.0, step=0.05)
        with col_C:
            C_val = st.number_input("Capacitance C [F]", value=default_elec["C"], min_value=0.001, max_value=2.0, step=0.001, format="%.4f")
        electrical_params = {"R": R_val, "L": L_val, "C": C_val}
        mechanical_params = get_mechanical_params(electrical_params)

    standard_params = get_standard_params(mechanical_params if input_mode.startswith("Mechanical") else electrical_params)
    damping_label = classify_damping(standard_params)
    w_d_val = standard_params["w_n"] * np.sqrt(max(1 - standard_params["zeta"] ** 2, 0.0))

    col_mech, col_elec = st.columns(2)
    with col_mech:
        st.markdown("**Mechanical parameters**")
        st.table(pd.DataFrame(mechanical_params, index=["Value"]).T)
    with col_elec:
        st.markdown("**Electrical equivalents**")
        st.table(pd.DataFrame(electrical_params, index=["Value"]).T)

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("wn [rad/s]", f"{standard_params['w_n']:.3f}")
    col_b.metric("zeta", f"{standard_params['zeta']:.3f}")
    col_c.metric("w_d [rad/s]", f"{w_d_val:.3f}")
    col_d.metric("Damping", damping_label)

    loop_modes = st.multiselect("Systems to display", ["Open-loop", "Closed-loop"], default=["Open-loop", "Closed-loop"])
    response_choice = st.radio("Time response to plot", ["Impulse", "Step", "Ramp"], index=1, horizontal=True)
    freq_decades = st.slider("Bode frequency range (powers of 10 rad/s)", -4, 5, value=(-2, 3))

    if not loop_modes:
        st.info("Select at least one system to visualize.")
        return

    response_flags = {
        "impulse": response_choice == "Impulse",
        "step": response_choice == "Step",
        "ramp": response_choice == "Ramp",
    }

    metrics_frames = []
    for loop_label in loop_modes:
        closed_loop = loop_label == "Closed-loop"
        tf_obj, poles_sym, _ = instantiate_transfer_function(standard_params, closed_loop=closed_loop)
        horizon = choose_time_window(standard_params["w_n"], standard_params["zeta"])

        st.divider()
        st.subheader(f"{loop_label} analysis")
        st.latex(sp.latex(tf_obj.to_expr()))

        metrics_df = get_time_response_metrics(poles_sym, response_flags, standard_params)
        metrics_df["loop"] = loop_label
        metrics_df["response"] = response_choice.lower()
        metrics_df["damping"] = damping_label
        metrics_df["K"] = standard_params["K"]
        metrics_frames.append(metrics_df)

        bode_fig = bode_figure(tf_obj, freq_decades)
        pz_fig = pole_zero_figure(tf_obj)
        resp_fig = time_response_figure(tf_obj, response_choice, horizon)

        col_bode, col_pz = st.columns([2, 1])
        with col_bode:
            st.pyplot(bode_fig)
        with col_pz:
            st.pyplot(pz_fig)
        st.pyplot(resp_fig)

    if metrics_frames:
        st.subheader("Key parameters")
        merged = pd.concat(metrics_frames, ignore_index=True)
        st.dataframe(merged.round(4), width="stretch")


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
