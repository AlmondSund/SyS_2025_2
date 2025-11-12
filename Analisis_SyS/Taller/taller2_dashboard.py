"""
Dashboard en Streamlit para presentar el contenido de Taller2_SyS.ipynb
manteniendo la estructura de títulos, subtítulos y resaltados originales.
"""
# streamlit run Analisis_SyS/Taller/taller2_dashboard.py

from __future__ import annotations

import io
import json
from pathlib import Path
from time import perf_counter
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE_DIR / "Taller2_SyS.ipynb"
AUDIO_PATH = BASE_DIR / "audio.wav"
DFT_LENGTHS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

SECTION_DEFS = [
    {
        "key": "presentacion",
        "label": "Presentación",
        "start": "### **TALLER #2 SEÑALES Y SISTEMAS - 2025 2S**",
        "end": "### **1. Transformada de Fourier**",
    },
    {
        "key": "sec1_1",
        "label": "1.1. Consultar y realizar los ejercicios propuestos",
        "start": "#### **1.1.",
        "end": "#### **1.2.",
    },
    {
        "key": "sec1_2",
        "label": "1.2. Semejanzas y diferencias entre series/transformadas",
        "start": "#### **1.2.",
        "end": "#### **1.3.",
    },
    {
        "key": "sec1_3",
        "label": "1.3. Función de densidad espectral",
        "start": "#### **1.3.",
        "end": "#### **1.4.",
    },
    {
        "key": "sec1_4",
        "label": "1.4. Aplicación de propiedades",
        "start": "#### **1.4.",
        "end": None,
    },
]

SECTION_INDEX = {item["key"]: item for item in SECTION_DEFS}
DEFAULT_SECTION_KEY = SECTION_DEFS[0]["key"]

NAV_STRUCTURE = [
    {
        "type": "section",
        "label": SECTION_INDEX["presentacion"]["label"],
        "section_key": "presentacion",
        "children": [],
    },
    {
        "type": "group",
        "label": "1. Transformada de Fourier",
        "section_key": None,
        "children": ["sec1_1", "sec1_2", "sec1_3", "sec1_4"],
    },
]


st.set_page_config(page_title="Taller 2 - Señales y Sistemas", layout="wide", page_icon="📈")


def _extract_section(text: str, start: str, end: str | None) -> str:
    start_idx = text.find(start)
    if start_idx == -1:
        return ""
    end_idx = text.find(end, start_idx + len(start)) if end else -1
    if end_idx == -1:
        end_idx = len(text)
    return text[start_idx:end_idx].strip()


def _ensure_nav_selected():
    if "nav_selected" not in st.session_state or st.session_state["nav_selected"] not in SECTION_INDEX:
        st.session_state["nav_selected"] = DEFAULT_SECTION_KEY


def render_navigation_tree() -> str:
    """Emula el panel de navegación de un PDF con expanders por título."""
    _ensure_nav_selected()
    selected = st.session_state["nav_selected"]

    st.sidebar.markdown(
        """
        <style>
        div[data-testid="stSidebar"] button {
            width: 100%;
            justify-content: flex-start;
            text-align: left;
            padding-left: 0.4rem;
            background-color: transparent;
            color: inherit;
            border: none;
        }
        div[data-testid="stSidebar"] button:hover {
            background-color: rgba(255, 255, 255, 0.08);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    nav_container = st.sidebar.container()
    for node in NAV_STRUCTURE:
        if node["type"] == "section":
            sec_key = node["section_key"]
            is_selected = selected == sec_key
            prefix = "▣ " if is_selected else "▢ "
            if nav_container.button(
                f"{prefix}{node['label']}",
                key=f"btn-{sec_key}",
                use_container_width=True,
            ):
                st.session_state["nav_selected"] = sec_key
                selected = sec_key
        elif node["type"] == "group":
            child_keys = node.get("children", [])
            expanded_default = selected in child_keys
            with nav_container.expander(node["label"], expanded=expanded_default):
                for child_key in child_keys:
                    child_label = SECTION_INDEX[child_key]["label"]
                    prefix = "▣ " if selected == child_key else "▢ "
                    if st.button(
                        f"{prefix}{child_label}",
                        key=f"btn-{child_key}",
                        use_container_width=True,
                    ):
                        st.session_state["nav_selected"] = child_key
                        selected = child_key

    st.sidebar.caption("Colapsa o expande cada título como en el panel de un PDF.")
    return selected


@st.cache_resource
def load_markdown_sections(notebook_path: str) -> Dict[str, str]:
    """Carga las celdas markdown del notebook y las separa por sección principal."""
    path = Path(notebook_path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    md_cells = ["".join(cell.get("source", "")) for cell in data.get("cells", []) if cell.get("cell_type") == "markdown"]
    full_md = "\n\n".join(md_cells)
    sections: Dict[str, str] = {}
    for entry in SECTION_DEFS:
        sections[entry["key"]] = _extract_section(full_md, entry["start"], entry["end"])
    return sections


@st.cache_data
def compute_dft_fft_benchmark(lengths: Tuple[int, ...]) -> pd.DataFrame:
    """Compara tiempos de cómputo entre la DFT directa y la FFT."""
    rng = np.random.default_rng(seed=24)

    def dft(signal: np.ndarray) -> np.ndarray:
        n = signal.size
        idx = np.arange(n)
        twiddle = np.exp(-2j * np.pi * idx[:, None] * idx / n)
        return twiddle @ signal

    dft_times, fft_times = [], []
    for length in lengths:
        x = rng.standard_normal(length)

        start = perf_counter()
        _ = dft(x)
        dft_times.append(perf_counter() - start)

        start = perf_counter()
        _ = np.fft.fft(x)
        fft_times.append(perf_counter() - start)

    return pd.DataFrame(
        {
            "Longitud N": list(lengths),
            "Tiempo DFT [s]": dft_times,
            "Tiempo FFT [s]": fft_times,
        }
    )


@st.cache_resource
def load_audio_file(audio_path: str) -> Tuple[np.ndarray, int]:
    """Lee el archivo de audio estéreo empleado en el taller."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de audio en {audio_path}.")
    audio, sample_rate = sf.read(path)
    return audio, sample_rate


def slice_audio(audio: np.ndarray, fs: int, start: float, duration: float) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae un segmento del audio y devuelve la señal y el eje temporal."""
    start = max(0.0, float(start))
    duration = max(0.1, float(duration))
    first = int(start * fs)
    last = min(audio.shape[0], first + int(duration * fs))
    segment = audio[first:last]
    if segment.size == 0:
        raise ValueError("El segmento seleccionado no contiene muestras.")
    time_axis = np.arange(segment.shape[0]) / fs + start
    return segment, time_axis


def plot_time_frequency(t_axis: np.ndarray, time_signal: np.ndarray, f_axis: np.ndarray, spectrum: np.ndarray, title: str):
    """Grafica las versiones en tiempo y frecuencia de una señal estéreo."""
    time_2d = time_signal if time_signal.ndim == 2 else time_signal[:, None]
    spec_2d = spectrum if spectrum.ndim == 2 else spectrum[:, None]
    labels = ["Canal izquierdo", "Canal derecho"]
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for idx in range(time_2d.shape[1]):
        label = labels[idx] if idx < len(labels) else f"Canal {idx+1}"
        color = colors[idx % len(colors)]
        axes[0].plot(t_axis, time_2d[:, idx], label=label, color=color, linewidth=1.0)
        axes[1].plot(f_axis, np.abs(spec_2d[:, idx]), label=label, color=color, linewidth=1.0)
    axes[0].set_xlabel("Tiempo [s]")
    axes[0].set_ylabel("Amplitud")
    axes[0].set_title("Dominio temporal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Frecuencia [Hz]")
    axes[1].set_ylabel("Magnitud")
    axes[1].set_title("Dominio espectral")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def audio_to_bytes(signal: np.ndarray, fs: int) -> bytes:
    """Convierte un arreglo numpy a bytes WAV para usar con st.audio."""
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(signal, -1.0, 1.0).astype(np.float32), fs, format="WAV")
    buffer.seek(0)
    return buffer.read()


def build_filter_mask(freqs: np.ndarray, filter_type: str, low: float | None = None, high: float | None = None) -> np.ndarray:
    """Genera la máscara espectral para los filtros rectangulares utilizados en el taller."""
    abs_freqs = np.abs(freqs)
    mask = np.ones_like(freqs, dtype=float)
    if filter_type == "Pasa bajas":
        cutoff = float(low or 0.0)
        mask = (abs_freqs <= cutoff).astype(float)
    elif filter_type == "Pasa altas":
        cutoff = float(low or 0.0)
        mask = (abs_freqs >= cutoff).astype(float)
    elif filter_type == "Pasa bandas":
        low = float(low or 0.0)
        high = float(high or low)
        mask = ((abs_freqs >= low) & (abs_freqs <= high)).astype(float)
    elif filter_type == "Rechaza bandas":
        low = float(low or 0.0)
        high = float(high or low)
        mask = ((abs_freqs < low) | (abs_freqs > high)).astype(float)
    return mask


def render_benchmark_section():
    st.subheader("Comparación de tiempos de cómputo (1.1 b)")
    df = compute_dft_fft_benchmark(DFT_LENGTHS)
    st.caption("Se generan secuencias reales reproducibles y se mide el tiempo de la implementación $\\mathcal{O}(N^2)$ vs. FFT.")
    st.dataframe(
        df.style.format({"Tiempo DFT [s]": "{:.4e}", "Tiempo FFT [s]": "{:.4e}"}),
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["Longitud N"], df["Tiempo DFT [s]"], "o-", label="DFT directa")
    ax.plot(df["Longitud N"], df["Tiempo FFT [s]"], "s-", label="FFT (numpy)")
    ax.set_xlabel("Longitud N")
    ax.set_ylabel("Tiempo [s]")
    ax.set_title("Escalamiento temporal DFT vs FFT")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(loc="upper left")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_filter_lab():
    st.subheader("Laboratorio de filtrado espectral (1.1 d)")
    try:
        audio, fs = load_audio_file(str(AUDIO_PATH))
    except FileNotFoundError as exc:
        st.warning(str(exc))
        return

    total_duration = audio.shape[0] / fs
    st.caption(f"Archivo: {AUDIO_PATH.name} — {total_duration:.1f} s a {fs} Hz")

    col1, col2 = st.columns(2)
    start_time = col1.number_input("Tiempo inicial del segmento [s]", min_value=0.0, value=30.0, step=0.5)
    duration = col2.number_input("Duración del segmento [s]", min_value=1.0, value=5.0, step=0.5)
    if start_time >= total_duration:
        st.warning("El tiempo inicial excedía la duración total; se ajustó al máximo disponible.")
        start_time = max(0.0, total_duration - duration)
    if start_time + duration > total_duration:
        duration = max(0.5, total_duration - start_time)
        st.info("Se ajustó la duración para mantener el segmento dentro del audio.")

    try:
        segment, time_axis = slice_audio(audio, fs, start_time, duration)
    except ValueError as exc:
        st.error(str(exc))
        return

    spectrum = np.fft.fft(segment, axis=0)
    freq_axis = np.fft.fftfreq(segment.shape[0], d=1 / fs)

    st.markdown("**Referencia:**")
    reference_fig = plot_time_frequency(time_axis, segment, freq_axis, spectrum, "Audio original (5 s)")
    st.pyplot(reference_fig, use_container_width=True)
    plt.close(reference_fig)
    st.audio(audio_to_bytes(segment, fs), sample_rate=fs)

    filter_options = ["Pasa bajas", "Pasa altas", "Pasa bandas", "Rechaza bandas"]
    filter_type = st.selectbox("Tipo de filtro a aplicar", filter_options, index=0)
    nyquist = int(fs / 2)
    slider_min = 10
    slider_max = max(slider_min + 10, nyquist - 1)

    if filter_type in {"Pasa bajas", "Pasa altas"}:
        default_cutoff = 500 if filter_type == "Pasa bajas" else 5000
        cutoff = st.slider(
            "Frecuencia de corte [Hz]",
            min_value=slider_min,
            max_value=slider_max,
            value=int(np.clip(default_cutoff, slider_min, slider_max)),
            step=10,
        )
        mask = build_filter_mask(freq_axis, filter_type, low=cutoff)
        freq_text = f"$f_c = {cutoff}\\ \\mathrm{{Hz}}$"
    else:
        low_default = int(np.clip(500, slider_min, slider_max - 10))
        high_default = int(np.clip(5000, low_default + 10, slider_max))
        if low_default >= high_default:
            low_default, high_default = slider_min, slider_max
        low, high = st.slider(
            "Banda [Hz]",
            min_value=slider_min,
            max_value=slider_max,
            value=(low_default, high_default),
            step=10,
        )
        if low >= high:
            high = min(slider_max, low + 100)
        mask = build_filter_mask(freq_axis, filter_type, low=low, high=high)
        freq_text = f"$f_c \\in [{low}, {high}]\\ \\mathrm{{Hz}}$"

    filtered_spectrum = spectrum * mask[:, None]
    filtered_segment = np.fft.ifft(filtered_spectrum, axis=0).real

    st.markdown(f"**Respuesta filtrada ({filter_type}) {freq_text}:**")
    filtered_fig = plot_time_frequency(time_axis, filtered_segment, freq_axis, filtered_spectrum, f"{filter_type}")
    st.pyplot(filtered_fig, use_container_width=True)
    plt.close(filtered_fig)
    st.audio(audio_to_bytes(filtered_segment, fs), sample_rate=fs)


def main():
    sections = load_markdown_sections(str(NOTEBOOK_PATH))
    if not sections:
        st.error("No fue posible leer las celdas markdown del notebook original.")
        return

    st.sidebar.title("Navegación")
    section_choice = render_navigation_tree()

    section_text = sections.get(section_choice, "")
    if section_text:
        st.markdown(section_text)
    else:
        st.info("La sección no contiene texto adicional en el notebook original.")

    if section_choice == "sec1_1":
        st.divider()
        render_benchmark_section()
        st.divider()
        render_filter_lab()


if __name__ == "__main__":
    main()
