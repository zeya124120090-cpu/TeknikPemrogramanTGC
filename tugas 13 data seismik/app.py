import streamlit as st
import segyio
import numpy as np
import matplotlib.pyplot as plt

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Visualisasi Seismik SEG-Y",
    layout="wide"
)

st.title("📊 Visualisasi Data Seismik SEG-Y")
st.markdown("Aplikasi sederhana untuk menampilkan penampang seismik post-stack")

# =========================
# SIDEBAR - PENGATURAN
# =========================
st.sidebar.header("⚙ Pengaturan Visualisasi")

# 1. Colormap
colormap = st.sidebar.selectbox(
    "Pilih Colormap",
    ["gray", "seismic", "viridis", "plasma"]
)

# 2. Balik sumbu waktu
invert_time = st.sidebar.checkbox("Balik Sumbu Waktu")

# 3. Mode skala amplitudo
scale_mode = st.sidebar.radio(
    "Mode Skala Amplitudo",
    ["Auto", "Manual"]
)

# 4. Slider vmin & vmax (dipakai kalau Manual)
vmin = st.sidebar.slider(
    "vmin (amplitudo minimum)",
    -5000, 5000, -1000
)

vmax = st.sidebar.slider(
    "vmax (amplitudo maksimum)",
    -5000, 5000, 1000
)
plot_type = st.sidebar.selectbox(
    "Tipe Plot",
    ["Image", "Wiggle"]
)
# =========================
# LOAD FILE SEG-Y (FINAL & AMAN)
# =========================
uploaded_file = st.file_uploader(
    "📂 Upload File SEG-Y",
    type=["sgy", "segy"]
)

if uploaded_file is None:
    st.warning("Silakan upload file SEG-Y terlebih dahulu")
    st.stop()

import tempfile

with tempfile.NamedTemporaryFile(delete=False, suffix=".sgy") as tmp:
    tmp.write(uploaded_file.getvalue())
    segy_path = tmp.name

with segyio.open(segy_path, "r", ignore_geometry=True) as f:
    seismic_data = segyio.tools.collect(f.trace[:])

seismic_section = seismic_data

# =========================
# PLOT SEISMIK
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

if plot_type == "Image":
    # Image plot (pakai transpose)
    im = ax.imshow(
        seismic_section.T,
        cmap=colormap,
        aspect="auto"
    )
    plt.colorbar(im, ax=ax, label="Amplitude")

elif plot_type == "Wiggle":
    # Wiggle plot (tanpa transpose)
    n_traces, n_samples = seismic_section.shape
    t = np.arange(n_samples)

    for i in range(n_traces):
        trace = seismic_section[i]
        max_amp = np.max(np.abs(trace))
        if max_amp == 0:
            continue

        trace_norm = trace / max_amp
        ax.plot(
            i + trace_norm,
            t,
            color="black",
            linewidth=0.5
        )

    ax.invert_yaxis()
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time Sample")

ax.set_title("Penampang Seismik")
st.pyplot(fig)
