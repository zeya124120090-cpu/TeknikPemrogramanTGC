import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.interpolate import griddata, NearestNDInterpolator

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(page_title="Analisis Magnetik 2D", layout="wide")

st.title("🗺 Pengolahan Data Magnetik 2D")
st.markdown("""
Aplikasi ini melakukan pemisahan anomali **Regional** dan **Residual**
serta visualisasi data magnetik berbasis **Streamlit**.
""")

# =========================
# DATA DUMMY
# =========================
def get_dummy_dataframe(grid_size=50):
    x = np.linspace(0, 1000, grid_size)
    y = np.linspace(0, 1000, grid_size)
    X, Y = np.meshgrid(x, y)

    regional = 0.05 * X + 0.02 * Y + 45000
    r = np.sqrt((X - 500)**2 + (Y - 500)**2 + 50**2)
    residual = 5000 * (50 / r)**3
    noise = np.random.normal(0, 1, (grid_size, grid_size))
    total = regional + residual + noise

    df = pd.DataFrame({
        "x": X.flatten(),
        "y": Y.flatten(),
        "t_obs": total.flatten()
    })

    return df.sample(frac=0.8).reset_index(drop=True)

# =========================
# POLYFIT 2D
# =========================
def polyfit2d(x, y, z, order=1):
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    mask = ~np.isnan(z)
    x, y, z = x[mask], y[mask], z[mask]

    if order == 1:
        A = np.c_[np.ones(x.shape), x, y]
    else:
        A = np.c_[np.ones(x.shape), x, y, x**2, x*y, y**2]

    C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    if order == 1:
        return lambda xi, yi: C[0] + C[1]*xi + C[2]*yi
    else:
        return lambda xi, yi: C[0] + C[1]*xi + C[2]*yi + C[3]*xi**2 + C[4]*xi*yi + C[5]*yi**2


# =========================
# INPUT DATA (CSV / XLSX)
# =========================
st.sidebar.header("Input Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload File Data Magnetik",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.warning("Silakan upload file CSV atau XLSX")
    st.stop()

# Baca file sesuai ekstensi
if uploaded_file.name.endswith(".csv"):
    df_input = pd.read_csv(uploaded_file)
else:
    df_input = pd.read_excel(uploaded_file)

st.success("Data berhasil dimuat")
st.write("Preview Data:", df_input.head())

         
 # =========================
# CEK KOLOM OTOMATIS
# =========================
required_cols = {"x", "y", "t_obs"}

if not required_cols.issubset(df_input.columns):
    st.error(
        "File harus memiliki kolom: x, y, t_obs\n"
        f"Kolom ditemukan: {list(df_input.columns)}"
    )
    st.stop()

st.success("Kolom x, y, t_obs terdeteksi")
st.write(df_input[["x", "y", "t_obs"]].head())


# =========================
# GRIDDING
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Parameter Gridding")

x_min, x_max = df_input["x"].min(), df_input["x"].max()
y_min, y_max = df_input["y"].min(), df_input["y"].max()

cell_size = st.sidebar.number_input(
    "Ukuran Sel (meter)",
    min_value=1.0,
    value=(x_max - x_min) / 50,
    step=5.0
)

interp_method = st.sidebar.selectbox(
    "Metode Interpolasi",
    ["linear", "cubic", "nearest"]
)

xi = np.arange(x_min, x_max + cell_size, cell_size)
yi = np.arange(y_min, y_max + cell_size, cell_size)
X, Y = np.meshgrid(xi, yi)

T_obs = griddata(
    (df_input["x"], df_input["y"]),
    df_input["t_obs"],
    (X, Y),
    method=interp_method
)

# =========================
# CHALLENGE – VISUALISASI
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Pengaturan Visualisasi")

cmap_option = st.sidebar.selectbox(
    "Colormap",
    ["jet", "viridis", "plasma", "inferno", "seismic", "coolwarm", "terrain"]
)

scale_mode = st.sidebar.radio(
    "Mode Skala",
    ["Auto", "Manual"],
    horizontal=True
)

if scale_mode == "Manual":
    vmin = st.sidebar.number_input("vmin", value=-5000.0)
    vmax = st.sidebar.number_input("vmax", value=5000.0)
else:
    vmin, vmax = None, None

# =========================
# VISUALISASI DATA
# =========================
st.subheader("1. Visualisasi Data")
tab1, tab2 = st.tabs(["Peta Kontur", "Sebaran Titik"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    c1 = ax1.contourf(X, Y, T_obs, levels=25, cmap=cmap_option, vmin=vmin, vmax=vmax)
    ax1.scatter(df_input["x"], df_input["y"], c="k", s=5, alpha=0.2)
    ax1.set_aspect("equal")
    fig1.colorbar(c1, ax=ax1, label="nT")
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sc = ax2.scatter(df_input["x"], df_input["y"], c=df_input["t_obs"], cmap=cmap_option)
    fig2.colorbar(sc, ax=ax2, label="nT")
    st.pyplot(fig2)

st.divider()

# =========================
# REGIONAL – RESIDUAL
# =========================
st.subheader("2. Pemisahan Regional – Residual")

mask_nan = np.isnan(T_obs)
if np.any(mask_nan):
    interp_nn = NearestNDInterpolator(
        list(zip(X[~mask_nan], Y[~mask_nan])),
        T_obs[~mask_nan]
    )
    T_filled = interp_nn(X, Y)
else:
    T_filled = T_obs

method = st.selectbox(
    "Metode Pemisahan",
    ["2D Moving Average", "Trend Surface Analysis"]
)

if method == "2D Moving Average":
    window = st.slider("Lebar Window (grid)", 3, 200, 9, step=2)
    Regional = uniform_filter(T_filled, size=window)
else:
    order = st.radio("Orde Polinom", [1, 2], horizontal=True)
    poly = polyfit2d(X, Y, T_obs, order)
    Regional = poly(X, Y)

Residual = T_obs - Regional

# =========================
# HASIL
# =========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Regional")
    fig_r, ax_r = plt.subplots(figsize=(6, 5))
    c_r = ax_r.contourf(X, Y, Regional, levels=20, cmap=cmap_option, vmin=vmin, vmax=vmax)
    fig_r.colorbar(c_r, ax=ax_r)
    st.pyplot(fig_r)

with col2:
    st.markdown("### Residual")
    fig_res, ax_res = plt.subplots(figsize=(6, 5))

    if scale_mode == "Auto":
        valid = Residual[~np.isnan(Residual)]
        lim = np.percentile(np.abs(valid), 98)
        vmin_r, vmax_r = -lim, lim
    else:
        vmin_r, vmax_r = vmin, vmax

    c_res = ax_res.contourf(
        X, Y, Residual,
        levels=20,
        cmap=cmap_option,
        vmin=vmin_r,
        vmax=vmax_r
    )
    fig_res.colorbar(c_res, ax=ax_res)
    st.pyplot(fig_res)

# =========================
# CHALLENGE TAMBAHAN
# =========================
if st.button("💾 Simpan Peta Residual"):
    fig_res.savefig("peta_residual.png", dpi=300, bbox_inches="tight")
    st.success("Peta Residual berhasil disimpan")
