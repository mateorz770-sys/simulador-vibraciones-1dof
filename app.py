import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.ensemble import RandomForestRegressor
import time

st.set_page_config(page_title="Simulador de Vibraciones 1-DOF", layout="wide")

@st.cache_resource
def entrenar_modelo_final():
    rng = np.random.default_rng(42)
    N = 6000
    m_arr = rng.uniform(10_000, 100_000, N)
    k_arr = rng.uniform(2e6, 10e6, N)
    zeta_arr = rng.uniform(0.02, 0.15, N)
    c_arr = 2 * zeta_arr * np.sqrt(k_arr * m_arr)
    F0_arr = rng.uniform(5_000, 50_000, N)

    wn_arr = np.sqrt(k_arr / m_arr)
    r_arr = np.concatenate([
        rng.uniform(0.85, 1.15, N // 2),
        rng.uniform(0.20, 0.85, N // 4),
        rng.uniform(1.15, 2.00, N // 4)
    ])
    w_arr = r_arr * wn_arr
    x_est_arr = F0_arr / k_arr
    X_teo_arr = (F0_arr / m_arr) / np.sqrt((wn_arr**2 - w_arr**2)**2 + (2 * zeta_arr * wn_arr * w_arr)**2)

    # Pico máximo incluyendo sobrepaso transitorio desde reposo
    overshoot = 1.0 + np.exp(-np.pi * zeta_arr / np.sqrt(1.0 - zeta_arr**2)) * np.exp(-2.5 * np.abs(r_arr - 1.0))
    y_xmax = X_teo_arr * np.where(r_arr < 1.05, np.maximum(1.0, 0.65 * overshoot), 1.0)

    X_train = np.column_stack([m_arr, k_arr, c_arr, F0_arr, w_arr, wn_arr, zeta_arr, r_arr, x_est_arr, X_teo_arr])
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)
    rf.fit(X_train, y_xmax)
    return rf

rf_final = entrenar_modelo_final()

st.title("Simulador Interactivo de Vibraciones y Resonancia (1-DOF)")
st.markdown("Modifica los parámetros físicos del sistema estructural para visualizar en tiempo real la respuesta temporal (**RK45**) y la curva de resonancia (**Random Forest Final**).")

st.sidebar.header("Parámetros del Sistema")
m  = st.sidebar.slider("Masa m [kg]", 10_000, 100_000, 30_000, step=1_000)
k  = st.sidebar.slider("Rigidez k [N/m]", 2_000_000, 10_000_000, 5_000_000, step=200_000)
c  = st.sidebar.slider("Amortiguamiento c [N·s/m]", 5_000, 150_000, 38_730, step=500)
F0 = st.sidebar.slider("Fuerza externa F₀ [N]", 5_000, 50_000, 20_000, step=1_000)
w  = st.sidebar.slider("Frecuencia externa ω [rad/s]", 2.0, 35.0, 12.91, step=0.05)

wn = np.sqrt(k / m)
zeta = c / (2 * np.sqrt(k * m))
r = w / wn
x_est = F0 / k
X_teo = (F0 / m) / np.sqrt((wn**2 - w**2)**2 + (2 * zeta * wn * w)**2)

t_eval = np.linspace(0, 20, 1500)
sol = solve_ivp(
    lambda t, y: [y[1], (F0 * np.cos(w * t) - c * y[1] - k * y[0]) / m],
    [0, 20], [0.0, 0.0], method='RK45', t_eval=t_eval, rtol=1e-6, atol=1e-8
)
x_max_rk = np.max(np.abs(sol.y[0]))

t0 = time.perf_counter()
r_grid = np.linspace(0.20, 2.00, 1000)
w_grid = r_grid * wn
X_teo_grid = (F0 / m) / np.sqrt((wn**2 - w_grid**2)**2 + (2 * zeta * wn * w_grid)**2)
X_in = np.column_stack([
    np.full_like(r_grid, m), np.full_like(r_grid, k), np.full_like(r_grid, c),
    np.full_like(r_grid, F0), w_grid, np.full_like(r_grid, wn),
    np.full_like(r_grid, zeta), r_grid, np.full_like(r_grid, x_est), X_teo_grid
])
curve_ia = rf_final.predict(X_in)
x_max_ia = rf_final.predict([[m, k, c, F0, w, wn, zeta, r, x_est, X_teo]])[0]
dt_ms = (time.perf_counter() - t0) * 1000

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Frecuencia natural ωₙ", f"{wn:.2f} rad/s")
col2.metric("Amortiguamiento ζ", f"{zeta:.3f}")
col3.metric("Relación r = ω/ωₙ", f"{r:.3f}")
col4.metric("x_max Real (RK45)", f"{x_max_rk:.4f} m")
col5.metric("x_max Predicho (IA)", f"{x_max_ia:.4f} m", f"Barrido: {dt_ms:.1f} ms")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=130)

ax1.plot(t_eval, sol.y[0], color='#1f77b4', lw=1.2)
ax1.set_title('Respuesta Temporal x(t) [RK45]', fontweight='bold')
ax1.set_xlabel('Tiempo t [s]')
ax1.set_ylabel('Desplazamiento x(t) [m]')
ax1.grid(True, alpha=0.3)

ax2.axvspan(0.85, 1.15, color='#ffcccc', alpha=0.45, label='Zona resonancia (0.85 ≤ r ≤ 1.15)')
ax2.plot(r_grid, curve_ia, color='#ff7f0e', lw=2.0, label='Curva IA (RF Final)')
ax2.axvline(r, color='#d62728', ls='--', lw=1.4, label=f'Frecuencia actual (r = {r:.2f})')
ax2.set_title('Curva de Resonancia Instantánea (IA)', fontweight='bold')
ax2.set_xlabel('Razón de frecuencias r = ω / ωₙ')
ax2.set_ylabel('Desplazamiento máximo x_max [m]')
ax2.legend(loc='upper right', fontsize=8.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
