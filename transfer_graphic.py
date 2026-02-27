import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, "данные_перелета.csv")
MU_KERBOL      = 1.1723e18       
R_KERBIN_ORBIT = 13_599_840_256 
R_DUNA_ORBIT   = 20_726_155_264 
A_TR = (R_KERBIN_ORBIT + R_DUNA_ORBIT) / 2.0
E_TR = (R_DUNA_ORBIT - R_KERBIN_ORBIT) / (R_DUNA_ORBIT + R_KERBIN_ORBIT)
T_TR = math.pi * math.sqrt(A_TR**3 / MU_KERBOL) 
df = pd.read_csv(csv_path, decimal=',')
df.columns = df.columns.str.strip()

def col_float(series):
    return pd.to_numeric(
        series.astype(str).str.replace(',', '.', regex=False),
        errors='coerce'
    ).values

time_raw  = col_float(df['TimeSinceMark'])
alt_raw   = col_float(df['AltitudeASL'])        
speed_raw = col_float(df['SpeedOrbital'])       
mass_raw  = col_float(df['Mass']) * 1000        

mask = np.isfinite(time_raw) & np.isfinite(alt_raw) & \
       np.isfinite(speed_raw) & np.isfinite(mass_raw)
time_raw, alt_raw, speed_raw, mass_raw = (
    arr[mask] for arr in (time_raw, alt_raw, speed_raw, mass_raw)
)
start_idx = int(np.argmax(speed_raw))
time_ksp  = time_raw[start_idx:] - time_raw[start_idx]
alt_ksp   = alt_raw[start_idx:]
speed_ksp = speed_raw[start_idx:]
mass_ksp  = mass_raw[start_idx:]

def solve_kepler(M: np.ndarray, e: float, tol: float = 1e-8) -> np.ndarray:
    E_arr = M.copy()
    for _ in range(100):
        dE = (M - E_arr + e * np.sin(E_arr)) / (1.0 - e * np.cos(E_arr))
        E_arr += dE
        if np.max(np.abs(dE)) < tol:
            break
    return E_arr


def transfer_model(t_max_s: float, n_points: int = 800):
    t_end   = min(t_max_s, T_TR)  
    t_model = np.linspace(0, t_end, n_points)
    M_arr   = np.pi * t_model / T_TR     
    E_arr   = solve_kepler(M_arr, E_TR)
    r_model = A_TR * (1.0 - E_TR * np.cos(E_arr))
    v_model = np.sqrt(MU_KERBOL * (2.0 / r_model - 1.0 / A_TR))
    m_model = np.full(n_points, mass_ksp[0])
    return t_model, r_model, v_model, m_model

t_m, r_m, v_m, m_m = transfer_model(t_max_s=time_ksp[-1])

dv_dep = abs(
    math.sqrt(MU_KERBOL * (2 / R_KERBIN_ORBIT - 1 / A_TR)) -
    math.sqrt(MU_KERBOL / R_KERBIN_ORBIT)
)
dv_arr = abs(
    math.sqrt(MU_KERBOL / R_DUNA_ORBIT) -
    math.sqrt(MU_KERBOL * (2 / R_DUNA_ORBIT - 1 / A_TR))
)



#  ГРАФИКИ 

C_MODEL = '#1E88E5'   # синий  — математическая модель
C_KSP   = '#E53935'   # красный — данные игры

fig, axes = plt.subplots(
    3, 1, figsize=(13, 14), sharex=True,
    gridspec_kw=dict(hspace=0.10, top=0.92, bottom=0.08,
                     left=0.09, right=0.97)
)

t_g_d = time_ksp / 86400
t_m_d = t_m      / 86400

x_max = t_g_d[-1]

# 1. Высота  
ax = axes[0]
ax.plot(t_m_d, r_m / 1e9,     lw=2.2, color=C_MODEL,
        label='Мат. модель', zorder=3)
ax.plot(t_g_d, alt_ksp / 1e9, lw=2.0, color=C_KSP,
        label='KSP', ls='--', zorder=2)
ax.set_ylabel('Расстояние от Кербина, Гм', fontsize=11)
ax.set_title('Высота от времени', fontsize=11, pad=5)
ax.legend(fontsize=9, loc='upper left', framealpha=0.88)
ax.grid(True, alpha=0.22)
ax.set_xlim(0, x_max)

# 2. Скорость  
ax = axes[1]
ax.plot(t_m_d, v_m / 1000,       lw=2.2, color=C_MODEL,
        label='Мат. модель', zorder=3)
ax.plot(t_g_d, speed_ksp / 1000, lw=2.0, color=C_KSP,
        label='KSP', ls='--', zorder=2)
ax.set_ylabel('Скорость, км/с', fontsize=11)
ax.set_title('Скорость от времени', fontsize=11, pad=5)
ax.legend(fontsize=9, loc='upper right', framealpha=0.88)
ax.grid(True, alpha=0.22)
ax.set_xlim(0, x_max)

# 3. Масса
ax = axes[2]
ax.plot(t_m_d, m_m,      lw=2.2, color=C_MODEL,
        label='Мат. модель')
ax.plot(t_g_d, mass_ksp, lw=2.0, color=C_KSP,
        label='KSP', ls='--', zorder=2)
ax.set_ylabel('Масса, кг', fontsize=11)
ax.set_xlabel('Время, сут', fontsize=11)
ax.set_title('Масса от времени', fontsize=11, pad=5)
ax.legend(fontsize=9, loc='upper right', framealpha=0.88)
ax.grid(True, alpha=0.22)
ax.set_xlim(0, x_max)

fig.suptitle(
    'Сравнение математической модели и данных KSP\n'
    '(перелёт Кербин → Дюна)',
    fontsize=14, fontweight='bold'
)
save_path = os.path.join(script_dir, "перелет_модель_KSP.png")
plt.savefig(save_path, dpi=160, bbox_inches='tight')
print(f"\nГрафик сохранён: {save_path}")
plt.show()