import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import interp1d
import os


G        = 6.67430e-11
g0       = 9.81
M_KERBIN = 5.2915793e22
R_KERBIN = 600_000      
ATM_H    = 70_000        
P0       = 101_325      
T_ATM    = 293.15        
M_AIR    = 0.028965
R_GAS    = 8.314

def gravity(h):
    return G * M_KERBIN / (R_KERBIN + max(h, 0.0)) ** 2

def atm_density(h):
    if h >= ATM_H:
        return 0.0
    g_h = gravity(h)
    p = P0 * np.exp(-M_AIR * g_h * max(h, 0.0) / (R_GAS * T_ATM))
    return p * M_AIR / (R_GAS * T_ATM)


script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path   = os.path.join(script_dir, "данные_взлета.csv")

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

time_ksp  = df['TimeSinceMark'].values
alt_ksp   = df['AltitudeASL'].values       
speed_ksp = df['SpeedSurface'].values      
mass_ksp  = df['Mass'].values * 1000         
pitch_ksp = df['Pitch'].values               

STAGES = [
    dict(name="Ст. 5: 4×RT-30 + осн. двиг.",
         t0=0.0,    t1=59.2,
         m0=92_883, m1=45_778,
         F_atm=2_300_000, F_vac=2_560_000,
         S=30.0, Cx_atm=0.42, Cx_vac=0.20),

    dict(name="Ст. 4: сброс 2-й пары RT-30",
         t0=59.2,   t1=60.74,
         m0=45_778, m1=40_244,
         F_atm=345_000, F_vac=415_000,
         S=22.0, Cx_atm=0.38, Cx_vac=0.18),

    dict(name="Ст. 3: LV-T30 + LV-T45",
         t0=60.74,  t1=134.22,
         m0=40_244, m1=29_440,
         F_atm=345_000, F_vac=415_000,
         S=18.0, Cx_atm=0.32, Cx_vac=0.15),

    dict(name="Пауза (нет тяги)",
         t0=134.22, t1=221.8,
         m0=29_440, m1=29_440,
         F_atm=0,   F_vac=0,
         S=12.0, Cx_atm=0.28, Cx_vac=0.05),

    dict(name="Ст. 2а: разгонный блок",
         t0=221.8,  t1=250.7,
         m0=29_440, m1=21_395,    
         F_atm=0,   F_vac=908_000, 
         S=10.0, Cx_atm=0.25, Cx_vac=0.05),

    dict(name="Ст. 2б: LV-909 (малый)",
         t0=251.3,  t1=275.5,
         m0=8_029,  m1=7_667,
         F_atm=0,   F_vac=52_100,  
         S=5.0,  Cx_atm=0.25, Cx_vac=0.05),
]

def get_stage(t):
    for s in STAGES:
        if s['t0'] <= t < s['t1']:
            return s
    return STAGES[-1]

def stage_mass(s, t):
    if s['m0'] == s['m1']:
        return float(s['m0'])
    f = (t - s['t0']) / (s['t1'] - s['t0'])
    return s['m0'] + (s['m1'] - s['m0']) * min(max(f, 0.0), 1.0)

pitch_fn = interp1d(time_ksp, pitch_ksp, kind='linear',
                    bounds_error=False,
                    fill_value=(pitch_ksp[0], pitch_ksp[-1]))

def theta(t):
    pitch = float(pitch_fn(t))      
    return np.deg2rad(90.0 - pitch)  

speed_fn = interp1d(time_ksp, speed_ksp, kind='linear',
                    bounds_error=False, fill_value=(speed_ksp[0], speed_ksp[-1]))
alt_fn   = interp1d(time_ksp, alt_ksp,   kind='linear',
                    bounds_error=False, fill_value=(alt_ksp[0], alt_ksp[-1]))

def simulate():
    dt = 0.2   
    t   = 0.0
    h   = float(alt_ksp[0]) 
    p0  = np.radians(float(pitch_fn(0.0)))
    vx  = 0.0
    vy  = 0.0
    m   = float(mass_ksp[0])

    out_t, out_h, out_v, out_m = [], [], [], []

    while t <= 276.5:
        s = get_stage(t)
        m = stage_mass(s, t)
        p_h = P0 * np.exp(-M_AIR * gravity(h) * max(h, 0.0) / (R_GAS * T_ATM)) \
              if h < ATM_H else 0.0
        af  = 1.0 - p_h / P0               
        F   = s['F_atm'] + af * (s['F_vac'] - s['F_atm'])
        Cx  = s['Cx_atm'] + af * (s['Cx_vac'] - s['Cx_atm'])
        th = theta(t)
        Fx = F * np.sin(th)
        Fy = F * np.cos(th)
        rho = atm_density(h)
        v   = np.hypot(vx, vy)
        Fd  = 0.5 * Cx * rho * s['S'] * v ** 2
        if v > 0:
            Fdx = -Fd * vx / v
            Fdy = -Fd * vy / v
        else:
            Fdx = Fdy = 0.0
        g_h = gravity(h)
        a_centrifugal = (vx ** 2) / (R_KERBIN + max(h, 0.0))
        ax  = (Fx + Fdx) / m
        ay  = (Fy - m * g_h + Fdy) / m + a_centrifugal
        vx += ax * dt
        vy += ay * dt
        h   = max(h + vy * dt, 0.0)
        t  += dt
        v_now = np.hypot(vx, vy)
        if v_now > 1.0:
            pitch_now = float(pitch_fn(t))
            p_rad = np.radians(pitch_now)       
            vx = v_now * np.cos(p_rad)             
            vy = v_now * np.sin(p_rad)             

        out_t.append(t)
        out_h.append(h)
        out_v.append(np.hypot(vx, vy))
        out_m.append(m)

    return map(np.array, (out_t, out_h, out_v, out_m))

V_ROT_KERBIN = 2 * np.pi * R_KERBIN / 21_600  

t_m, h_m, v_m, m_m = simulate()
v_m_surface = np.maximum(v_m - V_ROT_KERBIN, 0.0)

#  ГРАФИКИ
fig, axes = plt.subplots(
    3, 1, figsize=(13, 14), sharex=True,
    gridspec_kw=dict(hspace=0.10, top=0.92, bottom=0.08,
                     left=0.09, right=0.97))

C_MODEL = '#1E88E5'   
C_KSP   = '#E53935'   

# 1. Высота 
ax = axes[0]
ax.plot(t_m,      h_m / 1000,     lw=2.2, color=C_MODEL, label='Мат. модель', zorder=3)
ax.plot(time_ksp, alt_ksp / 1000, lw=2.0, color=C_KSP,   label='KSP',
        ls='--', zorder=2)
ax.set_ylabel('Высота, км', fontsize=11)
ax.set_title('Высота от времени', fontsize=11, pad=5)
ax.legend(fontsize=9, loc='upper left', framealpha=0.88)
ax.grid(True, alpha=0.22)


# 2. Скорость
ax = axes[1]
ax.plot(t_m,      v_m_surface / 1000,  lw=2.2, color=C_MODEL, label='Мат. модель', zorder=3)
ax.plot(time_ksp, speed_ksp / 1000,    lw=2.0, color=C_KSP,   label='KSP',
        ls='--', zorder=2)
ax.set_ylabel('Скорость, км/с', fontsize=11)
ax.set_title('Скорость от времени', fontsize=11, pad=5)
ax.legend(fontsize=8.5, loc='upper left', framealpha=0.88)
ax.grid(True, alpha=0.22)


#  3. Масса 
ax = axes[2]
ax.plot(t_m,      m_m / 1000,      lw=2.2, color=C_MODEL, label='Мат. модель', zorder=3)
ax.plot(time_ksp, mass_ksp / 1000, lw=2.0, color=C_KSP,   label='KSP',
        ls='--', zorder=2)
ax.set_ylabel('Масса, т', fontsize=11)
ax.set_xlabel('Время, с', fontsize=11)
ax.set_title('Масса от времени', fontsize=11, pad=5)
ax.legend(fontsize=9, loc='upper right', framealpha=0.88)
ax.grid(True, alpha=0.22)


fig.suptitle(
    'Сравнение математической модели и данных KSP\n'
    '(взлёт с Кербина → орбита 100 км)',
    fontsize=14, fontweight='bold')

dh = abs(h_m[-1] - alt_ksp[-1]) / 1000
dv = abs(v_m_surface[-1] - speed_ksp[-1]) / 1000
dm = abs(m_m[-1] - mass_ksp[-1]) / 1000
save_path = os.path.join(script_dir, "сравнение_модель_KSP.png")
plt.savefig(save_path, dpi=160, bbox_inches='tight')
plt.show()