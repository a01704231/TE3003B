import numpy as np
import matplotlib.pyplot as plt

# funciones de predicción
def mmk_f(A, mkp, B, uk):
    mmk = A @ mkp + B @ uk
    return mmk

def eek_f(A, ekp, qk):
    eek = A @ ekp @ np.transpose(A) + qk
    return eek

# funciones de corrección
def kk_f(eek, C, rk):
    Ct = np.transpose(C)
    kk = eek @ Ct @ np.linalg.inv(C @ eek @ Ct + rk)
    return kk

def mk_f(mmk, kk, zk, C):
    mk = mmk + kk @ (zk - C @ mmk)
    return mk

def ek_f(eek, kk, C):
    ek = eek - kk @ C @ eek
    return ek

# condiciones iniciales
ax = 1
ay = 0.5
dt = 0.1
e0 = 0.1 * np.identity(4)
q = 0.1 * np.identity(4)
r = 5 * np.identity(2)
m0 = np.array([[0], [0], [0], [0]])
u = np.array([[ax], [ay]])
A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
B = np.array([[0.5 * dt ** 2, 0], [dt, 0], [0, 0.5 * dt ** 2], [0, dt]])
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
m_prev = m0
e_prev = e0

# declaración de incertidumbre
n = 0
nn = []
px = []
py = []
u1_px = []
u1_py = []
u2_px = []
u2_py = []
vx = []
vy = []
u1_vx = []
u1_vy = []
u2_vx = []
u2_vy = []

while n < 150:
    n += 1
    
    # predicción
    mm = mmk_f(A, m_prev, B, u)
    ee = eek_f(A, e_prev, q)
    if n != 30 and n != 60 and n != 90 and n != 120 and n != 150:

        # guardar datos previos
        m_prev = mm
        e_prev = ee
    else:

        # corrección
        z = [[mm[0][0] + np.random.normal(0, 5)], [mm[2][0] + np.random.normal(0, 5)]]
        k = kk_f(ee, C, r)
        m = mk_f(mm, k, z, C)
        e = ek_f(ee, k, C)
        
        # guardar datos previos
        m_prev = m
        e_prev = e
        print("\n")
        print(n)
        print("mm")
        print(mm)
        print("ee")
        print(ee)
        print("k")
        print(k)
        print("m")
        print(m)
        print("e")
        print(e)

    # guardar incertidumbre
    nn.append(n)
    px.append(m_prev[0][0])
    py.append(m_prev[2][0])
    u1_px.append(m_prev[0][0] + e_prev[0][0])
    u1_py.append(m_prev[2][0] + e_prev[2][2])
    u2_px.append(m_prev[0][0] - e_prev[0][0])
    u2_py.append(m_prev[2][0] - e_prev[2][2])
    vx.append(m_prev[1][0])
    vy.append(m_prev[3][0])
    u1_vx.append(m_prev[1][0] + e_prev[1][1])
    u1_vy.append(m_prev[3][0] + e_prev[3][3])
    u2_vx.append(m_prev[1][0] - e_prev[1][1])
    u2_vy.append(m_prev[3][0] - e_prev[3][3])

# valores de escala de gráficas
min_p = min([min(u2_px), min(u2_py)])
max_p = max([max(u1_px), max(u1_py)])
min_v = min([min(u2_vx), min(u2_vy)])
max_v = max([max(u1_vx), max(u1_vy)])

# crear gráficas de posición
fig = plt.figure()
ax = fig.add_subplot(221)
ax.plot(nn, px, 'b')
ax.plot(nn, u1_px, 'g')
ax.plot(nn, u2_px, 'g')
plt.ylim(min_p, max_p)
plt.xlabel("t")
plt.ylabel("x")
plt.fill_between(nn, u1_px, px, color='y', alpha=0.3)
plt.fill_between(nn, u2_px, px, color='y', alpha=0.3)
ax = fig.add_subplot(222)
ax.plot(nn, py, 'b')
ax.plot(nn, u1_py, 'g')
ax.plot(nn, u2_py, 'g')
plt.ylim(min_p, max_p)
plt.xlabel("t")
plt.ylabel("y")
plt.fill_between(nn, u1_py, py, color='y', alpha=0.3)
plt.fill_between(nn, u2_py, py, color='y', alpha=0.3)

# crear gráficas de velocidad
ax = fig.add_subplot(223)
ax.plot(nn, vx, 'b')
ax.plot(nn, u1_vx, 'g')
ax.plot(nn, u2_vx, 'g')
plt.ylim(min_v, max_v)
plt.xlabel("t")
plt.ylabel("vel x")
plt.fill_between(nn, u1_vx, vx, color='y', alpha=0.3)
plt.fill_between(nn, u2_vx, vx, color='y', alpha=0.3)
ax = fig.add_subplot(224)
ax.plot(nn, vy, 'b')
ax.plot(nn, u1_vy, 'g')
ax.plot(nn, u2_vy, 'g')
plt.ylim(min_v, max_v)
plt.xlabel("t")
plt.ylabel("vel y")
plt.fill_between(nn, u1_vy, vy, color='y', alpha=0.3)
plt.fill_between(nn, u2_vy, vy, color='y', alpha=0.3)

# mostrar gráficas
plt.show()

