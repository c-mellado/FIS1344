# Catalina Mellado Valdés
# 21.773.395-2
# FIS1344 Métodos Numéricos y Probabilidades
# 24/11/2026
# Tarea 9.1

import numpy as np

# rk4 (con un contador de las evaluaciones de f)
def rk4(f, a, b, y0, N, imprimir = True):

    h = (b-a)/N
    t = a
    w = y0
    ts = np.zeros(N+1)
    sol = np.zeros(N+1)
    n_evaluaciones = 0   #contador

    for i in range(N):
        ts[i] = t
        sol[i] = w
        k1 = h*f(t, w)
        k2 = h*f(t+h/2, w+k1/2)
        k3 = h*f(t+h/2, w+k2/2)
        k4 = h*f(t + h, w + k3)
        n_evaluaciones += 4 # 4 evaluaciones por paso
        w = w + (k1 + 2*k2 + 2*k3 + k4)/6
        t = t + h

    ts[N] = t
    sol[N] = w
    if imprimir:
        print("RK4: evaluaciones de f =", n_evaluaciones)
    return ts, sol

# RKF (con contador de evaluaciones de f)
class MinStep(Exception):
    pass

def rkf(f, a, b, y0, tol, hmin, hmax, imprimir = True):
    t = a
    w= y0
    h = hmax
    flag = 1
    n_evaluaciones = 0 # contador

    ts = [t]
    sol = [w]
    hs = []
    rs = []

    while flag:

        k1 = h*f(t, w)
        k2 = h*f(t + h/4, w + k1/4)
        k3 = h*f(t + 3*h/8, w + 3*k1/32 + 9*k2/32)
        k4 = h*f(t + 12*h/13, w + 1932*k1/2197 - 7200*k2/2197 + 7296*k3/2197)
        k5 = h*f(t + h, w + 439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104)
        k6 = h*f(t + h/2, w - 8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40)
        n_evaluaciones += 6   # 6 evaluaciones por intento de paso (considera también los rechazados)

        # estima del error (dif. entre los RK de orden 4 y 5)
        R = abs(k1/360 - 128*k3/4275 - 2197*k4/75240 + k5/50 + 2*k6/55)

        # en el caso de que el error sea menor que la tolerancia se acepta el paso
        if R <= tol:
            t += h
            w = w + 25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5
            ts.append(t)
            sol.append(w)
            hs.append(h)
            rs.append(R)

        # factor para actualizar h (Si R = 0 evitamos dividir por 0 y se usa el q máx.)
        if R == 0:
            q = 4
        else:
            q = 0.84*(tol/R)**(1/4) # el factor 0.84 lo vuelve más conservador

        # q mín. y máx. para no quedarse atrapado/perder detalle
        # (los factores 0.1 y 4 se pueden cambiar pero utilizo los mismos que vimos en la clase)
        if q <= 0.1:   
            h = 0.1*h  #mín para no quedarse atrapado
        elif q >= 4:
            h = 4*h
        else:
            h = q*h
        if h > hmax:
            h = hmax

        # si nos pasamos de b, terminamos y el último paso debe llegar a b
        if t >= b:
            flag = 0
        elif t + h > b:
            h = b - t
        elif h < hmin:
            raise MinStep("Se excedió el paso mínimo.")

    if imprimir:
        print("RKF: evaluaciones de f =", n_evaluaciones)
    return np.array(ts), np.array(sol), np.array(hs), np.array(rs)

# problema de valor inicial
def f(t, y):
    return t*np.exp(3*t) - 2*y

def y_exacta(t):   #solución exacta dada
    return t*np.exp(3*t)/5 - np.exp(3*t)/25 + np.exp(-2*t)/25

# valores dados
a, b, y0 = 0, 1, 0

# error máximo en los nodos respecto a la solución exacta
def error_max(ts, sol):
    return np.max(np.abs(sol - y_exacta(ts))) 

# RKF partiendo con una tolerancia de 1e-6, se reduce la tolerancia hasta que el error máx. real sea menor a 1e-6 
# la tolerancia acota el error LOCAL estimado (en cada paso), no el global
tol = 1e-6
ts_rkf, sol_rkf, hs_rkf, rs_rkf = rkf(f, a, b, y0, tol, 0.01, 0.25, imprimir = False)
error_rkf =  error_max(ts_rkf, sol_rkf)
while error_rkf >= 1e-6:
    tol = tol/10
    ts_rkf, sol_rkf, hs_rkf, rs_rkf = rkf(f, a, b, y0, tol, 0.01, 0.25, imprimir = False)
    error_rkf = error_max(ts_rkf, sol_rkf)

# RK4 se aumenta N hasta que su error máx sea menor o igual al de RKF, es decir, que tengan errores similares
N = 4
ts_rk4, sol_rk4 = rk4(f, a, b, y0, N, imprimir = False)
error_rk4 = error_max(ts_rk4, sol_rk4)
while error_rk4 > error_rkf:
    N += 1
    ts_rk4, sol_rk4 = rk4(f, a, b, y0, N, imprimir = False)
    error_rk4 = error_max(ts_rk4, sol_rk4)

# se repiten las soluciones finales imprimiendo el número de evaluaciones
rkf(f, a, b, y0, tol, 0.01, 0.25)
rk4(f, a, b, y0, N)

print("RKF: tolerancia =", tol, "| error máximo:", error_rkf)
print("RK4: N =", N, "| error máximo:", error_rk4)

# discusión final
print("\nDiscusión:")
print(f"Con errores similares, {error_rkf:.3e} para RKF y {error_rk4:.3e} para RK4, RKF necesita menos evaluaciones de f, \nes decir, tiene menos costo computacional en este caso (véase los conteos impresos en la terminal).")
print("RKF es preferible aquí porque adapta el paso a la solución (que crece como e^(3t)) \ny estima el error en cada paso, se controla con una tolerancia sin conocer la solución exacta.")
print("En RK4 se usa un paso fijo y no se estima el error, además el N se escoge con ensayo y error \ncomparando con la solución exacta (aunque no es lo usual tenerla).")
print("Aun así, RKF usa 6 evaluaciones por intento de paso (considerando los rechazados) vs. 4 de RK4, \nentonces la ventaja de los métodos va a depender del problema.")


# Nota: 7.0
# Base: 1/1; funcionamiento: 4/4; comentarios: 1/1; claridad: 1/1.
