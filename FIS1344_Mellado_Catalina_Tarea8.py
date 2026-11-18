# Catalina Mellado Valdés
# 21.773.395-2
# FIS1344 Métodos Numéricos y Probabilidades
# 17/11/2026
# Tarea 8.5

import numpy as np
import matplotlib.pyplot as plt

# tres puntos con datos tabulados (según la clase)
def deriv3_datos(fx0, fx2, h):
    return (fx2 - fx0)/(2*h)  #recibe f(x-h) y f(x+h)

# tercera derivada a cinco puntos (deducida con taylor)
# con S_k = f(x+kh) - f(x-kh) = (2*k*h)f' + (((k*h)^3)/3)f''' + (((k*h)^5)/60)f^(5) + ...
# S_1 = (2*h)f' + ((h^3)/3)f''' + ((h^5)/60)f^(5)
# S_2 = (4*h)f' + ((8*h^3)/3)f''' + ((32*h^5)/60)f^(5)
# S_2 - 2*S_1 se elimina f':  S_2 - 2*S_1 = (2*h^3)f''' + ((h^5)/2)f^(5)
# por lo tanto: f'''(x) =[f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)]/(2h^3)  -  ((h^2)/4)f^(5)

def tercera_deriv_datos(fm2, fm1, fp1, fp2, h):
    return (fp2 - 2*fp1 + 2*fm1 - fm2)/(2*h**3)

# comprobación de que es exacta para polinomios de grado menor o igual a 4
x0, hh = 0.79, 0.02
for g, g3 in [(lambda x: x**3, lambda x: 6 + 0*x), (lambda x: x**4, lambda x: 24*x)]:
    est = tercera_deriv_datos(g(x0 - 2*hh), g(x0 - hh), g(x0 + hh), g(x0 + 2*hh), hh)
    assert np.isclose(est, g3(x0)), "fórmula de f''' incorrecta"

# datos
sigma = np.array([0.75, 0.77, 0.79, 0.81, 0.83])
n = np.array([1.1324e-3, 1.1376e-3, 1.1386e-3, 1.1454e-3, 1.1474e-3])
delta = 1e-6   #ruido en n
h = 0.02   #paso entre datos
i = 2   #indice de sigma = 0.79

# derivada tres puntos en sigma = 0.79
dn = deriv3_datos(n[i-1], n[i+1], h)

# error del método
# |f'(x) - aprox| = (h^2)/6 |f'''(xi)|   |  f''' se estima con la  fórmula de cinco puntos
f3 = tercera_deriv_datos(n[i-2], n[i-1], n[i+1], n[i+2], h)
error_met = (h**2/6)*abs(f3)

# error por ruido y cota total
# |error| menor o igual a delta/h + M*(h^2)/6   con M = |f'''|
error_ruido = delta/h
error_tot = error_ruido + error_met

# ver tamaño adecuado de h mediante el paso óptimo y comparación con h = 2*0.02
# delta/h + M*(h^2)/6 se minimiza -> h_opt^3 = 3*delta/M  |  error_min = M*(h_opt^2)/2
M = abs(f3)
h_opt = (3*delta/M)**(1/3)
error_min = (M*h_opt**2)/2

# con h = 0.04 se usan sigma 0.75, 0.79, 0.83 (suponiendo f''' ~ cte)
h2 = 2*h
dn2 = deriv3_datos(n[i-2], n[i+2], h2)
error_tot2 = delta/h2 + (h2**2/6)*M

# incertidumbres de f''' por el ruido
#(suma de los coef. que multiplican cada dato en la fórmula de cinco puntos |coef.| ->  1+2+2+1 = 6 )
incertidumbre_f3 = 6*delta/(2*h**3)

# la estimación de f''' es poco fiable si M puede llegar a  |f'''| + incertidumbre (debido al ruido de los datos)
M_max = M + incertidumbre_f3
h_opt_min = (3*delta/M_max)**(1/3)  #h_opt si M fuera el máx. compatible
M_cruce = delta/h**3  #para |f'''| < M_cruce , h2 da menor cota que h

# cota de error en función de h para ver como cambia el error de la derivada al modificar el tamaño del paso h
hs = np.logspace(-2.5, -0.5, 200)
plt.figure()
plt.loglog(hs, delta/hs, 
           label= "ruido: delta/h",
           linestyle = "--", linewidth = 1.5,
           color = "royalblue")
plt.loglog(hs, hs**2/6*M, 
           label= "método: (h^2)M/6",
           linestyle = ":", linewidth = 1.5,
           color = "deeppink")
plt.loglog(hs, delta/hs + hs**2/6*M, 
           label= "cota total",
           linewidth = 1,
           color = "darkviolet")
plt.axvline(h, color="slategrey", label = f"h = {h:.2e}")
plt.axvline(h_opt, color = "gold", label=f"h óptimo = {h_opt:.2e}")
plt.xlabel("h")
plt.ylabel("cota del error")
plt.grid(True, alpha=0.4)
plt.legend()
plt.title("Error de la derivada a tres puntos")
plt.show()

# resultados
print(f"dn/dsigma (h = {h:.2e}) = {dn:.4e}")
print(f"f''' estimada (5 puntos) = {f3:.4e}")
print(f"Error del método = {error_met:.2e}")
print(f"Error por ruido (delta/h) = {error_ruido:.2e}")
print(f"Cota total = {error_tot:.2e}")
print(f"dn/dsigma = {dn:.4e} +- {error_tot:.2e}\n")
print(f"h óptimo = {h_opt:.2e} (error mín. {error_min:.2e})")
print(f"Con h = {h2:.2e}: dn/dsigma = {dn2:.4e}, cota total = {error_tot2:.2e}")
print(f"Incertidumbre de f''' por ruido = +-{incertidumbre_f3:.2e} (comparar con |f'''| = {M:.2e})")
print(f"Estimación de |f'''| considerando el ruido: hasta {M_max:.2e} y h óptimo asociado = {h_opt_min:.2e}")
print(f"h = {h2:.2e} es mejor que h = {h:.2e} solo si |f'''| < {M_cruce:.2e}")
print("\nConclusión:")
print("Con la f''' estimada el ruido domina y un paso h mayor reduce la cota.")
print(f"Sin embargo, f''' tiene una incertidumbre de hasta {incertidumbre_f3:.2e} debido al ruido (|f'''| estimada = {M:.2e}).")
print("Entonces NO se puede asegurar que aumentar el paso SIEMPRE mejore el resultado.")
print("El paso óptimo calculado es una estimación basada en la tercera derivada obtenida.")
print(f"Aun así reducir h no mejora la cota significativamente, por lo tanto h={h:.2e} es adecuado (se puede considerar ligeramente pequeño).")
