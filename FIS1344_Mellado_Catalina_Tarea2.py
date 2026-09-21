# Catalina Belén Mellado Valdés
# 21.773.395-2
# FIS1344 Métodos Numéricos y Probabilidades
# Tarea 2.5
# 20 de Septiembre 2026

# Ejercicio 1.2.6 del libro “Fundamentals of Numerical Computation: Julia Edition” de Driscoll y Braun.
# Encuentre el número de condicionamiento para el problema de encontrar las raices del polinomio cuadrático 
# p(x) = ax^2 + bx + c  bajo cambios al coeficiente b.

#lib
import numpy as np


# épsilon de máquina para num float
eps_maq = np.finfo(float).eps #np.finfo(float) características tipo float al valor .eps

#entrada e interacción con la terminal para ingresar coeficientes de p(x)
enter = input('Ingrese coeficientes a b c separados por espacios para calcular raíces y número de condicionamiento:\n').split() # input() lee línea de texto y .split() corta por espacios
if len(enter) != 3:  
    raise SystemExit('Se deben ingresar 3 coeficientes')
try:
    a, b, c = map(float, enter)
except ValueError:
    raise SystemExit('Los 3 coeficientes deben ser números')
if a == 0:
    raise SystemExit('a = 0 implica p(x) no es cuadrático') #mensaje cuando p(x) no tiene grado 2

#discriminante ec. cuadratica
def dscr(a, b, c):
   return b**2 - 4*a*c 

# definición funciones para calcular kpp_b


def rts(a, b, c): #función para las raíces desde la ec. cuadrática polinomio p(x)
    D = dscr(a, b, c)
    r = np.emath.sqrt(D) # variable para calcular raíz del discriminante, núm. complejo si D < 0 (sale con j)
    return  (-b + r) / (2*a) , (-b - r) / (2*a) # devuelve las raíces r1 y r2 (fórmula general ec. cuadrática)


# derivación implicita para kpp_b (respecto b, con a y c ctes y r' es dr/db)
# a*r^2 + b*r + c = 0 
# 2*a*r*r' + r + b*r' = 0
# r' = -r/(2ar + b)  
# kpp_b = |b*r' / r| 
# kpp_b = |b / (2*a*r + b)|
 
 
def kpp_b (a, b, c): #función para kappa b (núm. de condicionamiento bajo cambios en b)
    D = dscr(a, b, c)
    if D == 0: # en caso raíz doble: 2*a*r + b = 0
        return np.inf if b != 0 else np.nan
    r1 , _ = rts(a, b, c)
    denom = 2*a*r1 + b
    return abs((b)/(denom))#devuelve el valor listo de kpp_b 

# guardar valores y resultados en la terminal
r1, r2 = rts(a, b, c)
k = kpp_b(a, b, c)
print('\nRaíces:\n r1 =', r1, '\n r2 =', r2)
print('\nNúmero de condicionamiento:\nkappa_b =', k)

# verificación por diferencia finita cuando b ≠ 0, r1 ≠ 0, D ≠ 0


# definición empírica (condición relativa):
# κ_num = |Δr / r| / |deltab / b| -> |(b/r)(dr/db)| 
# cuando deltab tiende a 0 aproxima y queda
# = κ_analítico.

if b != 0 and r1 != 0 and dscr(a, b, c) != 0:#cuando b, r1, D distintos a 0
    h = np.sqrt(eps_maq) #perturbación relativa pequeña
    bp = b * (1 + h) #se perturba b como b*(1+h),  bp es b perturbado
    db = bp - b  # perturbación aplicada, puede ser ligeramente distinto a b*h
    r1p, _ = rts(a, bp, c) #raíz con b perturbado, r1 recalculado
    kappa_num = abs((r1p - r1) / r1) / abs(db / b) # k_num = |(r1p - r1)/r1| / |db/b| = ... = k_analítico valor obtenido de la perturbación numerica
    print(f'\nVerificación numérica (perturbando b): kappa_b ≈ {kappa_num:.6e}')
    if np.isfinite(k) and k != 0:  # comprobación k distinto de 0
        err = abs(kappa_num - k) / k  #error relativo, queda en valor absoluto
        print(f'Error relativo vs fórmula analítica: {err:.2e}') #imprime el error en la terminal 
else:
    print('\nVerificación numérica no aplica (b=0, r1=0 o D=0).') #casos no verificables cuando k analítico es 0 (b=0), indefinido porque divide por 0 (r1=0) o k=infinito (D=0)
 