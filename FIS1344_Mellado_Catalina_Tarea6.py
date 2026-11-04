# Catalina Mellado Valdés
# 21.773.395-2
# FIS1344 Métodos Numéricos y Probabilidades
# 04/11/2026
# Tarea 6.6 
# Use cinco puntos equidistantes para encontrar un polinomio interpolante de la función f(x) = sinh(x)
# entre x = -5 y x = 5. Grafique el polinomio. Estime el error cometido para cada x, usando la fórmula
# de error de los polinomios de Legendre. Luego grafique |f(x) - P(x)| y compare con el error estimado.

import numpy as np
import matplotlib.pyplot as plt
from math import factorial

# función de interpolación (vista en clases)
# Se construye con P(x) = sum_i [f(xi)Ln,i(x)]
def lagrange(x, puntos): 
    '''
    Interpola una función que pasa por n puntos como un polinomio
    de Legendre y lo evalúa en el punto x. Los puntos deben ser
    dados como una lista de tuplas (xi, f(xi)).
    '''
    Lis = np.ones(len(puntos))   # se acumulan los L_{n,i}(x) para cada i
    fs = np.ones(len(puntos))   #se guardan los f(x_i) para cada uno de los nodos
    for i in range (len(puntos)):
        fs[i] = puntos[i][1]
        for j in range(len(puntos)):
            if j != i:   #se excluye del producto j=i
                Lis[i] *=(x - puntos[j][0])/(puntos[i][0] - puntos[j][0])
    return sum(fs*Lis)   # P(x) = sum f(x_i)*L_{n,i}(x)

# datos del problema
f = np.sinh  # función a interpolar
a, b = -5, 5  # extremos del intervalo
n_puntos = 5  # núm. de nodos de interpolación
n = n_puntos - 1  # grado del polinomio -> 4, determinado por n + 1 puntos

# nodos (equidistantes)
nodos = np.linspace(a, b, n_puntos)

# lista de tuplas
puntos = [(xi, f(xi)) for xi in nodos]


# puntos donde se evalúa f y P
x = np.linspace(a, b, 401)  # 400 es múltiplo de 4, con un paso de 0.025 los nodos caen en el grid

P = np.array([lagrange(xi, puntos) for xi in x]) #se evalúa P en cada elemento x

# gráfica del polinomio interpolado
plt.figure(figsize=(8, 6)) 
plt.plot(x, f(x),    #curva de la función a interpolar
        label="sinh(x)",
        linewidth = 1.5,
        alpha=0.5,
        color="maroon")  
plt.plot(x, P,         #curva del polinomio interpolante
        linestyle="--", linewidth=1.5,
        label="P(x) (Lagrange, 5 puntos)",
        color ="cornflowerblue",
        dashes=[8,5])
plt.scatter(nodos, f(nodos), #puntos de interpolación
            color="mediumorchid",
            zorder=3,
            label="Nodos") 
plt.xlabel("x")
plt.ylabel("y")
plt.xticks(nodos)
plt.grid(alpha=0.3)
plt.legend()
plt.title("Interpolación de sinh(x)")
plt.figtext(0.5, 0.03,
            r"$\it{Nota.}$ Gráfica de la función "
            r"$f(x)=\sinh(x)$ y su polinomio interpolante.",
            ha="center", fontsize=9)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()

# error estimado
# de acuerdo a la fórmula de error de interpolación para cada x existe un xi en (a,b), tq.:
# f(x) - P(x) = f^(n+1)(xi) / (n+1)!*(x-x0)(x-x1)...(x-xn)
# las derivadas de sinh(x):
# sinh' = cosh | sinh'' = sinh | sinh''' = cosh | sinh^(4) = sinh | sinh^(5) = cosh
# como n+1 = 5 entonces f^(5) = cosh
# se acota |f^(5)(xi)| por su máximo en [a,b], además el cosh es par y crece con |x| entonces máx. en los extremos -> cosh(5)
M = np.cosh(max(abs(a), abs(b)))  

# producto (x-x0)(x-x1)...(x-xn)
prod_nodos = np.array([np.prod(xi - nodos) for xi in x])

# cota superior del error en cada x 
e_estimado = M/factorial(n + 1)*np.abs(prod_nodos)

# error real (diferencia entre la función y el polinomio, en valor absoluto)
e_real = np.abs(f(x) - P)

# comparación entre los errores y gráfica
plt.figure(figsize=(8, 6))
plt.plot(x, e_real, #gráfica del error real
        label="|f(x) - P(x)|",
        linewidth=1.5,
        color="maroon")
plt.plot(x, e_estimado, #gráfica del error estimado (cota superior)
        linestyle="--", linewidth=1.5,
        label="Cota estimada con la fórmula",
        color="cornflowerblue")
plt.scatter(nodos, np.zeros(n_puntos),
            color="mediumorchid",
            zorder=3,
            label="Nodos")
plt.xlabel("x")
plt.ylabel("error")
plt.xticks(nodos)
plt.grid(alpha=0.3)
plt.legend()
plt.title("Error real vs error estimado")
plt.figtext(0.5, 0.03,
            r"$\it{Nota.}$ En los nodos el error real y el estimado valen 0, " 
            r"$P(x) = f(x)$."
            "\n La cota superior queda por sobre el error real, ya que usa el máximo de la quinta derivada para todo el intervalo.",
            ha = "center", fontsize=9)
plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.show()

# errores máximos (real y estimado)
print("Error real máximo:", e_real.max(), "en x =", x[e_real.argmax()])
print("Error estimado máximo:", e_estimado.max(), "en x =", x[e_estimado.argmax()])

# cantidad de veces en que la cota supera al error real para el peor caso (razón entre valores máx.)
print("máx. cota / máx. error real:", e_estimado.max()/e_real.max()) 

# Nota: 7.0
# Base: 1/1; funcionamiento: 4/4; comentarios: 1/1; claridad: 1/1.
# 
