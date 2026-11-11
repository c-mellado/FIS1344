# Catalina Mellado Valdés
# 21.773.395-2
# FIS1344 Métodos Numéricos y Probabilidades
# 10/11/2026
# Tarea 7.9

import numpy as np
from scipy import integrate

# cuadratura gauss de 5 puntos (vista en clases)
# aproxima la int de a hasta b de f(x)dx = sum_i c_i f(x_i), 
# con x_i los ceros del polinomio de Legendre (de orden 5) y c_i los coeficientes asociados
# exacta para polinomios de grado menor o igual a 9 

def gauss_quad_5(f, a, b):  

    def x(t):
        return ((b-a)*t + a + b)/2 #nodos t: desde [-1, 1] al intervalo [a, b]

    c = [0.2369268850, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850] #coef.
    r = [0.9061798459, 0.5384693101, 0.0000000000, -0.5384693101, -0.9061798459] #nodos

    return ((b-a)/2)*sum(c[i]*f(x(r[i])) for i in range(5))

#cuadratura 5 puntos en 2d (gauss en y para cada uno de los nodos x)
def gauss_quad_5_2d(f, a, b, c, d):

    cg = [0.2369268850, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268850] 
    rg = [0.9061798459, 0.5384693101, 0.0000000000, -0.5384693101, -0.9061798459]

    h1 = (b-a)/2
    h2 = (a+b)/2
    result = 0

    for i in range(5):
        x = h1*rg[i] + h2
        c1 = c(x)
        d1 = d(x)
        k1 = (d1 - c1)/2
        k2 = (d1 + c1)/2
        result +=  cg[i]*k1*sum(cg[j]*f(x, k1*rg[j] + k2) for j in range(5))

    return h1*result

# para no trabajar con límites infinitos, hacemos cambio de variable: x= 1/u, y=1/v, dx = -du/(u^2) y el integrando queda como arctan(1/(uv))/(sqrt(u) + u^2)
# dada la pista: arctan(1/(uv)) = pi/2 - arctan(uv), entonces la int: I = (pi/2)*A - B con:
# A = int de 0 a 1 del integrando: 1/(sqrt(u) + u^2) du     (que contiene la divergencia)
# B = int de 0 a 1 del integrando:  du  int desde 0 hasta 1 de arctan(uv)/(sqrt(u) + u^2) dv   (NO diverge)

# A es la int (de 0 a 1) g(u)/sqrt(u) du con g(u) = 1/(1 + u^(3/2))
# usando taylor para la parte que se integra analíticamente:
# se aproxima g(u), usando el polinomio de primer orden en u=0
# P1(u) = g(0) + g'(0)*u = 1  [g(0)=1 y g'(0)=0]


def resto_A(u):  #resto (g - P1)/sqrt(u)
    return -u/(1 + u**(3/2))

A = 2 + gauss_quad_5(resto_A, 0, 1) #el 2 sale de que int(de 0 a 1) u^(-1/2) du = 2

# sustitución en B: u = s^2 para suavizar el integrando que tiene raíz (sqrt) cerca de u=0:
# B = int (de 0 a 1) ds int (de 0 a 1) 2*arctan((s^2)*v)/(1+s^3) dv

def integrando_B(s,v):
    return 2*np.arctan(s**2*v)/(1+s**3)

B = gauss_quad_5_2d(integrando_B, 0, 1, lambda s:0, lambda s: 1)

I =(np.pi/2)*A - B

# integral usando scipy en el dom. transformado [0, 1] 
# para ver error en la cuadratura de 5 puntos haciendo la diferencia con gauss
A_ref = 2 + integrate.quad(resto_A, 0, 1)[0]

def integral_v(s):
    return integrate.quad(lambda v: integrando_B(s, v), 0, 1)[0]
B_ref = integrate.quad(integral_v, 0, 1)[0]

I_ref = (np.pi/2)*A_ref - B_ref

print("Integral usando Gauss (con la divergencia aislada):", I)
print("Integral usando scipy quad (en el mismo dominio transformado):", I_ref)
print("El error absoluto es:", abs(I - I_ref))
print("El error relativo es:", abs(I - I_ref)/abs(I_ref))



# Nota: 7.0
# Base: 1/1; funcionamiento: 4/4; comentarios: 1/1; claridad: 1/1.
