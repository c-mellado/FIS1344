# Catalina Mellado Valdés
# 21.773.395-3
# FIS1344 Métodos Numericos y Probabilidades
# Tarea 1.8
# Martes 08/09/2026

# Grafique en una misma figura la siguiente función junto 
# con su serie de Taylor alrededor de x = 0
# truncada al término número 10. 
# ¿Hasta cuál valor de x parece ser una buena aproximación?
# f(x) = x sin(x)
# Use colores y estilos de línea diferentes para cada línea.
# Escoja un rango de valores de que represente lo que quiera 
# concluir sobre esta aproximación. 
# Incluya una leyenda para facilitar la lectura del gráfico.


#librerías
import numpy as np
import matplotlib.pyplot as plt
import math


#función
x = np.linspace(-10, 10, 1000)#preparar los datos:1000 puntos desde x=-10 hasta x=10
f = x * np.sin(x) # la función por si sola

#serie de taylor
N = 10  #cantidad de términos en la serie
p = 0  
for n in range(N):
   p = p + (-1)**n * x**(2*n + 2) / math.factorial(2*n + 1)   #expresión obtenida multiplicando la serie de taylor de sin(x) por x

#gráfico
plt.figure(figsize=(10, 5)) #Ajuste del tamaño, no se leía la nota y visualizaba el gráfico correctamente
plt.plot(x, f,       #curva de la función por si sola
        label = "f(x)= x*sin(x)", 
        linewidth = 1.5, #grosor de la línea
        color ="maroon") 
plt.plot(x, p,     #curva de la serie de taylor
        label = f"Serie de Taylor (N = {N} términos)",
        linestyle ="--", linewidth = 1.5,
        color ="cornflowerblue",
        dashes=[8,5])
plt.legend()
plt.grid(alpha=0.2) #cuadrícula con alpha 0.2 de transparencia
plt.xlabel("x") #eje x
plt.ylabel("y") #eje y
plt.title("Función f(x) y aproximación mediante Serie de Taylor")
plt.xlim(-12, 12) # limite de eje x para visualizar mejor la diferencia entre la función y la aproximación
plt.ylim(-12, 12) # limite de eje y
plt.subplots_adjust(bottom=0.2) #espacio entre pie de gráfico y figura
plt.figtext (0.5, 0.03,  #pie del gráfico
            " $\it{Nota.}$ Comparación entre la función $f(x)=x\\sin(x)$ y su aproximación por Serie de Taylor con $N=10$ términos," 
            "\n se observa alrededor de x=0 como ambas graficas coinciden y se desvía en los extremos.",
            ha="center",
            va="center", #centrar texto
            fontsize =9)
plt.show()  
