# Catalina Mellado Valdés
# 21.773.395-2
# FIS1344 Métodos Numéricos y Probabilidades
# 29/09/2026
# Tarea 3.4 

# Ejercicio sacado de “Basic Probability Theory”, R. B. Ash, Dover, 1970.

# En una cierta población, el 20% de las personas tiene una enfermedad. 
# Se hace un examen a los miembros de la población que tiene la característica
# de dar positivo el 90% de las veces si la persona tiene la enfermedad y el 30%
# de las veces si la persona no tiene la enfermedad (falso positivo). A los que 
# dan positivo se les suministra una droga que puede causar manchas rojas en la piel 
# en un 20% de los pacientes.

# Si una persona tiene esas manchas rojas. ¿Cuál es la probabilidad de que haya tenido la enfermedad?

# Los datos del problema 
sick = 0.2  # P(S) pacientes enfermos
p_sick = 0.9     # P(+ | S) probabilidad de positivo si está enfermo
p_healthy = 0.3   # P(+ | H) probabilidad de dar positivo si está sano(falso positivo)
spots_p = 0.2   # P(sp | +) probabilidad de tener manchas si es positivo


def prob_spots(s, p_s, p_h, sp_p):
    #s -> P(S) prob. de estar enfermo
    #p_s -> P(+|S) prob. de dar positivo si está enfermo
    #p_h -> P(+|H) prob. de dar positivo si está sano
    #sp_p -> P(sp| +) prob. de tener manchas si es positivo

    h = 1 - s    # pacientes sanos, la suma de los pacientes sanos (h) y enfermos (s) es igual a 100% de la población (1)

    #P(sp | S) y P(sp | H) probabilidad de tener manchas si está enfermo o sano
    sp_s = sp_p * p_s # P(sp|S) = P(sp|+)*P(+|S)
    sp_h = sp_p * p_h  # P(sp|H) = P(sp|+)*P(+|H)

    #P(sp) prob. de manchas obtenida por marginalización entre enfermos y sanos
    sp = sp_s*s + sp_h*h


    #usando Bayes P(S|sp) = P(sp|S)*P(S)/P(sp) para la prob. de estar enfermo si tiene manchas
    return sp_s*s / sp

#resultado
result = prob_spots(
    sick, p_sick, p_healthy, spots_p
)
print(f"P(S|sp) = {result:.4f}: la probabilidad de estar enfermo "
      f"si tiene manchas es aprox.: {result:.1%}")
#imprime el resultado con 4 decimales de P(S|sp) y en porcentaje con un decimal

# Nota: 7.0
# Base: 1/1; funcionamiento: 4/4; comentarios: 1/1; claridad: 1/1.
# El cálculo de la probabilidad condicionada mediante Bayes es correcto y todas sus etapas están comentadas.
