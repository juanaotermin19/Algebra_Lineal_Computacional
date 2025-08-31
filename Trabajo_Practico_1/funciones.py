import numpy as np
import pandas as pd
import scipy.linalg
import matplotlib.pyplot as plt

#Para poder leer el archivo hay que estar parado en la carpeta que contenga la entrega
#Leemos el archivo y lo convertimos a csv para poder trabajar con el
data = pd.read_excel('matrizlatina2011.xlsx', sheet_name=1)

data.to_csv('matrizlatina2011.csv',index=False)

#%% PUNTO 2
def busqueda_no_nulo(matriz,columna): 
    """
    Función para buscar el índice de la primera fila que tiene un 
    valor no nulo en una columna especificada
    
    Parámetros
    ----------
    A : matriz R^(nxn)
    Columna: índice de la columna donde se busca el pivote no nulo

    Returns
    -------
    idx: índice donde se encuentra el primer pivote no nulo
    
    """
    n= matriz.shape[0]  #calculamos la dimension de las filas de la matriz
    idx = 0                      #partimos desde el indice 0
    for i in range(0,n):           #iteramos sobre las filas hasta encontrar el pivote no nulo
        if matriz[i,columna] != 0 :
            idx = i                    # una vez que lo encontramos cambiamos el valor del idx por la fila donde se encuentra el no nulo
            return idx

def calcular_LU(A):
   """
   Función que calcula la factorización LU de una matriz PA
   
   Parámetros
   ----------
   A : matriz R^(nxn)

   Returns
   -------
   L : matriz triangular inferior de R^(nxn) con 1´s en la diagonal
   U : matriz triangular superior de R^(nxn)
   P: matriz de permutacion de R^(nxn)
   """
   m = A.shape[0] #Toma la dimension de las filas de A
   n = A.shape[1]  #Toma la dimension de las columnas de A
   Ac = A.astype(float)  # Convierte a float para evitar problemas de tipo
   P = np.eye(m)  #Crea P igual a la identidad con las dimensiones mxm
   iteracion = 0
    
   #si la matriz no es cuadrada no se puede realizar la descomposicion
   if m != n:                         
       print('Matriz no cuadrada')
       return
    
   # Bucle principal que realiza la factorización LU con pivotación parcial
   while iteracion < n:
       # Se obtiene el valor diagonal de la matriz en la posición actual
       diag = Ac[iteracion, iteracion]
        
       # Si el valor diagonal es 0, se necesita hacer un intercambio de filas para evitar dividir por cero
       if diag== 0:
           # Si la fila siguiente tiene un valor no nulo en la misma columna, se intercambian las filas
           if Ac[iteracion+1, iteracion] !=0:
               # Se intercambian las filas correspondientes en la matriz de permutación P y en la matriz Ac
                P[[iteracion, iteracion+1]] = P[[iteracion+1, iteracion]]
                Ac[[iteracion, iteracion+1]] = Ac[[iteracion+1, iteracion]]
           else:
               # Si la siguiente fila también tiene un 0, se busca la primera fila más abajo que tenga un valor no nulo en esa columna
               idx= busqueda_no_nulo(Ac, iteracion)
               # Se intercambian las filas correspondientes en la matriz de permutación P y en la matriz Ac
               P[[iteracion, idx]] = P[[idx, iteracion]]
               Ac[[iteracion, idx]] = Ac[[idx, iteracion]]
                
       # Se actualiza el valor diagonal de la matriz en la posición actual (después del intercambio, si ocurrió)
       diag = Ac[iteracion, iteracion]
       # Se divide el bloque inferior de la columna actual entre el valor diagonal, para generar los coeficientes de L
       Ac[iteracion+1:, iteracion] /= diag
        
       # Se actualizan las filas restantes para formar la matriz U, restando el producto de la columna inferior de L con la fila correspondiente de U
       Ac[iteracion+1:, iteracion+1:] -= np.outer(Ac[iteracion+1:, iteracion], Ac[iteracion, iteracion+1:])
        
       iteracion+=1
    
   # Se extrae la matriz L, tomando la parte inferior de Ac (excluyendo la diagonal) y agregando 1s en la diagonal 
   L = np.tril(Ac, -1) + np.eye(A.shape[0]) 
   # Se extrae la matriz U, tomando la parte superior de Ac
   U = np.triu(Ac)

   # Se redondean las entradas de L y U a 3 decimales para mejorar la legibilidad
   L = np.round(L,decimals=3)
   U= np.round(U, decimals=3)
    
   return L, U, P


def inversa_LU(L,U,P): 
    """
    Función que calcula la inversa de una matriz partiendo de su descomposicion
    PA = LU
    
    Parámetros
    ----------
      L : matriz triangular inferior de R^(nxn) con 1´s en la diagonal
      U : matriz triangular superior de R^(nxn)
      P: matriz de permutacion de R^(nxn)
    Returns
    -------
    A^-1: matriz inversa de A de R^(nxn) 
    
    """
    # Se obtiene la dimensión de la matriz L (y también de U y P, ya que todas son nxn)
    n = L.shape[0]
    # Se crea la matriz identidad de tamaño nxn, que servirá para calcular la inversa usando métodos de sistemas lineales
    Id = np.eye(n)
    # Se inicializan la matriz A_inv como matriz de ceros
    A_inv = np.zeros((n, n))

    for i in range(n):
        # Resolver L y = P @ e_i
        e_i_permutado = P @ Id[:, i]
        y = scipy.linalg.solve_triangular(L, e_i_permutado, lower=True)
        # Resolver U x = y
        x = scipy.linalg.solve_triangular(U, y)
        A_inv[:, i] = x

    return A_inv
