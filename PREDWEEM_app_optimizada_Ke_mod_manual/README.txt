PREDWEEM — NUEVO app_emergenciacombinado.py
============================================

OBJETIVO
--------
Crear una copia nueva del modelo original con los parámetros óptimos del
13/07/2026, dejando solamente estos dos parámetros configurables a mano:

1. Coeficiente hídrico del suelo (Ke)
2. Modulador térmico del suelo

El archivo original NO se modifica.

UBICACIÓN
---------
Descomprima esta carpeta dentro de la raíz del repositorio LOLIUM-PERGA2026,
al mismo nivel que app_emergenciacombinado.py.

EJECUCIÓN EN WINDOWS
--------------------
Haga doble clic en:

    CREAR_NUEVO_MODELO.bat

El resultado se crea en:

    PREDWEEM_app_optimizada_Ke_mod_manual/
        modelo_optimizado_manual/
            app_emergenciacombinado.py

Luego puede revisar ese archivo y copiarlo a la raíz del repositorio cuando
desee sustituir el modelo operativo.

PARÁMETROS FIJOS OPTIMIZADOS
----------------------------
Wmax:                    17.514229108604354 mm
Humedad p50:              0.3273917872173965
Pendiente hídrica:       10.0
Corte hídrico:            0.0616661684226577
Recarga relativa:         0.6355770431092194
Latencia:                JD 28
Ventana térmica:          7 días
Termoinhibición:         26.025250899067515 °C
Ventana de lluvia:        3 días
Choque hídrico:          40.47186394276612 mm
Fin de choque:           JD 110
Techo de choque:          0.75
Primer pico:              0.7702092359274952
Persistencia:             1 día
Lag:                     +40 días

PARÁMETROS MANUALES
-------------------
Ke:
    rango 0.05–1.20
    valor inicial 0.25
    referencia del optimizador 0.10

Modulador térmico:
    rango 0.50–1.20
    valor inicial 0.85
    referencia del optimizador 0.85

NOTA METODOLÓGICA
-----------------
Los parámetros proceden de validación cruzada temporal interna con un único
set de campo. No constituyen una validación externa independiente.
