# PREDWEEM — Lolium Pergamino 2026

Repositorio correspondiente a la implementación de **PREDWEEM** para la predicción de la emergencia y la dinámica fenológica de *Lolium multiflorum* en Pergamino, provincia de Buenos Aires, Argentina.

> **Propiedad intelectual**  
> Copyright © 2026 Guillermo R. Chantre / PREDWEEM.  
> Todos los derechos reservados.
>
> Este repositorio constituye software propietario. Su disponibilidad pública no concede autorización para utilizar, copiar, modificar, redistribuir, sublicenciar, realizar ingeniería inversa ni explotar comercialmente el código, los modelos, los parámetros, los pesos neuronales, la documentación o los datos incluidos.
>
> Consulte el aviso completo en [COPYRIGHT.md](COPYRIGHT.md).

## Finalidad

PREDWEEM es una herramienta de apoyo a la toma de decisiones agronómicas basada en la integración de datos meteorológicos, modelos predictivos y filtros ecofisiológicos para anticipar los flujos de emergencia de raigrás anual.

La implementación de este repositorio está orientada a **Pergamino** y debe utilizarse considerando el dominio geográfico, climático y agronómico para el cual fue configurada, así como su estado específico de validación.

## Preparación para repositorio privado

La aplicación fue acondicionada para ejecutarse desde un checkout privado:

- los datos, modelos y recursos visuales se cargan localmente;
- no se depende de archivos públicos servidos desde `raw.githubusercontent.com`;
- la ausencia de pesos o modelos reales detiene la aplicación, evitando simulaciones con activos aleatorios;
- GitHub Actions puede continuar actualizando `meteo_daily.csv` dentro del repositorio privado;
- Streamlit Community Cloud debe estar autorizado para acceder a los repositorios privados de `PREDWEEM`.

El procedimiento completo se encuentra en [PRIVATE_REPOSITORY.md](PRIVATE_REPOSITORY.md).

### Archivo principal de Streamlit

- rama: `main`;
- archivo: `app_emergenciacombinado.py`.

### Recursos científicos requeridos

- `IW.npy`;
- `LW.npy`;
- `bias_IW.npy`;
- `bias_out.npy`;
- `modelo_clusters_k3.pkl`;
- `meteo_daily.csv`.

## Automatización meteorológica

El workflow **Actualizar SIGA Pergamino y ECMWF ENS** construye una serie diaria continua mediante la siguiente jerarquía:

1. **SIGA–INTA Pergamino (`A872814`)** como fuente observada prioritaria.
2. **ECMWF IFS histórico** como reemplazo provisional de cualquier fecha vencida sin una observación SIGA completa y válida.
3. **ECMWF IFS ENS 0,25°** para hoy y los próximos seis días, utilizando P50 como valor operativo de Tmax, Tmin, Tmedia y precipitación.

Las observaciones SIGA con Tmax y Tmin válidas pero sin Tmedia conservan su condición observada y utilizan `TMEDIA = (TMAX + TMIN) / 2`. En cambio, una precipitación ausente nunca se interpreta como 0 mm: la fila incompleta se excluye del tramo observado y su fecha pasa al puente provisional ECMWF.

El ensamble exige 24 horas válidas por miembro y día, empareja temperatura y precipitación mediante el identificador real del miembro y requiere al menos 30 miembros válidos y el 80 % de los miembros disponibles. Las medias, P10, P50 y P90 se conservan para auditoría.

Antes de guardar `meteo_daily.csv`, GitHub Actions verifica continuidad diaria, ausencia de nulos críticos, coherencia física, ubicación temporal de observaciones/provisionales/pronósticos, correspondencia exacta de las variables operativas con P50 y cantidad mínima de miembros.

Antes y después de privatizar debe ejecutarse manualmente el workflow **Verificar despliegue privado**.

### Fin de carga de cada campaña

La última fecha meteorológica de cada campaña es el **1 de octubre, inclusive** (en 2026, el 1/10/2026). El 1 de enero siguiente comienza la campaña nueva (`_campania_vigente()` en `actualizar_meteo_pergamino.py`): mientras SIGA no publica días del año, el puente provisional ECMWF los cubre, y la caché SIGA se limita a la campaña vigente.
El pronóstico de siete días se recorta al alcanzar ese límite, tanto en la
serie operativa como en los nuevos archivos de pronóstico. Desde el 2 de
octubre no se consulta el ensamble; las actualizaciones de SIGA y del puente
provisional quedan limitadas al 1 de octubre, permitiendo incorporar la
observación de ese día cuando se publique y reemplazar datos provisionales.
La validación admite una serie cerrada sin pronóstico y rechaza fechas
posteriores al cierre. Los archivos históricos existentes se conservan.

## Umbral de termoinhibición por defecto: 26 °C

El valor por defecto pasa de 20 °C (interfaz) y 24 °C (funciones del motor) a **26 °C**, en línea con los repositorios de Tres Arroyos, Lartigau, Azul y Bordenave.

Evidencia (motor de este repositorio, cobertura 80 %, Wmax 18,81 mm, lag 15 días, choque hídrico 45 mm; TVD = masa de emergencia mal ubicada, menor es mejor):

| Umbral | Pergamino 2024 (curva Red DIMA digitalizada) | Pergamino 2026 (conteos propios, desde el 01/04) |
|---|---|---|
| 20 °C (anterior, interfaz) | 74,4 % | 58,2 % |
| 24 °C | 41,3 % | 47,8 % |
| **26 °C** | **30,5 %** | **47,8 %** |
| 28 °C | 28,5 % | 44,2 % |

Limitaciones: son dos campañas. La de 2024 es una curva relativa suavizada leída de un gráfico (Boletín #3 de Red DIMA, AAPPCE; incertidumbre de lectura ≈ ±3 pp) y no hay réplicas; la de 2026 es una sola serie sin réplicas. Los umbrales de 28 °C o más mejoran algo más pero equivalen a casi anular la regla y no se adoptan sin más campañas. El ajuste no corrige la emergencia de mayo–junio 2026 (ocurrida con lluvia casi nula), por lo que el modelo no debe considerarse validado para Pergamino. El lag fijo de 15 días no se modificó: mejora 2026 y empeora 2024.

## Alerta de flushes

La página muestra la **alerta de flushes** de *Lolium multiflorum*: la probabilidad de que en los próximos 7 días emerja
≥ 5 % de la campaña, con dos niveles (*Atención* desde 15 %, *Alerta* desde 25 %), la emergencia acumulada esperada y la
ventana de control post-emergente (600–800 °Cd). El modelo (`alerta_lolmu.py`, `alerta_lolmu_params.json`,
`visualizacion_alerta_lolmu.py`) es el calibrado con 17 campañas de Bordenave; combina el calendario medio de LOLMU, la
respuesta a las lluvias registradas y, opcionalmente, la fecha de las primeras plántulas y conteos.

**Inicio tardío y variable.** En Pergamino la emergencia empezó a fines de febrero en 2024, en abril en 2026 y a mediados
de abril en 2023. Las lluvias de febrero–marzo no distinguen esos años y el primer pico del motor PREDWEEM adelanta el inicio
(05/03/2026). Por eso: (1) el reloj térmico y la ventana de control se calculan solo desde la fecha de primeras plántulas que
informa el usuario (o el primer conteo con plántulas); (2) mientras no se informen plántulas, la alerta llega como máximo a
*Atención* (`tope_sin_inicio`; la probabilidad no cambia). (3) Las semanas que terminaron antes del último conteo en cero sin
plántulas previas (el conteo confirma que no hubo emergencia) quedan por debajo de *Atención* (14 %; `tope_hasta_ultimo_cero`,
la probabilidad por clima queda en el Excel) y la curva acumulada arranca después de ese conteo (`curva_desde_ultimo_cero`).
La semana que sigue al último conteo en cero se evalúa normalmente, porque el primer flush suele venir justo después
(Pergamino 2026: cero el 01/04, flush al 13/04; en las 7 localidades, 37 % de las semanas siguientes a un conteo en cero
tuvieron flush).

**Validación** (2024 y 2026 con clima SIGA de INTA Pergamino; 30 intervalos, incluidas las semanas de febrero–marzo de 2026 sin
emergencia; 11 con flush; `data/campanias/`):

| Información disponible | AUC | Flushes en *Alerta* (con tope) | Falsas *Alerta* con tope / sin tope |
|---|---|---|---|
| Solo clima | 0,67 | 0 (9/11 en *Atención*) | 0/19 / 7/19 |
| + primeras plántulas | 0,74 | 7/11 (11/11 en *Atención* o *Alerta*) | 2/19 / 9/19 |
| + conteos | 0,81 | — | — |

Antes del inicio la alerta responde a las lluvias y no puede saber si la emergencia ya empezó: en febrero–marzo de 2026 daba
*Alerta* casi todas las semanas. Con las mismas semanas, la red neuronal del gemelo dio AUC 0,66. Un calendario propio armado con
las otras campañas de Pergamino empeoró la alerta (los años difieren mucho): la alerta y la curva acumulada usan el calendario de
Bordenave (error de la curva: 8 puntos en 2024, 27 en 2026). Límites: dos campañas evaluables; 2023 y 2024 son curvas relativas
quincenales digitalizadas de la Red DIMA; el clima SIGA de 2023 tiene huecos en abril–agosto.

La barra lateral conserva solo los parámetros que usa la alerta (temperaturas base, óptima y crítica, y objetivos 600 y
800 °Cd); los del motor quedan en sus valores por defecto (cobertura 80 %, termoinhibición 26 °C).

## Aviso de emergencia tardía (ventana de siembra)

Del 1 de mayo al 31 de agosto, debajo de la alerta de flushes, la app muestra la **probabilidad de emergencia tardía**:
que en los próximos 7 días emerja al menos 2 % de la campaña, con aviso desde 20 %. En esa época los flushes de ≥ 5 %
son raros, pero pocas plantas por semana importan para decidir la fecha de siembra. Es la misma alerta recalibrada por
nivel de información (`tardia` en `alerta_lolmu_params.json`); también aparece en el gráfico (línea punteada violeta) y
en la descarga en Excel.

| Información | AUC Bordenave (17 campañas, 196 semanas, 50 con ≥ 2 %) | Detectadas con el aviso | Semanas marcadas |
|---|---|---|---|
| Solo clima | 0,69 | 72 % | 52 % |
| + primeras plántulas | 0,78 | 82 % | 49 % |
| + conteos semanales | 0,84 | 74 % | 39 % |

Cada campaña se predijo con el ajuste sin ella. En las otras siete localidades (115 intervalos de mayo a agosto, 12 con
≥ 2 % por semana) la AUC fue 0,89 con primeras plántulas y 0,91 con conteos, con 92 % y 83 % de detección; allí la
probabilidad tiende a ser alta (19 % en promedio contra 10 % observado), por lo que hay más falsas alarmas.

## Condiciones de uso

No se concede licencia de uso por el solo hecho de acceder al repositorio. Cualquier utilización académica, técnica, institucional o comercial que exceda la visualización del contenido requiere autorización previa y escrita del titular de los derechos correspondientes.

Las solicitudes de autorización deben canalizarse mediante los medios de contacto del titular del repositorio PREDWEEM.

## Limitación de responsabilidad

PREDWEEM es una herramienta de soporte para decisiones y no sustituye el diagnóstico profesional, el monitoreo a campo ni la evaluación agronómica específica de cada lote. Las decisiones de manejo deben ser adoptadas por profesionales responsables considerando las condiciones locales y la normativa aplicable.

## Autoría

**PREDWEEM by Guillermo R. Chantre**
