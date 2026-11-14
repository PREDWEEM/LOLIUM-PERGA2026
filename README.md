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
plántulas previas (el conteo confirma que no hubo emergencia) quedan en 0 % (`tope_hasta_ultimo_cero`,
la probabilidad por clima queda en el Excel) y la curva acumulada arranca a mitad de camino entre ese conteo y el primer conteo con plántulas (`curva_desde_ultimo_cero`).
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

## Curva acumulada en tiempo térmico desde las primeras plántulas

Cuando se informa la fecha de las primeras plántulas (o hay un primer conteo con plántulas), la curva de emergencia
acumulada deja de seguir el calendario y sigue el **tiempo térmico acumulado desde esa fecha** (media y rango 10–90 % de
17 campañas de Bordenave, `curva_tt` en `alerta_lolmu_params.json`); más allá del clima disponible se suman temperaturas
medias históricas. Error medio de la emergencia acumulada en las fechas de conteo desde las primeras plántulas (puntos %):

| Localidad | Calendario por fecha | Tiempo térmico desde el inicio |
|---|---|---|
| Bordenave (17 campañas, dejando cada una afuera) | 12,9 | 11,3 |
| Lartigau 2026 | 9,9 | 5,8 |
| Olavarría 2026 | 13,7 | 10,7 |
| Pergamino 2024 y 2026 | 17,5 | 7,2 |
| San Pedro 2023, 2024 y 2026 | 19,5 | 14,4 |

Mejora sobre todo donde la fecha de inicio varía mucho entre años. En Tres Arroyos y Balcarce sigue el calendario propio
del sitio (error 6,1 y 6,9 puntos) y en Azul el calendario de Bordenave (5,3 contra 15,8 en su única campaña).

## Señal de lluvia (actualización de noviembre de 2026)

La señal de lluvia usa ahora el **modelo de pulsos ajustado con las 17 campañas de *Lolium multiflorum* de Bordenave**.
El tamaño de cada pulso depende del agua que entró al suelo, de cuán seco estaba y de la época del año (máximo para
lluvias de fines de mayo); la mitad de la respuesta emerge dentro de 114 °Cd de tiempo térmico de suelo y el 90 %
dentro de 252 °Cd. Con el nuevo núcleo se reajustaron los modelos logísticos, los percentiles de referencia y la
recalibración de flujos tardíos (`alerta_lolmu_params.json`). El balance de agua y la temperatura de suelo no cambian.

Validación con cada campaña de Bordenave predicha con el ajuste sin ella (incluido el núcleo de lluvia): AUC 0,77 solo
con clima, 0,83 con primeras plántulas y 0,87 con conteos. En las ocho localidades de validación, sin reajustar: 0,89,
0,92 y 0,93 (calendario solo, 0,89). En Pergamino: 0,71, 0,81 y 0,82 (antes 0,63, 0,71 y 0,78); las tablas de
validación de este sitio que figuran más arriba corresponden a la versión anterior.

## Aviso de flujos tardíos

Del 1 de mayo al 30 de junio la app muestra además la **probabilidad de flujos tardíos**: que en los próximos 7 días
emerja al menos 2 % de la campaña. El aviso se da desde 20 %; sin datos del lote, desde 25 % (`tardia.umbral_tramos`).
En esa época los grandes flujos (≥ 5 %) son raros, pero pocas plantas por semana importan para decidir la fecha de siembra.
Es la misma alerta recalibrada por nivel de información (`tardia` en `alerta_lolmu_params.json`). Aparece en un recuadro
debajo de la alerta y en la descarga en Excel.

| Información | AUC Bordenave (17 campañas, 130 semanas de mayo–junio, 42 con ≥ 2 %) | Detectadas con el aviso | Semanas marcadas |
|---|---|---|---|
| Solo clima (aviso desde 25 %) | 0,57 | 67 % | 68 % |
| + primeras plántulas | 0,79 | 90 % | 62 % |
| + conteos semanales | 0,87 | 86 % | 51 % |

Cada campaña se predijo con el ajuste sin ella. Sin datos del lote el aviso casi no distingue semanas: conviene cargar la
fecha de primeras plántulas. En otras seis localidades (58 intervalos de mayo–junio, 10 con ≥ 2 % por semana) la AUC fue
0,85 con primeras plántulas y 0,89 con conteos, con 90 % de detección; allí la probabilidad tiende a ser alta (30 % contra
17 % observado).

**Del 1 de julio al 1 de octubre no se muestra probabilidad** (`tardia.recorrida`): en Bordenave (80 semanas, 11 con
flujo tardío) no distinguió las semanas con emergencia (AUC 0,41–0,50), y bajar el umbral solo marcaba más semanas al
azar. En su lugar, la app recomienda **recorrer los lotes cada 2 semanas**, sobre todo donde suele haber emergencia
invernal (sudoeste bonaerense), antes de definir la fecha de siembra o un control.

En las apps con dos o más campañas propias (Bordenave, Tres Arroyos, Balcarce, San Pedro y Pergamino), el recuadro de
flujos tardíos muestra además cuánto emergió **después de la fecha de emisión en cada campaña del sitio**. Si en esas
campañas, en mediana, emergió menos de 3 % de la campaña desde esa fecha, el aviso se presenta como «Flujo tardío posible
según el modelo; en <sitio> la emergencia suele cortarse antes»: la recalibración de tardíos se ajustó con Bordenave, donde
la emergencia se extiende hasta el invierno, y en sitios donde se corta antes avisa de más (en Balcarce, en mayo de 2014,
2025 y 2026, avisó 8 semanas y hubo 2 flujos tardíos). La probabilidad se sigue mostrando.

## Residual a la siembra (guía de riesgo)

Debajo de la alerta, un recuadro siempre visible muestra, para la **fecha de siembra prevista** (1/05–31/07; si no se
carga, el 1/06 como referencia), el riesgo de nacimientos en la **ventana de acción del residual: de 10 días antes a 45 días
después de la siembra**, el período de acción promedio de un herbicida residual. Tiene tres tarjetas: riesgo, nacimientos en
la ventana (mediana de las campañas, en plantas/m² y en % de la campaña) y recomendación. Se calcula con las 2 campañas de Pergamino (`residual` en `alerta_lolmu_params.json`). Las plantas/m² usan la densidad de cada campaña con conteos; las campañas que solo
tienen la curva relativa usan la densidad de referencia del sitio (1719 pl/m² por campaña, mediana de las
campañas con conteos):

| Siembra | Nacimientos de −10 a +45 días (mediana) | Riesgo para el residual | Campañas del sitio |
|---|---|---|---|
| 1/05 | 917 pl/m² (37,8 %) | Muy alto | 2024: 44 pl/m² · 2026: 1790 pl/m² |
| 15/05 | 864 pl/m² (29,9 %) | Muy alto | 2024: 15 pl/m² · 2026: 1712 pl/m² |
| 1/06 | 672 pl/m² (21,3 %) | Muy alto | 2024: 2 pl/m² · 2026: 1342 pl/m² |
| 15/06 | 339 pl/m² (10,5 %) | Muy alto | 2024: 0 pl/m² · 2026: 677 pl/m² |
| 1/07 | 86 pl/m² (2,7 %) | Muy alto | 2024: 0 pl/m² · 2026: 173 pl/m² |

El nivel de riesgo surge de las plantas/m² que nacieron en la ventana en una campaña típica (mediana). Se usan plantas
y no el % de la campaña porque en sitios con mucha densidad una fracción chica son muchas plantas (en San Pedro, 0,8 % de
la campaña fueron 29–42 pl/m²):

| Riesgo | Nacimientos en la ventana | Recomendación |
|---|---|---|
| 🔴 Muy alto | ≥ 50 pl/m² | Usar residual |
| 🟠 Alto | 20–50 pl/m² | Residual recomendado |
| 🟡 Medio | 5–20 pl/m² | Según infestación: residual si el lote tiene alta infestación o antecedentes de raigrás |
| 🟢 Bajo | < 5 pl/m² | Residual solo en lotes muy infestados |

Sin otro dato, las densidades son las registradas en los ensayos del sitio. El usuario puede indicar la **infestación de raigrás de su lote** (opcional): baja (unas 200 pl/m² en toda la campaña), media (unas 1.000) o alta (unas 4.000). Con ese dato las plantas/m² se calculan con la densidad del lote y, como la infestación ya está considerada, las recomendaciones de los niveles medio y bajo pasan a «residual opcional» y «no hace falta residual».

Debajo de las tarjetas se indica el valor de cada campaña del sitio y qué proporción de las campañas cayó en cada nivel. Es
un **riesgo histórico, no una predicción de la campaña**: ni la fecha de primeras plántulas ni el clima anticiparon cuánta
emergencia queda después de la siembra.

En el gráfico, la segunda fila del **semáforo semanal** («Residual (siembra)») muestra, para cada semana del 1/05 al
31/07, el riesgo para el residual si se siembra el primer día de esa semana: rojo, alto o muy alto (≥ 20 pl/m²); amarillo, medio
(5–20 pl/m²); verde, bajo (< 5 pl/m²), con la infestación del lote si se indicó. Con fecha de siembra cargada, las semanas empiezan ese día, así la celda
de la siembra coincide con el recuadro del residual. La fecha de siembra prevista se marca
con una línea verde y una flecha con cartel.

## Condiciones de uso

No se concede licencia de uso por el solo hecho de acceder al repositorio. Cualquier utilización académica, técnica, institucional o comercial que exceda la visualización del contenido requiere autorización previa y escrita del titular de los derechos correspondientes.

Las solicitudes de autorización deben canalizarse mediante los medios de contacto del titular del repositorio PREDWEEM.

## Limitación de responsabilidad

PREDWEEM es una herramienta de soporte para decisiones y no sustituye el diagnóstico profesional, el monitoreo a campo ni la evaluación agronómica específica de cada lote. Las decisiones de manejo deben ser adoptadas por profesionales responsables considerando las condiciones locales y la normativa aplicable.

## Autoría

**PREDWEEM by Guillermo R. Chantre**

## Módulo común (LOLIUM-COMUN)

La alerta, su visualización y el control de acceso son comunes a las 8 apps de Lolium y están en el paquete
[PREDWEEM/LOLIUM-COMUN](https://github.com/PREDWEEM/LOLIUM-COMUN), fijado por versión en `requirements.txt`
(`lolium-comun @ git+https://github.com/PREDWEEM/LOLIUM-COMUN@<commit>  # v1.3.1`). En esta app, `alerta_lolmu.py`,
`visualizacion_alerta_lolmu.py` y `acceso_predweem.py` son archivos puente que cargan ese paquete con los parámetros de este
sitio (`alerta_lolmu_params.json`), que siguen acá. Los cambios a la alerta se hacen en LOLIUM-COMUN y llegan a la app al
actualizar la versión en `requirements.txt`.
