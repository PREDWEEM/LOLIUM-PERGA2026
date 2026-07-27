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

## Condiciones de uso

No se concede licencia de uso por el solo hecho de acceder al repositorio. Cualquier utilización académica, técnica, institucional o comercial que exceda la visualización del contenido requiere autorización previa y escrita del titular de los derechos correspondientes.

Las solicitudes de autorización deben canalizarse mediante los medios de contacto del titular del repositorio PREDWEEM.

## Limitación de responsabilidad

PREDWEEM es una herramienta de soporte para decisiones y no sustituye el diagnóstico profesional, el monitoreo a campo ni la evaluación agronómica específica de cada lote. Las decisiones de manejo deben ser adoptadas por profesionales responsables considerando las condiciones locales y la normativa aplicable.

## Autoría

**PREDWEEM by Guillermo R. Chantre**
