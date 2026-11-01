# Campañas de Pergamino: emergencia y clima

Son las campañas de *Lolium* de Pergamino publicadas en el gemelo digital (`PREDWEEM/PERGAMINO_DIGITALTWEEN`), con que se
validó la alerta. La actualización automática no modifica estas carpetas. En Pergamino el inicio de la emergencia es tardío y
variable: fines de febrero en 2024, abril en 2026 y mayo en 2023.

| Campaña | Emergencia | Clima |
|---|---|---|
| 2023 | `emergencia_pergamino_2023.csv`: curva relativa **quincenal** digitalizada del Reporte 4 de AAPPCE / Red DIMA (valores relativos, no plantas/m²); emergencia del 16/04 al 30/06. Procedencia en `procedencia_pergamino_2023.json`. | No disponible. |
| 2024 | `emergencia_pergamino_2024.csv`: curva relativa quincenal digitalizada del Boletín 3 de la Red DIMA; emergencia del 16/02 al 31/05. Procedencia en `procedencia_pergamino_2024.json`. | `clima_pergamino_2024.csv`: SIGA A872814 / INTA Pergamino, 366 días. |
| 2026 | `emergencia_pergamino_2026_original.xlsx` (planilla original, igual a `VALIDA.xlsx`) y `emergencia_pergamino_2026.csv`: 10 conteos del 13/04 al 03/07/2026 (plantas/m² por intervalo); el 01/04 es un cero convencional (según el responsable, no hubo emergencia en febrero ni marzo). | `clima_pergamino_2026.csv`: copia de `meteo_daily.csv` al cierre (274 días; SIGA A872814 y días provisionales ECMWF). |

En los CSV de emergencia: `PLM2` (valor del intervalo que termina en esa fecha), días desde el conteo anterior, fracción de cada
intervalo y emergencia acumulada (% del total registrado).

## Huellas SHA-256

| Archivo | SHA-256 |
|---|---|
| `2023/emergencia_pergamino_2023.csv` | `cfd0b1b5ec6f8d124df7b67d69ba91b9c9cc475ff59bd5701029fae65913d8e3` |
| `2023/procedencia_pergamino_2023.json` | `64084204494561cfdfc097f41db2b725267dcb161d7540b73631d8064a29557a` |
| `2024/clima_pergamino_2024.csv` | `a0841886a6e4758384ba333cd9abe64267a6c3187c0b73899dd02947559c091e` |
| `2024/emergencia_pergamino_2024.csv` | `215b7ec15f242bc0754c8dfa3a22d10e1ab77efdb094a36307250ea4ef7caa19` |
| `2024/procedencia_pergamino_2024.json` | `e73b9b558da22e2fe6df523ed9f7e59655efc738352875c344ccf3459d149d95` |
| `2026/clima_pergamino_2026.csv` | `a3cc7f944c04e848313235dd5ee504a1573134e15db88025a1364e5aa066724c` |
| `2026/emergencia_pergamino_2026.csv` | `2f1c5883d10860f9528647c2afc9d67553ef755b643c388e5331a6ea207245a8` |
| `2026/emergencia_pergamino_2026_original.xlsx` | `7e36cb92140d1a6a09c4b14f41bbfd6a0859a3e431bd36232b941c864395e0ac` |
