@echo off
setlocal
cd /d "%~dp0"

if not exist "..\app_emergenciacombinado.py" (
  echo ERROR: No se encontro ..\app_emergenciacombinado.py
  echo Coloque esta carpeta descomprimida dentro del repositorio LOLIUM-PERGA2026.
  pause
  exit /b 1
)

python scripts\crear_app_optimizada_manual.py ^
  "..\app_emergenciacombinado.py" ^
  --csv "data\parametros_optimos_2026-07-13.csv" ^
  --output "modelo_optimizado_manual\app_emergenciacombinado.py"

if errorlevel 1 (
  echo.
  echo No se pudo crear el modelo. Revise el mensaje anterior.
  pause
  exit /b 1
)

echo.
echo MODELO CREADO EN:
echo %CD%\modelo_optimizado_manual\app_emergenciacombinado.py
pause
