# Preparación para repositorio privado

Este repositorio fue acondicionado para ejecutarse desde un checkout privado sin depender de archivos servidos desde `raw.githubusercontent.com`.

## Cambios incluidos

- El punto de entrada `app_emergenciacombinado.py` aplica el adaptador `private_runtime.py` antes de ejecutar el motor científico.
- Los datos meteorológicos, pesos neuronales, modelo de clústeres, fondo y logotipo se cargan desde archivos locales del checkout.
- Si falta un activo científico, la aplicación se detiene y muestra el archivo faltante. No se generan pesos o modelos aleatorios.
- La actualización meteorológica mediante GitHub Actions continúa utilizando el token interno del repositorio.
- Se incorpora una verificación manual del despliegue privado.

## Antes de cambiar la visibilidad

1. Fusionar la rama o pull request de preparación privada en `main`.
2. En Streamlit Community Cloud, autorizar a Streamlit para acceder a los repositorios privados de la cuenta `PREDWEEM`.
3. Confirmar que la aplicación esté configurada con:
   - repositorio: `PREDWEEM/LOLIUM-PERGA2026`;
   - rama: `main`;
   - archivo principal: `app_emergenciacombinado.py`.
4. Revisar que GitHub Actions esté habilitado.
5. Conservar, cuando correspondan, los secretos opcionales:
   - `SIGA_PARAMS_JSON`;
   - `SIGA_POST_DATA_JSON`;
   - `SIGA_HEADERS_JSON`.
6. Ejecutar manualmente el workflow **Verificar despliegue privado**.
7. Ejecutar manualmente **Actualizar SIGA Pergamino y ECMWF ENS** y comprobar que finalice correctamente.

## Archivos obligatorios

La aplicación necesita los siguientes recursos en la raíz del checkout:

- `app_emergenciacombinado.py`;
- `app_emergenciacombinado_core.py`;
- `private_runtime.py`;
- `IW.npy`;
- `LW.npy`;
- `bias_IW.npy`;
- `bias_out.npy`;
- `modelo_clusters_k3.pkl`;
- `meteo_daily.csv`;
- `fondo_predweem_v3.png`;
- `logo_predweem.svg`.

Los datos de validación de campo pueden cargarse manualmente desde la interfaz y no son obligatorios para iniciar la aplicación.

## Cambio de visibilidad

Cuando las verificaciones anteriores sean satisfactorias:

1. Abrir **Settings** del repositorio.
2. Ingresar en **General**.
3. Buscar **Danger Zone**.
4. Seleccionar **Change repository visibility**.
5. Cambiar de `Public` a `Private` y confirmar el nombre del repositorio.

El cambio de visibilidad no modifica la URL del repositorio, pero Streamlit dejará de acceder a él si la autorización para repositorios privados no está activa.

## Verificación posterior

Después de privatizar:

1. Reiniciar o redeployar la aplicación en Streamlit.
2. Confirmar que se vea el logotipo local y que el fondo cargue correctamente.
3. Comprobar que `meteo_daily.csv` sea leído desde el checkout.
4. Verificar la carga de `IW.npy`, `LW.npy`, `bias_IW.npy`, `bias_out.npy` y `modelo_clusters_k3.pkl`.
5. Revisar que la descarga Excel continúe funcionando.
6. Ejecutar nuevamente los dos workflows manuales.
7. Comprobar que la actualización meteorológica pueda hacer commit y push sobre `main`.

## Seguridad

- No guardar claves, tokens o credenciales dentro del código.
- Utilizar GitHub Secrets o Streamlit Secrets para datos sensibles.
- No entregar permisos de administración a colaboradores que solo necesiten lectura.
- Revisar periódicamente los accesos al repositorio y a Streamlit Community Cloud.
- Mantener el repositorio `mis-apps` público únicamente como portal de acceso; el código científico y los activos del modelo permanecen en este repositorio privado.

## Reversión

Ante una falla posterior al cambio de visibilidad:

1. No eliminar la aplicación de Streamlit.
2. Revisar primero la autorización de acceso a repositorios privados.
3. Ejecutar el workflow de verificación.
4. Consultar los logs de Streamlit para identificar archivos faltantes.
5. Volver temporalmente a público solo como último recurso y después de revisar que no existan activos que deban permanecer reservados.
