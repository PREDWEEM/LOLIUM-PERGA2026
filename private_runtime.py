# -*- coding: utf-8 -*-
"""Adaptador de ejecución para desplegar PREDWEEM desde un repositorio privado.

El motor científico se conserva sin reescrituras masivas. Antes de ejecutarlo,
este módulo sustituye únicamente las dependencias que apuntaban al repositorio
público por recursos incluidos en el checkout privado y bloquea la creación de
modelos aleatorios cuando faltan activos científicos reales.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


REQUIRED_PRIVATE_ASSETS = (
    "IW.npy",
    "LW.npy",
    "bias_IW.npy",
    "bias_out.npy",
    "modelo_clusters_k3.pkl",
    "meteo_daily.csv",
    "fondo_predweem_v3.png",
    "logo_predweem.svg",
)


class PrivateRuntimeError(RuntimeError):
    """Indica que el motor no pudo adaptarse de forma segura al modo privado."""


def _replace_once(
    source: str,
    pattern: str,
    replacement: str,
    description: str,
    *,
    flags: int = 0,
) -> str:
    """Aplica una sustitución obligatoria y falla si el código esperado cambió."""
    updated, count = re.subn(
        pattern,
        lambda _match: replacement,
        source,
        count=1,
        flags=flags,
    )
    if count != 1:
        raise PrivateRuntimeError(
            f"No se pudo aplicar la adaptación privada: {description}. "
            "Revise si cambió app_emergenciacombinado_core.py."
        )
    return updated


def build_private_core_source(core_path: Path) -> str:
    """Devuelve el código del motor adaptado para trabajar solo con archivos locales."""
    source = core_path.read_text(encoding="utf-8")

    source = _replace_once(
        source,
        r'set_bg_hack\("fondo_predweem_v3\.png"\)',
        'set_bg_hack(str(BASE / "fondo_predweem_v3.png"))',
        "carga local del fondo",
    )

    source = _replace_once(
        source,
        r"def create_mock_files_if_missing\(\):.*?\ncreate_mock_files_if_missing\(\)",
        '''def validate_required_private_assets():
    """Detiene la app si faltan modelos o recursos del checkout privado."""
    required_assets = (
        "IW.npy",
        "LW.npy",
        "bias_IW.npy",
        "bias_out.npy",
        "modelo_clusters_k3.pkl",
        "meteo_daily.csv",
        "fondo_predweem_v3.png",
        "logo_predweem.svg",
    )
    missing_assets = [
        name for name in required_assets if not (BASE / name).is_file()
    ]
    if missing_assets:
        st.error(
            "Faltan recursos obligatorios del despliegue privado: "
            + ", ".join(missing_assets)
        )
        st.info(
            "Verifique el checkout de Streamlit y la autorización de acceso "
            "al repositorio privado. No se generarán modelos aleatorios."
        )
        st.stop()

validate_required_private_assets()''',
        "validación de activos científicos",
        flags=re.DOTALL,
    )

    source = _replace_once(
        source,
        r"^def load_data\(file_uploader, default_name\):"
        r".*?(?=^def |\Z)",
        '''def load_data(file_uploader, default_name):
    """Carga archivos aportados por el usuario o recursos del checkout privado."""
    if file_uploader:
        suffix = Path(file_uploader.name).suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(file_uploader)
        return pd.read_csv(file_uploader)

    local_candidates = (
        BASE / f"{default_name}.csv",
        BASE / f"{default_name}.xlsx",
        BASE / f"{default_name}.xls",
    )
    for candidate in local_candidates:
        if candidate.is_file():
            if candidate.suffix.lower() in {".xlsx", ".xls"}:
                return pd.read_excel(candidate)
            return pd.read_csv(candidate)

    st.warning(
        f"No se encontró un archivo local para '{default_name}'. "
        "Verifique que el recurso esté incluido en el checkout privado."
    )
    return None


''',
        "carga local de datos",
        flags=re.DOTALL | re.MULTILINE,
    )

    source = _replace_once(
        source,
        r"    with col_upload:\n.*?\n    with col_rastrojo:",
        '''    with col_upload:
        st.markdown("#### ☁️ Datos Climáticos")
        modo_meteo = st.radio(
            "Fuente de datos meteorológicos",
            options=["Automática", "Carga manual"],
            index=0,
            horizontal=True,
            help=(
                "Automática usa meteo_daily del repositorio privado. "
                "Carga manual permite usar un archivo CSV, XLSX o XLS."
            ),
        )

        archivo_meteo = None
        if modo_meteo == "Carga manual":
            archivo_meteo = st.file_uploader(
                "Cargar datos meteorológicos",
                type=["csv", "xlsx", "xls"],
                key="archivo_meteo_manual",
                help=(
                    "Columnas requeridas: FECHA o Fecha, TMAX, TMIN "
                    "y PREC o LLUVIA."
                ),
            )
            if archivo_meteo is None:
                st.warning(
                    "Seleccione un archivo meteorológico para ejecutar "
                    "la simulación en modo manual."
                )
            else:
                st.success(f"Archivo meteorológico cargado: {archivo_meteo.name}")
        else:
            st.info("🔄 Carga automática de clima activada.")

        st.markdown("#### 🌱 Datos de Validación")
        archivo_campo = st.file_uploader(
            "Opcional: Cargar archivo manual de Campo",
            type=["xlsx", "csv"],
        )

    with col_rastrojo:''',
        "selector de carga meteorológica",
        flags=re.DOTALL,
    )

    source = _replace_once(
        source,
        r'df_meteo_raw = load_data\(None, "meteo_daily"\)',
        '''if modo_meteo == "Carga manual":
    df_meteo_raw = (
        load_data(archivo_meteo, "meteo_daily")
        if archivo_meteo is not None
        else None
    )
else:
    df_meteo_raw = load_data(None, "meteo_daily")''',
        "selección de fuente meteorológica",
    )

    source = _replace_once(
        source,
        r'st\.sidebar\.image\("https://raw\.githubusercontent\.com/PREDWEEM/LOLIUM-PERGA2026/main/logo\.png", width="stretch"\)',
        'st.sidebar.image(str(BASE / "logo_predweem.svg"), width="stretch")',
        "carga local del logotipo",
    )

    forbidden_reference = "raw.githubusercontent.com/PREDWEEM/LOLIUM-PERGA2026"
    if forbidden_reference in source:
        raise PrivateRuntimeError(
            "Persisten referencias al repositorio público en el motor adaptado."
        )

    return source


def verify_private_checkout(base: Path) -> None:
    """Verifica recursos y sintaxis sin ejecutar Streamlit ni el modelo."""
    missing = [name for name in REQUIRED_PRIVATE_ASSETS if not (base / name).is_file()]
    if missing:
        raise PrivateRuntimeError(
            "Faltan recursos obligatorios: " + ", ".join(missing)
        )

    core_path = base / "app_emergenciacombinado_core.py"
    if not core_path.is_file():
        raise PrivateRuntimeError(
            "No se encontró app_emergenciacombinado_core.py en el checkout."
        )

    private_source = build_private_core_source(core_path)
    compile(private_source, str(core_path), "exec")


def main() -> int:
    base = Path(__file__).resolve().parent
    try:
        verify_private_checkout(base)
    except PrivateRuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("OK: el checkout está preparado para despliegue privado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
