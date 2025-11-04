# streamlit_app_similarity_matrix.py
# -----------------------------------------------------------
# Aplicación Streamlit para generar matrices de co-ocurrencia
# y similitud a partir de resultados de Card Sorting abierto.
#
# Entrada esperada: varios ficheros Excel/CSV con 3 columnas:
# - participant_id  (identificador del participante)
# - card_title      (título de la tarjeta)
# - category        (categoría asignada por el participante)
#
# La app calcula, para cada par de tarjetas (i, j):
#   N_ij = nº de participantes que clasificaron ambas tarjetas
#   C_ij = nº de participantes que pusieron i y j en la misma categoría
#   S_ij = 100 * C_ij / N_ij  (si N_ij = 0, se deja vacío/NaN)
#
# Incluye: heatmap interactivo (Plotly), ordenación alfabética o por
# clúster (jerárquico), umbral de resaltado, y descargas (CSV/Excel).
# -----------------------------------------------------------

import io
import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import plotly.graph_objs as go

# -------------------------
# Utilidades
# -------------------------

REQUIRED_COLS_DEFAULT = {
    "participant_id": "participant_id",
    "card_title": "card_title",
    "category": "category",
}


def _read_single_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """Lee un Excel/CSV en DataFrame, intentando detectar formato por extensión."""
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".csv"):
        # Intento con separación por coma; si falla, reintenta con ;
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=';')
    else:
        raise ValueError("Formato no soportado. Usa .xlsx, .xls o .csv")


essential_tip = (
    """
    **Consejo:** En Card Sorting abierto, si un participante no clasificó
    alguna tarjeta, ese participante **no cuenta** para el denominador de ese par.
    Esta app sigue esa convención (N_ij se calcula sólo con quienes vieron ambas tarjetas).
    """
)


# -------------------------
# Cálculo de matrices
# -------------------------

def build_participant_index(df: pd.DataFrame, cols: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Devuelve un diccionario: participant -> {card_title: category}.
    Filtra filas con valores nulos en las 3 columnas esenciales.
    """
    p, c, g = cols["participant_id"], cols["card_title"], cols["category"]
    work = df[[p, c, g]].dropna()
    # Aseguramos tipos string para estabilidad
    work[p] = work[p].astype(str).str.strip()
    work[c] = work[c].astype(str).str.strip()
    work[g] = work[g].astype(str).str.strip()

    participants: Dict[str, Dict[str, str]] = {}
    for pid, card, cat in work[[p, c, g]].itertuples(index=False):
        participants.setdefault(pid, {})[card] = cat
    return participants


def compute_matrices(participants: Dict[str, Dict[str, str]],
                     card_order: List[str] | None = None
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calcula matrices de:
      - N (denominadores): nº de participantes que vieron ambos pares.
      - C (co-ocurrencia): nº de participantes que agruparon i y j igual.
      - S (similitud): 100 * C / N (NaN si N=0). Diagonal = 100.
    """
    # Universo de tarjetas
    all_cards = set()
    for cards in participants.values():
        all_cards.update(cards.keys())
    cards = sorted(all_cards) if card_order is None else card_order

    n = len(cards)
    idx = {card: i for i, card in enumerate(cards)}

    N = np.zeros((n, n), dtype=float)
    C = np.zeros((n, n), dtype=float)

    for pid, mapping in participants.items():
        present_cards = list(mapping.keys())
        # Denominadores: para cualquier par dentro del conjunto clasificado por el participante
        for a, b in itertools.combinations(present_cards, 2):
            i, j = idx[a], idx[b]
            N[i, j] += 1
            N[j, i] += 1
            if mapping[a] == mapping[b]:
                C[i, j] += 1
                C[j, i] += 1

    # Diagonales
    for i in range(n):
        # En diagonal, definimos N[i,i] = nº de participantes que vieron esa tarjeta
        seen_i = sum(1 for m in participants.values() if cards[i] in m)
        N[i, i] = seen_i
        C[i, i] = seen_i

    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.where(N > 0, (C / N) * 100.0, np.nan)
    np.fill_diagonal(S, 100.0)

    N_df = pd.DataFrame(N, index=cards, columns=cards).round(0).astype('Int64')
    C_df = pd.DataFrame(C, index=cards, columns=cards).round(0).astype('Int64')
    S_df = pd.DataFrame(S, index=cards, columns=cards).round(1)
    return N_df, C_df, S_df


def cluster_order_from_similarity(S_df: pd.DataFrame) -> List[str]:
    """Calcula orden de tarjetas mediante clustering jerárquico (average linkage) sobre distancia = 1 - S/100.
    Para valores NaN en S (pares no comparables), se sustituyen por 0 de similitud (distancia 1).
    """
    S = S_df.to_numpy().astype(float)
    # Sustituimos NaN por 0 de similitud (distancia máxima)
    S = np.nan_to_num(S, nan=0.0)
    # Aseguramos simetría
    S = (S + S.T) / 2.0
    np.fill_diagonal(S, 100.0)

    # Convertimos a distancias en forma condensada
    D = 1.0 - (S / 100.0)
    # Aseguramos rango [0,1]
    D = np.clip(D, 0.0, 1.0)
    # squareform espera la parte superior de la matriz de distancias
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method='average')
    order_idx = leaves_list(Z)
    ordered_cards = list(S_df.index[order_idx])
    return ordered_cards


# -------------------------
# Visualización
# -------------------------

def make_heatmap(S_df: pd.DataFrame,
                 C_df: pd.DataFrame,
                 N_df: pd.DataFrame,
                 title: str,
                 threshold: float = 50.0,
                 show_values: bool = False) -> go.Figure:
    cards = list(S_df.index)
    z = S_df.to_numpy().astype(float)
    text = []
    for i, row in enumerate(cards):
        text_row = []
        for j, col in enumerate(cards):
            s_val = z[i, j]
            c_val = C_df.iloc[i, j]
            n_val = N_df.iloc[i, j]
            if pd.isna(s_val):
                txt = f"{row} × {col}<br>Sin datos (N=0)"
            else:
                txt = (
                    f"{row} × {col}<br>Similitud: {s_val:.1f}%<br>"
                    f"Co-ocurrencias: {int(c_val) if pd.notna(c_val) else 0} / N={int(n_val) if pd.notna(n_val) else 0}"
                )
            text_row.append(txt)
        text.append(text_row)

    # Máscara para atenuar valores por debajo del umbral
    mask = np.where(pd.isna(z), 0, (z >= threshold).astype(float))
    z_display = np.where(pd.isna(z), None, z)

    fig = go.Figure(data=go.Heatmap(
        z=z_display,
        x=cards,
        y=cards,
        colorscale='Blues',
        zmin=0,
        zmax=100,
        hoverinfo='text',
        text=text,
        showscale=True,
        colorbar=dict(title="% similitud")
    ))

    # Atenuación con anotaciones de cuadrícula (opcional). Aquí usamos la opacidad por celda
    # imitando el umbral: por debajo del umbral, reducimos la opacidad.
    fig.data[0].hoverongaps = False

    # Plotly no permite opacidad por celda directamente en Heatmap; como aproximación,
    # añadimos un segundo heatmap binario semitransparente para resaltar por encima del umbral.
    highlight = np.where(pd.isna(z), None, mask)
    fig.add_trace(go.Heatmap(
        z=highlight,
        x=cards,
        y=cards,
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
        showscale=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=45, automargin=True),
        yaxis=dict(autorange='reversed'),
        margin=dict(l=80, r=40, t=60, b=120),
        height=800
    )

    if show_values:
        # Añadimos anotaciones con porcentaje (puede ser denso en matrices grandes)
        annotations = []
        for i, y in enumerate(cards):
            for j, x in enumerate(cards):
                val = S_df.iloc[i, j]
                if pd.isna(val):
                    continue
                annotations.append(dict(
                    x=x, y=y, text=f"{val:.0f}", showarrow=False, font=dict(size=10)
                ))
        fig.update_layout(annotations=annotations)

    return fig


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name)
            # Formato condicional para similitud si existe
            if name.lower().startswith("simil"):
                wb = writer.book
                ws = writer.sheets[name]
                # Rango de datos
                nrows, ncols = df.shape
                # Offset de encabezados (fila 0/col 0 reservadas a índices/columnas)
                rng = xl_range = xl = f"B2:{chr(65 + ncols)}{1 + nrows}"
                # Escala de color azul
                ws.conditional_format(rng, {
                    'type': '3_color_scale',
                    'min_color': '#f7fbff',  # azul muy claro
                    'mid_color': '#9ecae1',
                    'max_color': '#08306b'
                })
    return output.getvalue()


# -------------------------
# Interfaz Streamlit
# -------------------------

def main():
    st.set_page_config(page_title="Matriz de similitud – Card Sorting", layout="wide")
    st.title("Matriz de similitud (Card Sorting abierto)")
    st.caption("Carga tus resultados (Excel/CSV con 3 columnas: participant_id, card_title, category)")

    st.sidebar.header("Ajustes")
    threshold = st.sidebar.slider("Umbral de resaltado (%)", 0, 100, 50, step=5)
    order_opt = st.sidebar.radio("Orden de tarjetas", ["Alfabético", "Clustering jerárquico"], index=1)
    show_values = st.sidebar.checkbox("Mostrar valores en celdas", value=False)

    st.sidebar.markdown(essential_tip)

    uploaded_files = st.file_uploader(
        "Sube uno o varios ficheros (.xlsx, .xls, .csv)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("💡 Sube tus archivos para comenzar. Puedes combinar varios ficheros.")
        st.stop()

    # Lectura y concatenación
    frames = []
    for f in uploaded_files:
        try:
            frames.append(_read_single_file(f))
        except Exception as e:
            st.error(f"No se pudo leer {f.name}: {e}")
            st.stop()

    data = pd.concat(frames, ignore_index=True)

    # Mapeo de columnas (por si el usuario usa nombres distintos)
    st.subheader("Mapeo de columnas")
    cols = list(data.columns)

    c_part = st.selectbox("Columna de participante", options=cols, index=cols.index(REQUIRED_COLS_DEFAULT["participant_id"]) if REQUIRED_COLS_DEFAULT["participant_id"] in cols else 0)
    c_card = st.selectbox("Columna de tarjeta (card_title)", options=cols, index=cols.index(REQUIRED_COLS_DEFAULT["card_title"]) if REQUIRED_COLS_DEFAULT["card_title"] in cols else 0)
    c_cat  = st.selectbox("Columna de categoría", options=cols, index=cols.index(REQUIRED_COLS_DEFAULT["category"]) if REQUIRED_COLS_DEFAULT["category"] in cols else 0)

    cols_map = {
        "participant_id": c_part,
        "card_title": c_card,
        "category": c_cat,
    }

    participants = build_participant_index(data, cols_map)

    # Métricas básicas
    all_cards = sorted({card for m in participants.values() for card in m.keys()})
    num_participants = len(participants)
    num_cards = len(all_cards)

    m1, m2, m3 = st.columns(3)
    m1.metric("Participantes", f"{num_participants}")
    m2.metric("Tarjetas únicas", f"{num_cards}")
    # densidad aproximada de pares comparables
    # N_ij>0 / (n*(n-1)/2)

    N_df, C_df, S_df = compute_matrices(participants)

    # Ordenación
    if order_opt == "Alfabético":
        order_cards = all_cards
    else:
        order_cards = cluster_order_from_similarity(S_df)

    N_df = N_df.loc[order_cards, order_cards]
    C_df = C_df.loc[order_cards, order_cards]
    S_df = S_df.loc[order_cards, order_cards]

    st.subheader("Heatmap de similitud")
    fig = make_heatmap(S_df, C_df, N_df, title="Matriz de similitud (% de participantes que emparejaron cada par)", threshold=threshold, show_values=show_values)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Descargas")
    cA, cB, cC, cD = st.columns(4)
    # CSVs
    cA.download_button(
        label="📥 Similitud (%) – CSV",
        data=S_df.to_csv(index=True).encode('utf-8-sig'),
        file_name="similitud_matrix.csv",
        mime="text/csv",
    )
    cB.download_button(
        label="📥 Co-ocurrencias (C) – CSV",
        data=C_df.to_csv(index=True).encode('utf-8-sig'),
        file_name="coocurrencias_matrix.csv",
        mime="text/csv",
    )
    cC.download_button(
        label="📥 Denominadores (N) – CSV",
        data=N_df.to_csv(index=True).encode('utf-8-sig'),
        file_name="denominadores_matrix.csv",
        mime="text/csv",
    )

    # Excel con varias hojas
    excel_bytes = to_excel_bytes({
        "Similitud_%": S_df,
        "Coocurrencias_C": C_df,
        "Denominadores_N": N_df,
    })
    cD.download_button(
        label="📥 Excel (todas las matrices)",
        data=excel_bytes,
        file_name="matrices_card_sorting.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.caption(
        "La diagonal se fija en 100% (y C=N) por definición. "
        "Para pares sin participantes en común (N=0), se muestran como vacíos."
    )


if __name__ == "__main__":
    main()
