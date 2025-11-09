# ==========================================================
# Importação de bibliotecas necessárias
# ==========================================================
import sqlite3
from typing import List

import pandas as pd
import streamlit as st
import altair as alt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ==========================================================
# Constantes
# ==========================================================
DB_PATH    = "preco_petroleo.db"
TABLE_NAME = "preco_petroleo_raw"

# ==========================================================
# Utilitários
# ==========================================================
@st.cache_data(ttl=600)
def load_from_sqlite(db_path: str = DB_PATH, table: str = TABLE_NAME) -> pd.DataFrame:
    """Lê a tabela do SQLite, garante colunas esperadas e ordena por data."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["data"])
    if {"data", "preco_petroleo"} - set(df.columns):
        raise RuntimeError("Colunas esperadas 'data' e 'preco_petroleo' não encontradas.")
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data", "preco_petroleo"]).sort_values("data").reset_index(drop=True)
    return df

def engineer_features(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Cria retornos, lags, médias móveis e volatilidades.
    Remove linhas com NaN (oriundas de lags/rollings).
    """
    df = df_base.copy()
    # Retornos e lags
    df["Retorno"]  = df["preco_petroleo"].pct_change()
    df["Preço t-1"] = df["preco_petroleo"].shift(1)
    df["Preço t-5"] = df["preco_petroleo"].shift(5)
    # Médias móveis
    df["Média Móvel 7 dias"]  = df["preco_petroleo"].rolling(7).mean()
    df["Média Móvel 30 dias"] = df["preco_petroleo"].rolling(30).mean()
    # Volatilidade (desvio padrão dos retornos)
    df["Volatilidade 7 dias"]  = df["Retorno"].rolling(7).std()
    df["Volatilidade 30 dias"] = df["Retorno"].rolling(30).std()
    # Limpa
    return df.dropna().reset_index(drop=True)

def available_numeric_cols(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    """Retorna apenas as colunas de 'candidates' que existem no df."""
    return [c for c in candidates if c in df.columns]

def build_scaler(name: str, min_val: float = 0.0, max_val: float = 1.0):
    """Retorna o scaler apropriado conforme seleção do usuário."""
    if name.startswith("StandardScaler"):
        return StandardScaler()
    if name.startswith("MinMaxScaler"):
        return MinMaxScaler(feature_range=(min_val, max_val))
    if name.startswith("RobustScaler"):
        return RobustScaler()
    return StandardScaler()

def df_preview_format(df: pd.DataFrame) -> pd.DataFrame:
    """Formata Data para exibição na UI sem alterar o df original."""
    out = df.copy()
    out["Data"] = out["data"].dt.strftime("%d/%m/%Y")
    return out

# ==========================================================
# Página "Normalização"
# ==========================================================
def show():
    st.subheader("📐 Normalização / Escalonamento")

    # --- Fonte de dados: EDA limpa (se existir) ou SQLite ---
    try:
        df_base = (
            st.session_state["eda_df_clean"].copy()
            if "eda_df_clean" in st.session_state and not st.session_state["eda_df_clean"].empty
            else load_from_sqlite()
        )
    except Exception as e:
        st.error(f"Erro ao carregar base: {e}")
        st.stop()

    if df_base.empty:
        st.info("Base vazia. Atualize os dados na página **Dados/EDA** e volte aqui.")
        st.stop()

    # --- Engenharia de atributos ---
    df = engineer_features(df_base)
    if df.empty:
        st.info("Após lags e médias móveis, não há linhas suficientes para exibir.")
        st.stop()

    # --- Configurações de normalização ---
    st.markdown("### ⚙️ Configurações de normalização")
    candidate_numeric = [
        "preco_petroleo", "Retorno", "Preço t-1", "Preço t-5",
        "Média Móvel 7 dias", "Média Móvel 30 dias", "Volatilidade 7 dias", "Volatilidade 30 dias"
    ]
    numeric_cols = available_numeric_cols(df, candidate_numeric)

    default_cols = [c for c in ["preco_petroleo", "Retorno", "Média Móvel 7 dias", "Média Móvel 30 dias"] if c in numeric_cols]
    cols_escolhidas = st.multiselect(
        "Selecione as colunas a normalizar",
        options=numeric_cols,
        default=default_cols,
    )

    scaler_tipo = st.radio(
        "Escolha o escalonador",
        options=["StandardScaler (z-score)", "MinMaxScaler [0,1]", "RobustScaler (mediana/IQR)"],
        index=0,
        horizontal=True,
    )

    # Parâmetros extras para MinMax (opcionais e validados)
    min_val, max_val = 0.0, 1.0
    if "MinMaxScaler" in scaler_tipo:
        col_a, col_b = st.columns(2)
        with col_a:
            min_val = st.number_input("Valor mínimo (MinMax)", value=0.0, step=0.1)
        with col_b:
            max_val = st.number_input("Valor máximo (MinMax)", value=1.0, step=0.1)
        if max_val <= min_val:
            st.warning("O valor máximo deve ser maior que o mínimo para o MinMaxScaler.")

    # --- Aplicar normalização ---
    aplicar = st.button("⚖️ Aplicar normalização", use_container_width=True)

    if aplicar:
        if not cols_escolhidas:
            st.warning("Selecione ao menos uma coluna numérica para normalizar.")
            st.stop()

        scaler = build_scaler(scaler_tipo, min_val, max_val)
        df_norm = df.copy()

        try:
            df_norm[cols_escolhidas] = scaler.fit_transform(df_norm[cols_escolhidas])
        except Exception as e:
            st.error(f"Erro ao normalizar: {e}")
            st.stop()

        # Guarda no estado para uso em outras páginas
        st.session_state["normalizado_df"] = df_norm

        st.success("✅ Normalização aplicada com sucesso.")

        # --- Gráfico: Original vs Normalizado (Preço do Petróleo) ---
        st.markdown(f"### 👁️ Preço do Petróleo: Original vs Normalizado · Método: {scaler_tipo}")

        comp = pd.DataFrame({
            "data": df_norm["data"],
            "Preço (original)": df["preco_petroleo"],
            "Preço (normalizado)": df_norm["preco_petroleo"] if "preco_petroleo" in cols_escolhidas else df["preco_petroleo"],
        })

        # Proteção de domínio (evita flatten em séries quase constantes)
        vmin = float(comp[["Preço (original)", "Preço (normalizado)"]].min().min())
        vmax = float(comp[["Preço (original)", "Preço (normalizado)"]].max().max())
        pad  = 0.1 * (vmax - vmin) if vmax > vmin else 1.0
        domain_left  = [vmin - pad, vmax + pad]

        line_orig = alt.Chart(comp).mark_line().encode(
            x=alt.X("data:T", title="", axis=alt.Axis(format="%b.%y")),
            y=alt.Y("Preço (original):Q", title="Preço original (USD)", scale=alt.Scale(domain=domain_left)),
            tooltip=[alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
                     alt.Tooltip("Preço (original):Q", format=".2f")],
        )

        line_norm = alt.Chart(comp).mark_line().encode(
            x="data:T",
            y=alt.Y("Preço (normalizado):Q", title="Preço normalizado"),
            tooltip=[alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
                     alt.Tooltip("Preço (normalizado):Q", format=".4f")],
        )

        chart = alt.layer(line_orig, line_norm).resolve_scale(y="independent").properties(height=340)
        st.altair_chart(chart, use_container_width=True)

        # --- Prévia tabular e download ---
        st.markdown("### 🔽 Prévia do dataset normalizado")
        preview = df_preview_format(df_norm)
        preview["Preço Petróleo"] = df_norm["preco_petroleo"]
        cols_show = ["Data", "Preço Petróleo"] + [c for c in df_norm.columns if c not in {"data", "preco_petroleo"}]
        st.dataframe(preview[cols_show].head(20), use_container_width=True)

        csv_norm = df_norm.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Baixar dataset normalizado (CSV)",
            data=csv_norm,
            file_name="preco_petroleo_normalizado.csv",
            mime="text/csv",
            key="download_csv_norm",
            use_container_width=True,
        )

    else:
        # --- Amostra antes da normalização ---
        st.markdown("### 🧾 Amostra antes da normalização")
        preview = df_preview_format(df)
        cols_show = ["Data"] + [c for c in df.columns if c != "data"]
        st.dataframe(preview[cols_show].head(20), use_container_width=True)
