# ==========================================================
# 📚 Importação de bibliotecas necessárias para a página Normalização
# ==========================================================
import streamlit as st
import pandas as pd
import sqlite3
import altair as alt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# ==========================================================
# ⚖️ Função principal para exibir a página "Normalização"
# ==========================================================
def show():
    # ------------------------------------------------------
    # 💾 Carregar dados base (prioriza EDA limpa; fallback: SQLite)
    # ------------------------------------------------------
    @st.cache_data(ttl=600)
    def load_from_sqlite(db_path="preco_petroleo.db", table="preco_petroleo_raw"):
        with sqlite3.connect(db_path) as conn:
            df_ = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["data"])
        df_["data"] = pd.to_datetime(df_["data"])
        df_ = df_.sort_values("data").reset_index(drop=True)
        return df_

    if "eda_df_clean" in st.session_state and not st.session_state["eda_df_clean"].empty:
        df_base = st.session_state["eda_df_clean"].copy()
    else:
        df_base = load_from_sqlite()

    st.subheader("📐 Normalização / Escalonamento")

    # ------------------------------------------------------
    # 🛠️ Engenharia de atributos
    # ------------------------------------------------------
    df = df_base.copy()
    # Retornos e lags
    df["Retorno"] = df["preco_petroleo"].pct_change()
    df["Preço t-1"] = df["preco_petroleo"].shift(1)
    df["Preço t-5"] = df["preco_petroleo"].shift(5)
    # Médias móveis
    df["Média Móvel 7 dias"] = df["preco_petroleo"].rolling(7).mean()
    df["Média Móvel 30 dias"] = df["preco_petroleo"].rolling(30).mean()
    # Volatilidade (desvio padrão dos retornos)
    df["Volatividade 7 dias"] = df["Retorno"].rolling(7).std()
    df["Volatividade 30 dias"] = df["Retorno"].rolling(30).std()

    # Remover linhas iniciais com NaN (por lags/rollings)
    df = df.dropna().reset_index(drop=True)

    # ------------------------------------------------------
    # 🧱 Seletor de colunas e tipo de escalonador
    # ------------------------------------------------------
    st.markdown("### ⚙️ Configurações de normalização")
    numeric_cols = [
        "preco_petroleo", "Retorno", "Preço t-1", "Preço t-5",
        "Média Móvel 7 dias", "Média Móvel 30 dias", "Volatividade 7 dias", "Volatividade 30 dias"
    ]

    cols_escolhidas = st.multiselect(
        "Selecione as colunas a normalizar",
        options=numeric_cols,
        default=["preco_petroleo", "Retorno", "Média Móvel 7 dias", "Média Móvel 30 dias"]
    )

    scaler_tipo = st.radio(
        "Escolha o escalonador",
        options=["StandardScaler (z-score)", "MinMaxScaler [0,1]", "RobustScaler (mediana/IQR)"],
        index=0,
        horizontal=True
    )

    # Parâmetros extras para MinMax (opcional)
    min_val, max_val = 0.0, 1.0
    if "MinMaxScaler" in scaler_tipo:
        col_a, col_b = st.columns(2)
        with col_a:
            min_val = st.number_input("Valor mínimo (MinMax)", value=0.0, step=0.1)
        with col_b:
            max_val = st.number_input("Valor máximo (MinMax)", value=1.0, step=0.1)
        if max_val <= min_val:
            st.warning("O valor máximo deve ser maior que o mínimo para o MinMaxScaler.")

    # ------------------------------------------------------
    # ▶️ Aplicar normalização nas colunas selecionadas
    # ------------------------------------------------------
    def build_scaler(name):
        if name.startswith("StandardScaler"):
            return StandardScaler()
        if name.startswith("MinMaxScaler"):
            return MinMaxScaler(feature_range=(min_val, max_val))
        if name.startswith("RobustScaler"):
            return RobustScaler()
        return StandardScaler()

    aplicar = st.button("⚖️ Aplicar normalização")

    if aplicar:
        if not cols_escolhidas:
            st.warning("Selecione ao menos uma coluna numérica para normalizar.")
            st.stop()

        scaler = build_scaler(scaler_tipo)
        df_norm = df.copy()

        try:
            # Ajuste-transformação somente nas colunas selecionadas
            df_norm[cols_escolhidas] = scaler.fit_transform(df_norm[cols_escolhidas])
        except Exception as e:
            st.error(f"Erro ao normalizar: {e}")
            st.stop()

        # Guardar no estado (opcional para uso posterior)
        st.session_state["normalizado_df"] = df_norm
        st.success("✅ Normalização aplicada com sucesso.")

        # ------------------------------------------------------
        # 👀 Visualização: comparação Preço Petróleo - Original vs Normalizado
        # ------------------------------------------------------
        st.markdown(f"### 👁️ Preço do Petróleo: Original vs Normalizado · Método: {scaler_tipo}")

        comp = pd.DataFrame({
            "data": df_norm["data"],
            "Preço (original)": df["preco_petroleo"],
            "Preço (normalizado)": df_norm["preco_petroleo"] if "preco_petroleo" in cols_escolhidas else df["preco_petroleo"]
        })

        import altair as alt

        # Linha para preço original (escala da esquerda)
        line_orig = alt.Chart(comp).mark_line(color="blue").encode(
            x=alt.X("data:T", title="", axis=alt.Axis(format="%b.%y")),
            y=alt.Y("Preço (original):Q", title="Preço original (USD)"),
            tooltip=[alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
                    alt.Tooltip("Preço (original):Q", format=".2f")]
        )

        # Linha para preço normalizado (escala da direita)
        line_norm = alt.Chart(comp).mark_line(color="orange").encode(
            x="data:T",
            y=alt.Y("Preço (normalizado):Q", title="Preço normalizado", axis=alt.Axis(titleColor="orange")),
            tooltip=[alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
                    alt.Tooltip("Preço (normalizado):Q", format=".4f")]
        )

        # Combinar com duas escalas
        chart = alt.layer(line_orig, line_norm).resolve_scale(
            y='independent'  # permite eixos Y separados
        ).properties(height=340)

        st.altair_chart(chart, use_container_width=True)


        # ------------------------------------------------------
        # 🔽 Visualização tabular e download
        # ------------------------------------------------------
        st.markdown("### 🔽 Prévia do dataset normalizado")
        preview = df_norm.copy()
        preview["Data"] = preview["data"].dt.strftime("%d/%m/%Y")
        preview["Preço Petróleo"] = preview["preco_petroleo"]
        #preview = preview.rename(columns={"preco_petroleo": "Preço Petróleo"})
           
        cols_show = ["Data", "Preço Petróleo"] + [c for c in df_norm.columns if c != "data" and c != "preco_petroleo"]
        st.dataframe(preview[cols_show].head(20), use_container_width=True)

        csv_norm = df_norm.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Baixar dataset normalizado (CSV)",
            data=csv_norm,
            file_name="preco_petroleo_normalizado.csv",
            mime="text/csv",
            key="download_csv_norm"
        )

    else:
        # ------------------------------------------------------
        # 🧾 Prévia do dataset antes da normalização
        # ------------------------------------------------------
        st.markdown("### 🧾 Amostra antes da normalização")
        preview = df.copy()
        preview["Data"] = preview["data"].dt.strftime("%d/%m/%Y")
        cols_show = ["Data"] + [c for c in df.columns if c != "data"]
        st.dataframe(preview[cols_show].head(20), use_container_width=True)
