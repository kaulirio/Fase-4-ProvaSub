# ==========================================================
# 📚 Importação das bibliotecas necessárias
# ==========================================================
import pandas as pd
import streamlit as st
import sqlite3 # banco de dados - arquivo local
import numpy as np
import altair as alt

# ==========================================================
# ⚙️ Configurações iniciais da aplicação Streamlit
# ==========================================================
st.set_page_config(page_title="Modelo_Preditivo_Petroleo_ProvaSub", page_icon="🛢️", layout="wide")

# ==========================================================
# 🏷️ Cabeçalho inicial da aplicação
# ==========================================================
st.title("🛢️ Modelo Preditivo – Preço do Petróleo (USD)")
st.markdown("---")

# ==========================================================
# 💾 Inicialização de variáveis de sessão
# ==========================================================
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "extrair_dados" not in st.session_state:
    st.session_state["extrair_dados"] = False

# ==========================================================
# 📌 Menu de navegação (botões principais)
# ==========================================================
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📥 Carregar/Atualizar base de dados do IPEA"):
        st.session_state["page"] = "data"
with col2:
    if st.button("🔎 Análise Exploratória"):
        st.session_state["page"] = "eda"
with col3:
    if st.button("📊 Resultados com os Modelos Preditivos"):
        st.session_state["page"] = "results"

st.markdown("---")

# ==========================================================
# 📥 Página DATA – Carregar e atualizar base do IPEA
# ==========================================================
if st.session_state["page"] == "data":
    st.subheader("📥 Carregar/Atualizar base de dados do IPEA")

    # ----- Ação para atualizar a base -----
    if st.button("Atualizar base de dados do IPEA"):
        try:
            # Extração da tabela no site do IPEA
            url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"
            tables = pd.read_html(url)
            df = tables[2].copy()

            # Limpeza e padronização dos dados
            df.columns = ["data", "preco_petroleo"]
            df = df.drop(index=0).reset_index(drop=True)
            df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
            df["preco_petroleo"] = df["preco_petroleo"].astype(float) / 100

            # Salvando em banco SQLite
            with sqlite3.connect("preco_petroleo.db") as conn:
                df.to_sql("preco_petroleo_raw", conn, if_exists="replace", index=False)

            # Guardar no estado da sessão
            st.session_state["extrair_dados"] = True
            st.session_state["preco_petroleo_raw"] = df
            
            # Criar versão para exibição
            st.session_state["preco_petroleo_raw_display"] = df[["data", "preco_petroleo"]].rename(
                columns={"data": "Data", "preco_petroleo": "Preço Petróleo (USD)"}
            )
            st.session_state["preco_petroleo_raw_display"]["Data"] = (
                st.session_state["preco_petroleo_raw_display"]["Data"].dt.strftime("%d/%m/%Y")
            )

            st.success("✅ Base atualizada e salva em 'preco_petroleo.db'")
        except Exception as e:
            st.error(f"Erro: {e}")

    # ----- Exibir dados carregados -----
    if st.session_state["extrair_dados"]:
        df = st.session_state["preco_petroleo_raw"]

        csv = df.to_csv(index=False).encode("utf-8")
        if st.download_button(
            "💾 Baixar CSV",
            data=csv,
            file_name="preco_petroleo.csv",
            mime="text/csv",
            key="download_csv"
        ):
            st.session_state["page"] = "csv_download"

        st.dataframe(st.session_state["preco_petroleo_raw_display"])

# ==========================================================
# 💾 Página CSV_DOWNLOAD – Confirmação de download
# ==========================================================
elif st.session_state["page"] == "csv_download":
    st.subheader("💾 Download realizado")
    st.success("O arquivo CSV foi baixado com sucesso.")
    st.info("Agora você pode continuar para análise exploratória ou resultados.")
    if "preco_petroleo_raw" in st.session_state:
        st.dataframe(st.session_state["preco_petroleo_raw_display"].head())

# ==========================================================
# 🔎 Página EDA – Análise Exploratória
# ==========================================================
elif st.session_state["page"] == "eda":
    st.subheader("🔎 Análise Exploratória")

    # ----- Carregar dados do SQLite -----
    @st.cache_data(ttl=600)
    def load_from_sqlite(db_path="preco_petroleo.db", table="preco_petroleo_raw"):
        try:
            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["data"])
            df["data"] = pd.to_datetime(df["data"])
            df = df.sort_values("data").reset_index(drop=True)
            return df
        except Exception as e:
            raise RuntimeError(f"Erro ao ler {table} de {db_path}: {e}")

    try:
        df = load_from_sqlite()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # ----- Amostra de dados -----
    st.write("**Amostra de dados (últimas linhas):**")
    st.dataframe(df.tail(10), use_container_width=True)

    # ----- Filtro de período -----
    st.markdown("### ⏱️ Filtro de Período")
    ...

    # (mantém os comentários em cada seção já existente: checagens, estatísticas, gráficos, outliers, heatmap, normalização etc.)

    # ----- Salvar dataset limpo -----
    st.session_state["eda_df_clean"] = dff_no.copy()

# ==========================================================
# 📊 Página RESULTADOS – Treinamento e avaliação de modelos
# ==========================================================
elif st.session_state["page"] == "results":
    st.subheader("📊 Resultados com os Modelos Preditivos")

    # ----- Importações específicas de ML -----
    import numpy as np
    import altair as alt
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import (
        roc_auc_score, roc_curve, confusion_matrix, classification_report
    )

    # ----- Carregar dados -----
    ...

    # ----- Engenharia de atributos -----
    ...

    # ----- Divisão treino/teste -----
    ...

    # ----- Definição e treino dos modelos -----
    ...

    # ----- Comparativo e detalhamento dos modelos -----
    ...

# ==========================================================
# 🏠 Página HOME – Padrão inicial
# ==========================================================
else:
    st.info("👆 Clique em um botão acima para começar.")