# ==========================================================
# 📚 Importação de bibliotecas necessárias
# ==========================================================
import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime


# ==========================================================
# 🖥️ Função principal para exibir a página "Dados"
# ==========================================================
def show():
    # ------------------------------------------------------
    # 📅 Mostrar a data/hora da última atualização da base
    # ------------------------------------------------------
    path = "preco_petroleo.db"
    mod_time = os.path.getmtime(path)
    ultima_atualizacao = datetime.fromtimestamp(mod_time).strftime("%d/%m/%Y %H:%M:%S")

    st.markdown(
        f"<p style='font-size:16px; color:#ffffff; font-weight:bold;'>📅 Dados atualizados pela última vez em: <span style='color:#9C8DF0;'>{ultima_atualizacao}</span></p>",
        unsafe_allow_html=True
    )
    st.markdown("🔗 [Acesse a fonte oficial dos dados (IPEA)](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view)")


    # ------------------------------------------------------
    # 🔄 Botão para atualizar a base de dados do IPEA
    # ------------------------------------------------------
    if st.button("Atualizar base de dados do IPEA"):
        try:
            # 🔗 URL da base de dados
            url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"
            tables = pd.read_html(url)
            df = tables[2].copy()

            # 🧹 Limpeza e padronização dos dados
            df.columns = ["data", "preco_petroleo"]
            df = df.drop(index=0).reset_index(drop=True)
            df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
            df["preco_petroleo"] = df["preco_petroleo"].astype(float) / 100

            # 💾 Salvando no banco SQLite
            with sqlite3.connect("preco_petroleo.db") as conn:
                df.to_sql("preco_petroleo_raw", conn, if_exists="replace", index=False)

            # 💾 Guardar dados no estado da sessão
            st.session_state["extrair_dados"] = True
            st.session_state["preco_petroleo_raw"] = df

            # 📊 Preparar dataframe formatado para exibição
            st.session_state["preco_petroleo_raw_display"] = (
                df[["data", "preco_petroleo"]]
                .rename(columns={"data": "Data", "preco_petroleo": "Preço Petróleo (USD)"})
                .assign(Data=lambda d: d["Data"].dt.strftime("%d/%m/%Y"))
            )

            st.success("✅ Base atualizada e salva em 'preco_petroleo.db'")

        except Exception as e:
            st.error(f"Erro: {e}")

    # ------------------------------------------------------
    # 📂 Exibir dados já carregados (download + dataframe)
    # ------------------------------------------------------
    if st.session_state["extrair_dados"]:
        df = st.session_state["preco_petroleo_raw"]

        # Criar CSV para download
        csv = df.to_csv(index=False).encode("utf-8")
        downloaded = st.download_button(
            "💾 Baixar CSV",
            data=csv,
            file_name="preco_petroleo.csv",
            mime="text/csv",
            key="download_csv"
        )

        # Mostrar confirmação de download
        if downloaded:
            st.caption("✅ Download efetuado")

        # Exibir dados no dataframe
        st.dataframe(st.session_state["preco_petroleo_raw_display"])
