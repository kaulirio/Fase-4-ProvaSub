# ==========================================================
# Importação de bibliotecas necessárias
# ==========================================================
import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st


# ==========================================================
# Constantes de configuração
# ==========================================================
IPEA_URL   = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"
DB_PATH    = "preco_petroleo.db"
TABLE_NAME = "preco_petroleo_raw"


# ==========================================================
# 🧩 Utilitários
# ==========================================================
def _fmt_mtime(path: str) -> str:
    """Retorna string de data/hora da última modificação do arquivo ou '—' se não existir."""
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M:%S")
    except FileNotFoundError:
        return "—"

def _pegar_tabela_ipea(url: str) -> pd.DataFrame:
    """
    Lê as tabelas do IPEA e retorna a tabela de interesse.
    Observação: no site atual a 3ª tabela (índice 2) contém a série.
    Inclui fallback caso a estrutura mude.
    """
    tables = pd.read_html(url)  # pode levantar ValueError se nada encontrado
    # 1) Tenta diretamente o índice conhecido
    if len(tables) >= 3:
        df = tables[2].copy()
    else:
        # 2) Fallback: pega a primeira tabela com pelo menos 2 colunas
        df = next((t.copy() for t in tables if t.shape[1] >= 2), None)
        if df is None:
            raise ValueError("Não foi possível identificar a tabela de preços no HTML.")
    return df

def _limpar_padronizar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza colunas, corrige tipos e ajusta o preço (÷ 100).
    Remove a 1ª linha se for cabeçalho duplicado (caso comum do site).
    """
    # Renomeia com segurança para 2 colunas esperadas
    df = df.iloc[:, :2]
    df.columns = ["data", "preco_petroleo"]

    # Remove possíveis linhas de cabeçalho/legenda
    df = df.dropna(how="all").reset_index(drop=True)
    if isinstance(df.loc[0, "data"], str) and df.loc[0, "data"].lower().startswith("data"):
        df = df.drop(index=0).reset_index(drop=True)

    # Converte data e preço
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y", errors="coerce")
    # Lida com valores com vírgula e/ou inteiros em centavos
    df["preco_petroleo"] = (
        df["preco_petroleo"]
        .astype(str)
        .str.replace(".", "", regex=False)  # remove separador de milhar caso exista
        .str.replace(",", ".", regex=False) # normaliza decimal
        .astype(float)
        / 100.0
    )

    # Remove linhas inválidas pós-conversão
    df = df.dropna(subset=["data", "preco_petroleo"]).reset_index(drop=True)
    return df

def _salvar_sqlite(df: pd.DataFrame, path: str, table: str) -> None:
    """Salva o DataFrame no SQLite, substituindo a tabela."""
    with sqlite3.connect(path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)

def _df_display(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara DataFrame formatado para exibição na UI."""
    out = (
        df[["data", "preco_petroleo"]]
        .rename(columns={"data": "Data", "preco_petroleo": "Preço Petróleo (USD)"})
        .assign(Data=lambda d: d["Data"].dt.strftime("%d/%m/%Y"))
    )
    return out


# ==========================================================
# Página "Dados"
# ==========================================================
def show():
    # --- Cabeçalho com última atualização do arquivo local ---
    ultima_atualizacao = _fmt_mtime(DB_PATH)
    st.markdown(
        f"<p style='font-size:16px; color:#ffffff; font-weight:bold;'>📅 Dados atualizados pela última vez em: "
        f"<span style='color:#9C8DF0;'>{ultima_atualizacao}</span></p>",
        unsafe_allow_html=True,
    )
    st.markdown(f"🔗 [Acesse a fonte oficial dos dados (IPEA)]({IPEA_URL})")

    # --- Botão de atualização (busca, limpeza, persistência) ---
    if st.button("Atualizar base de dados do IPEA", use_container_width=True):
        try:
            with st.spinner("Atualizando dados do IPEA..."):
                bruto = _pegar_tabela_ipea(IPEA_URL)
                df = _limpar_padronizar(bruto)
                _salvar_sqlite(df, DB_PATH, TABLE_NAME)

                # Estado de sessão (evita KeyError e facilita próximos passos)
                st.session_state["extrair_dados"] = True
                st.session_state["preco_petroleo_raw"] = df
                st.session_state["preco_petroleo_raw_display"] = _df_display(df)

            st.success("✅ Base atualizada e salva em 'preco_petroleo.db'")

        except Exception as e:
            st.error(f"❌ Erro ao atualizar: {e}")

    # --- Bloco de exibição/baixa dos dados já carregados ---
    if st.session_state.get("extrair_dados") and "preco_petroleo_raw" in st.session_state:
        df: pd.DataFrame = st.session_state["preco_petroleo_raw"]

        # CSV para download
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        downloaded = st.download_button(
            "💾 Baixar CSV",
            data=csv_bytes,
            file_name="preco_petroleo.csv",
            mime="text/csv",
            key="download_csv",
            use_container_width=True,
        )
        if downloaded:
            st.caption("✅ Download efetuado")

        # Tabela na UI
        st.dataframe(
            st.session_state.get("preco_petroleo_raw_display", _df_display(df)),
            use_container_width=True,
        )
    else:
        # Dica para primeiro uso
        st.info("Clique em **“Atualizar base de dados do IPEA”** para baixar e exibir a série.")
