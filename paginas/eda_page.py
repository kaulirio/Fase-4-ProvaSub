# ==========================================================
# Importação de bibliotecas necessárias
# ==========================================================
import sqlite3
from datetime import date
import pandas as pd
import streamlit as st
import altair as alt

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
    """
    Lê a tabela do SQLite e retorna DataFrame ordenado por data.
    Lança RuntimeError com mensagem amigável em caso de falha.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["data"])
        if "data" not in df.columns or "preco_petroleo" not in df.columns:
            raise ValueError("Colunas esperadas 'data' e 'preco_petroleo' não encontradas.")
        df["data"] = pd.to_datetime(df["data"], errors="coerce")
        df = df.dropna(subset=["data", "preco_petroleo"]).sort_values("data").reset_index(drop=True)
        return df
    except Exception as e:
        raise RuntimeError(f"Erro ao ler {table} de {db_path}: {e}")

def df_display(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara DataFrame amigável para tabela da UI (Data formatada + rótulos)."""
    out = (
        df[["data", "preco_petroleo"]]
        .rename(columns={"data": "Data", "preco_petroleo": "Preço Petróleo (USD)"})
        .assign(Data=lambda d: d["Data"].dt.strftime("%d/%m/%Y"))
    )
    return out

def validar_periodo(dini: date, dfim: date) -> bool:
    """Retorna True se o período é válido (início <= fim)."""
    return pd.to_datetime(dini) <= pd.to_datetime(dfim)

def filtrar_periodo(df: pd.DataFrame, dini: date, dfim: date) -> pd.DataFrame:
    """Filtra DataFrame no intervalo [dini, dfim]."""
    i = pd.to_datetime(dini)
    f = pd.to_datetime(dfim)
    return df[(df["data"] >= i) & (df["data"] <= f)].copy()

# ==========================================================
# Página "Análise Exploratória"
# ==========================================================
def show():
    st.subheader("🔎 Análise Exploratória")

    # --- Carregamento de dados (com handling de erro) ---
    try:
        df = load_from_sqlite()
    except Exception as e:
        st.error(str(e))
        st.stop()

    if df.empty:
        st.info("Base vazia. Atualize os dados na página **Dados** e volte aqui.")
        st.stop()

    # --- Amostra (últimas 10 linhas) ---
    st.write("**Amostra de dados (últimas 10 linhas):**")
    st.session_state["preco_petroleo_raw_display"] = df_display(df)
    st.dataframe(st.session_state["preco_petroleo_raw_display"].tail(10), use_container_width=True)

    # --- Filtro de período ---
    st.markdown("### ⏱️ Filtro de Período")
    min_d, max_d = df["data"].min(), df["data"].max()
    col_a, col_b, _ = st.columns([1, 1, 3])

    with col_a:
        data_ini = st.date_input("Data inicial", min_d.date(), min_value=min_d.date(), max_value=max_d.date())
    with col_b:
        data_fim = st.date_input("Data final",   max_d.date(), min_value=min_d.date(), max_value=max_d.date())

    if not validar_periodo(data_ini, data_fim):
        st.warning("⚠️ A data inicial é maior que a final. Ajuste o intervalo.")
        st.stop()

    dff = filtrar_periodo(df, data_ini, data_fim)
    if dff.empty:
        st.info("Sem dados no intervalo selecionado.")
        st.stop()

    # --- Checagens rápidas ---
    st.markdown("### 🔍 Checagens")
    c1, c2, c3 = st.columns([1, 1, 5])
    with c1:
        st.metric("Linhas", len(dff))
    with c2:
        st.metric("Nulos (preço)", int(dff["preco_petroleo"].isna().sum()))
    with c3:
        st.metric("Período", f'{dff["data"].min():%d/%m/%Y} → {dff["data"].max():%d/%m/%Y}')

    # --- Estatísticas descritivas ---
    st.markdown("### 📌 Estatísticas descritivas")
    k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 6])
    with k1: st.metric("Mín",     f'{dff["preco_petroleo"].min():.2f}')
    with k2: st.metric("Mediana", f'{dff["preco_petroleo"].median():.2f}')
    with k3: st.metric("Média",   f'{dff["preco_petroleo"].mean():.2f}')
    with k4: st.metric("Máx",     f'{dff["preco_petroleo"].max():.2f}')
    with k5: st.metric("Desvio Padrão", f'{dff["preco_petroleo"].std():.2f}')

    # --- Série temporal (linha + extremos) ---
    st.markdown("### 📈 Preço do Petróleo (USD)")
    dff_plot = dff.copy()

    # Proteção contra idxmin/idxmax em série vazia
    try:
        min_row = dff_plot.loc[dff_plot["preco_petroleo"].idxmin()]
        max_row = dff_plot.loc[dff_plot["preco_petroleo"].idxmax()]
        extremos = pd.DataFrame([min_row, max_row])
    except ValueError:
        extremos = pd.DataFrame(columns=dff_plot.columns)

    line = alt.Chart(dff_plot).mark_line(point=True).encode(
        x=alt.X("data:T", title="", axis=alt.Axis(format="%b.%y")),
        y=alt.Y("preco_petroleo:Q", title="Preço (USD)"),
        tooltip=[alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
                 alt.Tooltip("preco_petroleo:Q", title="Preço", format=".2f")]
    )

    pts = alt.Chart(extremos).mark_point(size=100, color="red").encode(
        x="data:T", y="preco_petroleo:Q"
    )

    labels = alt.Chart(extremos).mark_text(align="left", dx=6, dy=-6, color="red", fontWeight="bold").encode(
        x="data:T", y="preco_petroleo:Q", text=alt.Text("preco_petroleo:Q", format=".2f")
    )

    st.altair_chart((line + pts + labels).properties(height=400), use_container_width=True)

    # --- Distribuição (Histograma) e Boxplot por Ano ---
    st.markdown("### 📊 Distribuição (Histograma) e Dispersão (Boxplot) por Ano")
    h1, h2 = st.columns(2)

    with h1:
        bins = st.slider("Bins do histograma", min_value=10, max_value=100, value=40, step=5, key="eda_bins")
        hist = alt.Chart(dff).mark_bar().encode(
            x=alt.X("preco_petroleo:Q", bin=alt.Bin(maxbins=bins), title="Preço (USD)"),
            y=alt.Y("count():Q", title="Contagem"),
            tooltip=[alt.Tooltip("count()", title="Qtde")]
        )
        # rótulo de contagem (opcional; não aparece para bins vazios)
        labels = hist.mark_text(align="center", baseline="bottom", dy=-2).encode(
            text=alt.Text("count():Q", format="d")
        )
        st.altair_chart((hist + labels).properties(height=300), use_container_width=True)

    with h2:
        if dff["preco_petroleo"].dropna().empty:
            st.info("Sem dados para exibir no boxplot.")
        else:
            dff_box = dff.copy()
            dff_box["Ano"] = dff_box["data"].dt.year.astype(int).astype(str)
            anos = sorted(dff_box["Ano"].unique().tolist())
            anos_ticks = anos[::2] if len(anos) > 1 else anos

            base = alt.Chart(dff_box)
            box = base.mark_boxplot(size=14).encode(
                x=alt.X("Ano:N", title="", axis=alt.Axis(values=anos_ticks, labelAngle=-40, labelLimit=90, ticks=False)),
                y=alt.Y("preco_petroleo:Q", title="Preço (USD)", axis=alt.Axis(grid=True))
            )
            dots = base.mark_circle(size=12, opacity=0.18).encode(
                x="Ano:N",
                y="preco_petroleo:Q",
                tooltip=[alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
                         alt.Tooltip("preco_petroleo:Q", title="Preço", format=".2f")]
            )
            mean_by_year = (
                dff_box.groupby("Ano", as_index=False)["preco_petroleo"].mean()
                .rename(columns={"preco_petroleo": "media_ano"})
            )
            mean_pts = alt.Chart(mean_by_year).mark_point(size=55, filled=True).encode(
                x="Ano:N", y="media_ano:Q",
                tooltip=[alt.Tooltip("Ano:N", title="Ano"), alt.Tooltip("media_ano:Q", title="Média", format=".2f")]
            )
            chart = (box + dots + mean_pts).properties(height=420).configure_view(strokeOpacity=0)
            st.altair_chart(chart, use_container_width=True)

    # --- Heatmap de correlação (lags/volatilidade) ---
    st.markdown("### 🔗 Heatmap de correlação")
    aux = dff.copy()
    aux["ret_frac"] = aux["preco_petroleo"].pct_change()
    aux["Retorno diário (%)"] = aux["ret_frac"] * 100
    aux["Média móvel 7"] = aux["preco_petroleo"].rolling(7).mean()
    aux["Média móvel 30"] = aux["preco_petroleo"].rolling(30).mean()
    aux["Preço t-1"] = aux["preco_petroleo"].shift(1)
    aux["Preço t-5"] = aux["preco_petroleo"].shift(5)
    aux["Vol. 7 dias (σ) (%)"]  = aux["ret_frac"].rolling(7).std()  * 100
    aux["Vol. 30 dias (σ) (%)"] = aux["ret_frac"].rolling(30).std() * 100
    aux["Preço petróleo"] = aux["preco_petroleo"]

    cols_corr = [
        "Preço petróleo", "Preço t-1", "Preço t-5",
        "Retorno diário (%)", "Média móvel 7", "Média móvel 30",
        "Vol. 7 dias (σ) (%)", "Vol. 30 dias (σ) (%)"
    ]
    aux_num = aux[cols_corr].dropna()

    if len(aux_num) >= 2:
        corr = aux_num.corr()
        corr_reset = corr.reset_index().melt("index")
        corr_reset.columns = ["var_y", "var_x", "correlacao"]
        heat = alt.Chart(corr_reset).mark_rect().encode(
            x=alt.X("var_x:N", title=""),
            y=alt.Y("var_y:N", title=""),
            color=alt.Color("correlacao:Q", title="Correlação"),  # escala default para compatibilidade ampla
            tooltip=["var_x", "var_y", alt.Tooltip("correlacao:Q", format=".2f")]
        ).properties(height=320)
        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("Série muito curta após cálculos; não foi possível gerar a matriz de correlação.")

    # --- Volatilidade (desvio padrão móvel) ---
    st.markdown("### ⚡ Volatilidade (7 dias e 30 dias)")
    df_vol = dff.copy()
    df_vol["Vol_7d"]  = df_vol["preco_petroleo"].pct_change().rolling(7).std()
    df_vol["Vol_30d"] = df_vol["preco_petroleo"].pct_change().rolling(30).std()

    # st.line_chart é rápido, mas sem tooltips; Altair abaixo oferece interação melhor
    vol_chart = alt.Chart(df_vol).transform_fold(
        ["Vol_7d", "Vol_30d"], as_=["Janela", "Vol"]
    ).mark_line().encode(
        x=alt.X("data:T", title=""),
        y=alt.Y("Vol:Q", title="Volatilidade (DP do retorno)"),
        color=alt.Color("Janela:N", title="Janela"),
        tooltip=[alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
                 alt.Tooltip("Janela:N"), alt.Tooltip("Vol:Q", format=".4f")]
    ).properties(height=320)

    st.altair_chart(vol_chart, use_container_width=True)
