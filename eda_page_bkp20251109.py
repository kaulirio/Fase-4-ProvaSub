# ==========================================================
# 📚 Importação de bibliotecas necessárias para a página EDA
# ==========================================================
import streamlit as st
import pandas as pd
import sqlite3
import altair as alt


# ==========================================================
# 🔎 Função principal para exibir a página "Análise Exploratória"
# ==========================================================
def show():
    # ------------------------------------------------------
    # 💾 Carregar dados do SQLite (com cache)
    # ------------------------------------------------------
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

    # ------------------------------------------------------
    # 🧾 Tentar carregar o dataset; interromper a página em caso de erro
    # ------------------------------------------------------
    try:
        df = load_from_sqlite()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # ------------------------------------------------------
    # 👀 Amostra rápida (últimas linhas)
    # ------------------------------------------------------
    st.subheader("🔎 Análise Exploratória")
    st.write("**Amostra de dados (últimas 10 linhas):**")
    # 📊 Preparar dataframe formatado para exibição
    st.session_state["preco_petroleo_raw_display"] = (
        df[["data", "preco_petroleo"]]
        .rename(columns={"data": "Data", "preco_petroleo": "Preço Petróleo (USD)"})
        .assign(Data=lambda d: d["Data"].dt.strftime("%d/%m/%Y"))
    )
    #st.dataframe(st.session_state["preco_petroleo_raw_display"].tail(10),)
    st.dataframe(
        st.session_state["preco_petroleo_raw_display"]
        .tail(10)        
    )



    # ------------------------------------------------------
    # ⏱️ Filtro de período (início/fim)
    # ------------------------------------------------------
    st.markdown("### ⏱️ Filtro de Período")
    min_d, max_d = df["data"].min(), df["data"].max()

    # Criar duas colunas menores (20% cada) em vez de metade da tela
    col_a, col_b, _ = st.columns([1, 1, 3])  

    with col_a:
        data_ini = st.date_input(
            "Data inicial",
            min_d.date(),
            min_value=min_d.date(),
            max_value=max_d.date()
        )
    with col_b:
        data_fim = st.date_input(
            "Data final",
            max_d.date(),
            min_value=min_d.date(),
            max_value=max_d.date()
        )

    # Validar intervalo de datas
    if pd.to_datetime(data_ini) > pd.to_datetime(data_fim):
        st.warning("⚠️ A data inicial é maior que a final. Ajuste o intervalo.")
        st.stop()



    dff = df[(df["data"] >= pd.to_datetime(data_ini)) & (df["data"] <= pd.to_datetime(data_fim))].copy()

    # ------------------------------------------------------
    # 🔍 Checagens rápidas (linhas, nulos, período)
    # ------------------------------------------------------
    st.markdown("### 🔍 Checagens")
    c1, c2, c3 = st.columns([1, 1,  5])
    with c1:
        st.metric("Linhas", len(dff))
    with c2:
        st.metric("Nulos (preço)", int(dff["preco_petroleo"].isna().sum()))
    with c3:
        st.metric(
            "Período", 
            f'{dff["data"].min().strftime("%d/%m/%Y")} → {dff["data"].max().strftime("%d/%m/%Y")}'
        )


    # ------------------------------------------------------
    # 📌 Estatísticas descritivas (KPIs)
    # ------------------------------------------------------
    st.markdown("### 📌 Estatísticas descritivas")
    k1, k2, k3, k4, k5 = st.columns([1, 1, 1, 1, 6])
    with k1: st.metric("Mín", f'{dff["preco_petroleo"].min():.2f}')
    with k2: st.metric("Mediana", f'{dff["preco_petroleo"].median():.2f}')
    with k3: st.metric("Média", f'{dff["preco_petroleo"].mean():.2f}')
    with k4: st.metric("Máx", f'{dff["preco_petroleo"].max():.2f}')
    with k5: st.metric("Desvio Padrão", f'{dff["preco_petroleo"].std():.2f}')

    # ------------------------------------------------------
    # 📈 Série temporal do preço do petróleo
    # ------------------------------------------------------
    st.markdown("### 📈 Preço do Petróleo (USD)")

    # Copiar dataset filtrado
    dff_plot = dff.copy()

    # Identificar mínimo e máximo
    min_row = dff_plot.loc[dff_plot["preco_petroleo"].idxmin()]
    max_row = dff_plot.loc[dff_plot["preco_petroleo"].idxmax()]
    extremos = pd.DataFrame([min_row, max_row])

    # Gráfico principal (linha + pontos)
    line = alt.Chart(dff_plot).mark_line(point=True).encode(
        x=alt.X("data:T", title="", axis=alt.Axis(format="%b.%y")),  # eixo no formato Mmm.yy
        y=alt.Y("preco_petroleo:Q", title="Preço (USD)"),
        tooltip=[
            alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
            alt.Tooltip("preco_petroleo:Q", title="Preço", format=".2f")
        ]
    )

    # Pontos para maior e menor valor
    pts = alt.Chart(extremos).mark_point(size=100, color="red").encode(
        x="data:T",
        y="preco_petroleo:Q"
    )

    # Labels de valor
    labels = alt.Chart(extremos).mark_text(
        align="left", dx=6, dy=-6, color="red", fontWeight="bold"
    ).encode(
        x="data:T",
        y="preco_petroleo:Q",
        text=alt.Text("preco_petroleo:Q", format=".2f")
    )

    # Combinar
    chart = (line + pts + labels).properties(height=400)
    st.altair_chart(chart, use_container_width=True)



    # ------------------------------------------------------
    # 📊 Distribuição (Histograma) e Dispersão por ano (Boxplot)
    # ------------------------------------------------------
    st.markdown("### 📊 Distribuição (Histograma) e Dispersão (Boxplot) por Ano")
    h1, h2 = st.columns(2)

    with h1:
        bins = st.slider("Bins do histograma", min_value=10, max_value=100, value=40, step=5, key="eda_bins")

        # Histograma com contagem
        hist = alt.Chart(dff).mark_bar().encode(
            x=alt.X("preco_petroleo:Q", bin=alt.Bin(maxbins=bins), title="Preço (USD)"),
            y=alt.Y("count():Q", title="Contagem"),
            tooltip=[alt.Tooltip("count()", title="Qtde")]
        )

        # Labels acima das barras
        labels = hist.mark_text(
            align="center",
            baseline="bottom",
            dy=-2,  # desloca o texto para cima
            color="white"
        ).encode(
            text=alt.Text("count():Q", format="d")  # mostra o valor inteiro
        )

        st.altair_chart((hist + labels).properties(height=300), use_container_width=True)

    with h2:
        # -- checagem rápida
        if dff.empty or dff["preco_petroleo"].dropna().empty:
            st.info("Sem dados para exibir no boxplot.")
        else:
            dff_plot = dff.copy()
            # eixo X categórico garantido (string)
            dff_plot["Ano"] = dff_plot["data"].dt.year.astype(int).astype(str)

            # ticks do eixo (a cada 2 anos) – funciona mesmo como string
            anos = sorted(dff_plot["Ano"].unique().tolist())
            anos_ticks = anos[::2] if len(anos) > 1 else anos

            base = alt.Chart(dff_plot)

            # boxplot por ano (sem extent/opacity para máxima compatibilidade)
            box = base.mark_boxplot(size=14).encode(
                x=alt.X("Ano:N",
                        title="",
                        axis=alt.Axis(values=anos_ticks, labelAngle=-40, labelLimit=90, ticks=False)),
                y=alt.Y("preco_petroleo:Q", title="Preço (USD)", axis=alt.Axis(grid=True))
            )

            # pontos (dispersão) com baixa opacidade
            dots = base.mark_circle(size=12, opacity=0.18).encode(
                x="Ano:N",
                y="preco_petroleo:Q",
                tooltip=[
                    alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
                    alt.Tooltip("preco_petroleo:Q", title="Preço", format=".2f")
                ]
            )

            # média por ano (pré-calculada para evitar transformações complexas)
            mean_by_year = (
                dff_plot.groupby("Ano", as_index=False)["preco_petroleo"].mean()
                .rename(columns={"preco_petroleo": "media_ano"})
            )

            mean_pts = alt.Chart(mean_by_year).mark_point(size=55, filled=True).encode(
                x=alt.X("Ano:N"),
                y=alt.Y("media_ano:Q"),
                color=alt.value("#FFD166"),
                tooltip=[
                    alt.Tooltip("Ano:N", title="Ano"),
                    alt.Tooltip("media_ano:Q", title="Média", format=".2f")
                ]
            )

            chart = (box + dots + mean_pts).properties(height=420).configure_axis(
                gridColor="#444", gridOpacity=0.35, labelFontSize=11, titleFontSize=12
            ).configure_view(
                strokeOpacity=0
            )

            st.altair_chart(chart, use_container_width=True)




    # ------------------------------------------------------
    # 🔗 Heatmap de correlação (com lags e volatilidade)
    # ------------------------------------------------------
    st.markdown("### 🔗 Heatmap de correlação")

    aux = dff.copy()

    # Retorno diário (fração) e em %
    aux["ret_frac"] = aux["preco_petroleo"].pct_change()
    aux["Retorno diário (%)"] = aux["ret_frac"] * 100

    # Médias móveis do preço
    aux["Média móvel 7"] = aux["preco_petroleo"].rolling(7).mean()
    aux["Média móvel 30"] = aux["preco_petroleo"].rolling(30).mean()

    # Lags do preço (t-1, t-5)
    aux["Preço t-1"] = aux["preco_petroleo"].shift(1)
    aux["Preço t-5"] = aux["preco_petroleo"].shift(5)

    # Volatilidade (desvio padrão do retorno) em %
    aux["Vol. 7 dias (σ) (%)"]  = aux["ret_frac"].rolling(7).std()  * 100
    aux["Vol. 30 dias (σ) (%)"] = aux["ret_frac"].rolling(30).std() * 100

    # Nível do preço (rótulo amigável)
    aux["Preço petróleo"] = aux["preco_petroleo"]

    # Seleção apenas de colunas numéricas para correlação
    aux_num = aux[[
        "Preço petróleo",
        "Preço t-1", "Preço t-5",
        "Retorno diário (%)",
        "Média móvel 7", "Média móvel 30",
        "Vol. 7 dias (σ) (%)", "Vol. 30 dias (σ) (%)"
    ]].dropna()

    if len(aux_num) >= 2:
        corr = aux_num.corr()
        corr_reset = corr.reset_index().melt("index")
        corr_reset.columns = ["var_y", "var_x", "correlacao"]

        heat = alt.Chart(corr_reset).mark_rect().encode(
            x=alt.X("var_x:N", title=""),
            y=alt.Y("var_y:N", title=""),
            color=alt.Color("correlacao:Q", scale=alt.Scale(scheme="blueorange"), title="Correlação"),
            tooltip=["var_x", "var_y", alt.Tooltip("correlacao:Q", format=".2f")]
        ).properties(height=320)

        st.altair_chart(heat, use_container_width=True)
    else:
        st.info("Série muito curta após cálculos; não foi possível gerar a matriz de correlação.")


    # ------------------------------------------------------
    # ⚡ Volatilidade (desvio padrão móvel)
    # ------------------------------------------------------
    st.markdown("### ⚡ Volatilidade (7 dias e 30 dias)")

    df_vol = dff.copy()
    df_vol["7 dias"] = df_vol["preco_petroleo"].pct_change().rolling(7).std()
    df_vol["30 dias"] = df_vol["preco_petroleo"].pct_change().rolling(30).std()

    st.line_chart(df_vol.set_index("data")[["7 dias", "30 dias"]])


