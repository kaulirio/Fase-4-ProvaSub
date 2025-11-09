# ==========================================================
# 📚 Importações necessárias para a página de Modelos
# ==========================================================
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import altair as alt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ==========================================================
# 🤖 Página "Modelos Preditivos" — função principal
# ==========================================================
def show():
    st.subheader(" Modelos Preditivos — Preço do Petróleo (Regressão)")

    # ------------------------------------------------------
    # 💾 Carregar dados (prioriza dataset limpo da EDA; fallback: SQLite)
    # ------------------------------------------------------
    @st.cache_data(ttl=600)
    def load_sql(db_path="preco_petroleo.db", table="preco_petroleo_raw"):
        with sqlite3.connect(db_path) as conn:
            df_ = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["data"])
        df_["data"] = pd.to_datetime(df_["data"])
        df_ = df_.sort_values("data").reset_index(drop=True)
        return df_

    if "eda_df_clean" in st.session_state and not st.session_state["eda_df_clean"].empty:
        df_base = st.session_state["eda_df_clean"].copy()
    else:
        df_base = load_sql()

    if len(df_base) < 150:
        st.warning("Poucos dados para treinar. Carregue a base completa na página de dados/EDA.")
        st.stop()

    # ------------------------------------------------------
    # 🛠️ Engenharia de atributos (lags, médias, volatilidade, retornos)
    # ------------------------------------------------------
    df = df_base.copy()
    df = df.sort_values("data").reset_index(drop=True)

    # Retorno simples
    df["ret"] = df["preco_petroleo"].pct_change()

    # Lags do preço e do retorno
    for L in [1, 2, 3, 5, 7, 14]:
        df[f"preco_t-{L}"] = df["preco_petroleo"].shift(L)
        df[f"ret_t-{L}"] = df["ret"].shift(L)

    # Médias móveis do preço e do retorno
    for W in [7, 30, 90]:
        df[f"mm_prec_{W}"] = df["preco_petroleo"].rolling(W).mean()
        df[f"mm_ret_{W}"] = df["ret"].rolling(W).mean()

    # Volatilidade (desvio padrão dos retornos)
    for W in [7, 30]:
        df[f"vol_{W}"] = df["ret"].rolling(W).std()

    # Alvo: preço do dia seguinte
    df["y_next"] = df["preco_petroleo"].shift(-1)

    # Remover NaNs gerados por lags/médias/shift
    df = df.dropna().reset_index(drop=True)

    

    # ------------------------------------------------------
    # 🧱 Seleção de features e target (X, y)
    # ------------------------------------------------------
    feature_cols = [c for c in df.columns if c not in ["data", "preco_petroleo", "y_next"]]
    X = df[feature_cols].values
    y = df["y_next"].values
    datas = df["data"].values
    y_ref = df["preco_petroleo"].values   # preço observado (para comparação)

    # ------------------------------------------------------
    # 🔧 Configurações de treino/teste + escolha do modelo
    # ------------------------------------------------------
    st.markdown("##### ⚙️ Configurações de treino/teste e modelo")
    col1, col2, col3, col4 = st.columns([1.25, 0.25, 1.25, 0.25])  # centraliza no meio
    with col1:
        # Split temporal (teste = última parte da série)
        test_pct = st.slider("Proporção de teste (final da série)", 10, 40, 20, step=5)
        split_idx = int(len(X) * (1 - test_pct / 100))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        datas_test = datas[split_idx:]
        y_ref_test = y_ref[split_idx:]
    # Opções de modelos
    with col3: 
        modelo_nome = st.selectbox(
            "Modelo de regressão",
            ["LinearRegression (baseline)", "RandomForestRegressor", "GradientBoostingRegressor"],
            index=1
        )        


    # Mostrar no caption
    st.caption(
        f"**Treino:** {pd.to_datetime(datas[0]).strftime('%d.%m.%Y')} → {pd.to_datetime(datas[split_idx-1]).strftime('%d.%m.%Y')}  ({len(X_train)} linhas) | "
        f"**Teste:** {pd.to_datetime(datas[split_idx]).strftime('%d.%m.%Y')} → {pd.to_datetime(datas[-1]).strftime('%d.%m.%Y')} ({len(X_test)} linhas)"
    )
 

    st.markdown(
        """
        <hr style="border: none; height: 1px; background-color: white; margin: 0px 0;"> <br />
        """,
        unsafe_allow_html=True
    )



    # Hiperparâmetros essenciais
    col_a, col_b, col_c, col_d = st.columns([1.25, 0.25, 0.75, 0.75]) 
    with col_a:
        if modelo_nome == "RandomForestRegressor":
            n_estimators = st.slider("Árvores (n_estimators)", 100, 600, 300, step=50)
            max_depth = st.slider("Profundidade máx (max_depth)", 2, 20, 10, step=1)
        elif modelo_nome == "GradientBoostingRegressor":
            n_estimators = st.slider("Estágios (n_estimators)", 50, 500, 200, step=50)
            learning_rate = st.select_slider("Learning rate", options=[0.01, 0.03, 0.05, 0.1, 0.2], value=0.1)
        else:
            pass
    with col_c:
        if modelo_nome == "LinearRegression (baseline)":
            escalar = True
            st.checkbox(
                "Aplicar StandardScaler nas features (pipeline)",
                value=True,
                disabled=True  # fica bloqueado
            )
        elif modelo_nome == "RandomForestRegressor":
            escalar = False
            st.checkbox(
                "Aplicar StandardScaler nas features (pipeline)",
                value=False,
                disabled=True  # fica bloqueado
            )
        else:
            escalar = st.checkbox(
                "Aplicar StandardScaler nas features (pipeline)",
                value=False
            )

    with col_d:
        do_cv = st.checkbox("Validar com TimeSeriesSplit (5 dobras)", value=False)

    # ------------------------------------------------------
    # ▶️ Montar pipeline, treinar e (opcional) validar
    # ------------------------------------------------------
    if modelo_nome == "LinearRegression (baseline)":
        base_model = LinearRegression()
    elif modelo_nome == "RandomForestRegressor":
        base_model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
        )
    else:
        base_model = GradientBoostingRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, random_state=42
        )

    steps = []
    if escalar:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", base_model))
    pipe = Pipeline(steps)

    def mape(y_true, y_pred):
        denom = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
        return np.mean(np.abs((y_true - y_pred) / denom)) * 100

    # Cross-validation temporal (opcional)
    if do_cv:
        tscv = TimeSeriesSplit(n_splits=5)
        maes, rmses, mapes = [], [], []

        for tr_idx, te_idx in tscv.split(X_train):
            pipe.fit(X_train[tr_idx], y_train[tr_idx])
            y_pred_fold = pipe.predict(X_train[te_idx])

            maes.append(mean_absolute_error(y_train[te_idx], y_pred_fold))

            mse  = mean_squared_error(y_train[te_idx], y_pred_fold)
            rmses.append(np.sqrt(mse))

            mapes.append(mape(y_train[te_idx], y_pred_fold))

        st.info(
            f"Cross-validation (5 dobras) — "
            f"MAE: {np.mean(maes):.3f}, Desvio padrão (folds): {np.std(maes):.3f} | "
            f"RMSE: {np.mean(rmses):.3f}, Desvio padrão (folds): {np.std(rmses):.3f} | "
            f"MAPE: {np.mean(mapes):.2f}%, Desvio padrão (folds): {np.std(mapes):.2f}%"
        )

        # >>> Reajusta no treino completo e calcula métricas no TESTE final
        pipe.fit(X_train, y_train)
        y_pred_test = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mape_val = mape(y_test, y_pred_test)

    else:
        # Treino normal sem CV e métricas no TESTE final
        pipe.fit(X_train, y_train)
        y_pred_test = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mape_val = mape(y_test, y_pred_test)


    # ------------------------------------------------------
    # 📈 Previsões no conjunto de teste + métricas
    # ------------------------------------------------------    

    st.markdown("### 🧮 Métricas (Teste)")
    k1, k2, k3 = st.columns(3)
    k1.metric("MAE", f"{mae:.3f}")
    k2.metric("RMSE", f"{rmse:.3f}")
    k3.metric("MAPE (%)", f"{mape_val:.2f}")

    # ------------------------------------------------------
    # 📊 Gráfico: preço real vs. previsto no tempo (teste)
    # ------------------------------------------------------      
    st.markdown("### ⏱️ Real vs. Previsto — Janela de Teste")
    plot_df = pd.DataFrame({
        "data": pd.to_datetime(datas_test),
        "Preço (real)": y_test,
        "Preço (previsto)": y_pred_test
    })


    # supondo que seu DataFrame final para o gráfico seja plot_df (ou plot_df_f)
    # calcule um domínio comum para real e previsto
    vmin = float(np.nanmin(plot_df[["Preço (real)", "Preço (previsto)"]].values))
    vmax = float(np.nanmax(plot_df[["Preço (real)", "Preço (previsto)"]].values))

    # opcional: margem de 5% para não “colar” nas bordas
    pad = 0.1 * (vmax - vmin) if vmax > vmin else 1.0
    domain = [vmin - pad, vmax + pad]

    line_real = alt.Chart(plot_df).mark_line(color="#4FC3F7").encode(
        x=alt.X("data:T", axis=alt.Axis(format="%b.%y", title="")),
        y=alt.Y(
            "Preço (real):Q",
            title="Preço Real (USD)",
            scale=alt.Scale(domain=domain),                 # <- mesmo domínio
            axis=alt.Axis(titleColor="#4FC3F7")
        ),
        tooltip=[
            alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
            alt.Tooltip("Preço (real):Q", format=".2f"),
        ],
    )

    line_pred = alt.Chart(plot_df).mark_line(color="#FFA726").encode(
        x="data:T",
        y=alt.Y(
            "Preço (previsto):Q",
            title="Preço (previsto)",
            scale=alt.Scale(domain=domain),                 # <- mesmo domínio
            axis=alt.Axis(titleColor="#FFA726")
        ),
        tooltip=[
            alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"),
            alt.Tooltip("Preço (previsto):Q", format=".2f"),
        ],
    )

    # mantém dois eixos, mas com a MESMA escala (mesmo domínio em ambos)
    chart = alt.layer(line_real, line_pred).resolve_scale(y="independent").properties(height=360)
    st.altair_chart(chart, use_container_width=True)




    # dados: plot_df tem colunas ['data', 'Preço (real)', 'Preço (previsto)']
    plot_df["ano"] = plot_df["data"].dt.year

    min_ano = int(plot_df["ano"].min())
    max_ano = int(plot_df["ano"].max())

    # param com slider (v5)
    year_param = alt.param(
        name='ano_param',
        value=max_ano,
        bind=alt.binding_range(name='Ano (filtro):', min=min_ano, max=max_ano, step=1)
    )

    base = alt.Chart(plot_df).add_params(year_param).transform_filter(
        alt.datum.ano == year_param
    )

    line_real = base.mark_line(color="#4FC3F7").encode(
        x=alt.X("data:T", axis=alt.Axis(format="%d.%m.%y", title="")),
        y=alt.Y("Preço (real):Q", title="Preço (USD)")
    )

    line_prev = base.mark_line(color="#FFA726").encode(
        x="data:T",
        y="Preço (previsto):Q"
    )

    chart = alt.layer(line_real, line_prev).properties(
        height=360,
        title=alt.TitleParams(
            text="Filtro por Ano — Real vs Previsto",
            anchor="middle",   # centre it
            orient="top",      # put at the top
            fontSize=16,
            fontWeight="bold"        
        )
        )
    st.altair_chart(chart, use_container_width=True)



    # ------------------------------------------------------
    # 📉 Resíduos no tempo e Dispersão (real vs. previsto)
    # ------------------------------------------------------
    st.markdown("### 🔎 Análise de Resíduos")
    res_df = plot_df.copy()
    res_df["Resíduo"] = res_df["Preço (real)"] - res_df["Preço (previsto)"]

    res_line = alt.Chart(res_df).mark_line().encode(
        x=alt.X("data:T", axis=alt.Axis(format="%b.%y"), title=""),
        y=alt.Y("Resíduo:Q", title="Resíduo (USD)"),
        tooltip=[alt.Tooltip("data:T", format="%d/%m/%Y"), alt.Tooltip("Resíduo:Q", format=".2f")]
    ).properties(height=220)

    disp = alt.Chart(res_df).mark_circle(opacity=0.5).encode(
        x=alt.X("Preço (real):Q"),
        y=alt.Y("Preço (previsto):Q"),
        tooltip=[alt.Tooltip("Preço (real):Q", format=".2f"),
                 alt.Tooltip("Preço (previsto):Q", format=".2f")]
    ).properties(height=260)

    st.altair_chart(res_line, use_container_width=True)
    st.altair_chart(disp, use_container_width=True)

    feature_labels = {}

    for c in feature_cols:
        if c.startswith("preco_t-"):
            lag = c.split("-")[1]
            feature_labels[c] = f"Preço (t-{lag})"

        elif c.startswith("ret_t-"):
            lag = c.split("-")[1]
            feature_labels[c] = f"Retorno (t-{lag})"

        elif c.startswith("mm_prec_"):
            w = c.split("_")[2]
            feature_labels[c] = f"Média Móvel ({w} dias)"

        elif c.startswith("mm_ret_"):
            w = c.split("_")[2]
            feature_labels[c] = f"Média Móvel do Retorno ({w} dias)"

        elif c.startswith("vol_"):
            w = c.split("_")[1]
            feature_labels[c] = f"Volatilidade ({w} dias)"

        elif c == "ret":
            feature_labels[c] = "Retorno Diário"

        else:
            feature_labels[c] = c  # fallback para não quebrar


    # ------------------------------------------------------
    # 🌳 Importância das features (apenas modelos de árvore/boosting)
    # ------------------------------------------------------
    if modelo_nome in ["RandomForestRegressor", "GradientBoostingRegressor"]:
        try:
            importances = pipe.named_steps["model"].feature_importances_
            imp_df = pd.DataFrame({"feature": feature_cols, "importancia": importances})
            imp_df["feature_label"] = imp_df["feature"].map(feature_labels)

            imp_df = imp_df.sort_values("importancia", ascending=False).head(5)

            st.markdown("### 🧠 Importância das Features (Top 5)")
            bar = alt.Chart(imp_df).mark_bar().encode(
                x=alt.X("importancia:Q", title="Importância"),
                y=alt.Y("feature_label:N", sort="-x", title=""),
                tooltip=["feature_label", alt.Tooltip("importancia:Q", format=".4f")]
            ).properties(height=360)
            st.altair_chart(bar, use_container_width=True)
        except Exception:
            st.info("Importância de features não disponível para este modelo.")

    # ------------------------------------------------------
    # 💾 Baixar previsões (CSV)
    # ------------------------------------------------------
    # st.markdown("### 💾 Download das previsões (janela de teste)")
    # out = plot_df.copy()
    # out["data"] = out["data"].dt.strftime("%d/%m/%Y")
    # csv = out.to_csv(index=False).encode("utf-8")
    # st.download_button(
    #     "⬇️ Baixar CSV (real vs. previsto)",
    #     data=csv,
    #     file_name="previsoes_teste.csv",
    #     mime="text/csv",
    #     key="download_preds_csv"
    # )
