# ==========================================================
# Importação de bibliotecas necessárias
# ==========================================================
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

# ==========================================================
# Constantes
# ==========================================================
DB_PATH    = "preco_petroleo.db"
TABLE_NAME = "preco_petroleo_raw"
RANDOM_SEED = 42

# ==========================================================
# Utilitários
# ==========================================================
@st.cache_data(ttl=600)
def load_sql(db_path: str = DB_PATH, table: str = TABLE_NAME) -> pd.DataFrame:
    """Lê SQLite, garante colunas esperadas e ordena por data."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["data"])
    if {"data", "preco_petroleo"} - set(df.columns):
        raise RuntimeError("Colunas 'data' e 'preco_petroleo' não encontradas.")
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.dropna(subset=["data", "preco_petroleo"]).sort_values("data").reset_index(drop=True)
    return df

@dataclass
class FeatureConfig:
    lags: List[int] = (1, 2, 3, 5, 7, 14)
    windows_mm: List[int] = (7, 30, 90)
    windows_vol: List[int] = (7, 30)
    target_shift: int = -1  # prever t+1

def build_features(df_base: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Cria retornos, lags, MMs, volatilidades e target; remove NaNs."""
    df = df_base.copy().sort_values("data").reset_index(drop=True)
    df["ret"] = df["preco_petroleo"].pct_change()

    for L in cfg.lags:
        df[f"preco_t-{L}"] = df["preco_petroleo"].shift(L)
        df[f"ret_t-{L}"] = df["ret"].shift(L)

    for W in cfg.windows_mm:
        df[f"mm_prec_{W}"] = df["preco_petroleo"].rolling(W).mean()
        df[f"mm_ret_{W}"]  = df["ret"].rolling(W).mean()

    for W in cfg.windows_vol:
        df[f"vol_{W}"] = df["ret"].rolling(W).std()

    df["y_next"] = df["preco_petroleo"].shift(cfg.target_shift)
    df = df.dropna().reset_index(drop=True)
    return df

def select_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Separa features/target e devolve arrays + nomes de features."""
    feature_cols = [c for c in df.columns if c not in ["data", "preco_petroleo", "y_next"]]
    X = df[feature_cols].to_numpy()
    y = df["y_next"].to_numpy()
    datas = df["data"].to_numpy()
    y_ref = df["preco_petroleo"].to_numpy()
    return X, y, datas, y_ref, feature_cols

def holdout_split(X: np.ndarray, y: np.ndarray, datas: np.ndarray, yref: np.ndarray, test_pct: int):
    """Split temporal no final da série."""
    split_idx = int(len(X) * (1 - test_pct / 100))
    return (
        X[:split_idx], X[split_idx:],
        y[:split_idx], y[split_idx:],
        datas[split_idx:], yref[split_idx:], split_idx
    )

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE robusto a zeros."""
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def model_from_name(nome: str, hyper: Dict) -> object:
    """Constrói o estimador base a partir do nome e hiperparâmetros."""
    if nome == "LinearRegression (baseline)":
        return LinearRegression()
    if nome == "RandomForestRegressor":
        return RandomForestRegressor(
            n_estimators=hyper.get("n_estimators", 300),
            max_depth=hyper.get("max_depth", 10),
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    if nome == "GradientBoostingRegressor":
        return GradientBoostingRegressor(
            n_estimators=hyper.get("n_estimators", 200),
            learning_rate=hyper.get("learning_rate", 0.1),
            random_state=RANDOM_SEED,
        )
    # HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        learning_rate=hyper.get("learning_rate", 0.05),
        max_depth=hyper.get("max_depth", 6),
        l2_regularization=hyper.get("l2", 1.0),
        max_bins=255,
        early_stopping=True,
        random_state=RANDOM_SEED,
    )

def feature_labels_map(cols: List[str]) -> Dict[str, str]:
    """Rótulos amigáveis para exibição de importâncias."""
    labels = {}
    for c in cols:
        if c.startswith("preco_t-"):
            labels[c] = f"Preço (t-{c.split('-')[1]})"
        elif c.startswith("ret_t-"):
            labels[c] = f"Retorno (t-{c.split('-')[1]})"
        elif c.startswith("mm_prec_"):
            labels[c] = f"Média Móvel ({c.split('_')[2]} dias)"
        elif c.startswith("mm_ret_"):
            labels[c] = f"Média Móvel do Retorno ({c.split('_')[2]} dias)"
        elif c.startswith("vol_"):
            labels[c] = f"Volatilidade ({c.split('_')[1]} dias)"
        elif c == "ret":
            labels[c] = "Retorno Diário"
        else:
            labels[c] = c
    return labels

# ==========================================================
# Página "Modelos Preditivos" — função principal
# ==========================================================
def show():
    st.subheader(" Modelos Preditivos — Preço do Petróleo (Regressão)")

    # --- Fonte de dados: EDA limpa (se existir) ou SQLite ---
    df_base = (
        st.session_state["eda_df_clean"].copy()
        if "eda_df_clean" in st.session_state and not st.session_state["eda_df_clean"].empty
        else load_sql()
    )
    if len(df_base) < 150:
        st.warning("Poucos dados para treinar. Carregue a base completa na página de Dados/EDA.")
        st.stop()

    # --- Engenharia de atributos ---
    cfg = FeatureConfig()
    df = build_features(df_base, cfg)

    # --- Seleção de X/y e split temporal ---
    X, y, datas, y_ref, feature_cols = select_xy(df)

    st.markdown("##### ⚙️ Configurações de treino/teste e modelo")
    col1, _, col3, _ = st.columns([1.25, 0.25, 1.25, 0.25])

    with col1:
        test_pct = st.slider("Proporção de teste (final da série)", 10, 40, 20, step=5)
        X_train, X_test, y_train, y_test, datas_test, y_ref_test, split_idx = holdout_split(X, y, datas, y_ref, test_pct)

    with col3:
        modelo_nome = st.selectbox(
            "Modelo de regressão",
            ["LinearRegression (baseline)", "RandomForestRegressor", "GradientBoostingRegressor", "HistGradientBoostingRegressor"],
            index=0,
        )

    st.caption(
        f"**Treino:** {pd.to_datetime(datas[0]):%d.%m.%Y} → {pd.to_datetime(datas[split_idx-1]):%d.%m.%Y} "
        f"({len(X_train)} linhas) | "
        f"**Teste:** {pd.to_datetime(datas[split_idx]):%d.%m.%Y} → {pd.to_datetime(datas[-1]):%d.%m.%Y} "
        f"({len(X_test)} linhas)"
    )
    st.markdown("<hr style='border: none; height: 1px; background-color: white; margin: 0px 0;'> <br />", unsafe_allow_html=True)

    # --- Hiperparâmetros essenciais + scaler ---
    col_a, _, col_c, col_d = st.columns([1.25, 0.25, 0.75, 0.75])

    hyper = {}
    with col_a:
        if modelo_nome == "RandomForestRegressor":
            hyper["n_estimators"] = st.slider("Árvores (n_estimators)", 100, 600, 300, step=50)
            hyper["max_depth"]    = st.slider("Profundidade máx (max_depth)", 2, 20, 10, step=1)

        elif modelo_nome == "GradientBoostingRegressor":
            hyper["n_estimators"] = st.slider("Estágios (n_estimators)", 50, 500, 200, step=50)
            hyper["learning_rate"] = st.select_slider("Learning rate", options=[0.01, 0.03, 0.05, 0.1, 0.2], value=0.1)

        elif modelo_nome == "HistGradientBoostingRegressor":
            hyper["learning_rate"] = st.select_slider("Learning rate", options=[0.01, 0.02, 0.03, 0.05, 0.1], value=0.05)
            hyper["max_depth"]     = st.slider("Profundidade máx (max_depth)", 3, 12, 6)
            hyper["l2"]            = st.slider("Regularização L2", 0.0, 5.0, 1.0, step=0.1)

    with col_c:
        # Scaler apenas para baseline linear (evitar leakage com árvores não é crítico; elas não exigem escala)
        escalar = (modelo_nome == "LinearRegression (baseline)")
        st.checkbox("Aplicar StandardScaler nas features (pipeline)", value=escalar, disabled=True)

    with col_d:
        do_cv = st.checkbox("Validar com TimeSeriesSplit (5 dobras)", value=False)

    # --- Random Search opcional (RF) ---
    best_model = None
    if modelo_nome == "RandomForestRegressor":
        aplicar_rs = st.checkbox("Aplicar Random Search (tuning)", value=False, key="rs_rf")
        if aplicar_rs:
            cv_obj = TimeSeriesSplit(n_splits=3)
            param_dist = {
                "n_estimators": [100, 200, 300, 400, 500, 600],
                "max_depth": [3, 5, 7, 10, 12, 15, 20, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
            }
            with st.spinner("Executando RandomizedSearchCV…"):
                search = RandomizedSearchCV(
                    estimator=RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
                    param_distributions=param_dist,
                    n_iter=40,
                    cv=cv_obj,
                    scoring="neg_mean_absolute_error",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                )
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
            st.success("✅ Random Search concluído!")
            st.write("**Melhores hiperparâmetros encontrados:**")
            st.json(search.best_params_)

    # --- Pipeline, treino e (opcional) CV temporal ---
    base_model = best_model if best_model is not None else model_from_name(modelo_nome, hyper)
    steps = [("scaler", StandardScaler())] if escalar else []
    steps.append(("model", base_model))
    pipe = Pipeline(steps)

    if do_cv:
        tscv = TimeSeriesSplit(n_splits=5)
        maes, rmses, mapes = [], [], []
        for tr_idx, te_idx in tscv.split(X_train):
            pipe.fit(X_train[tr_idx], y_train[tr_idx])
            y_pred_fold = pipe.predict(X_train[te_idx])
            maes.append(mean_absolute_error(y_train[te_idx], y_pred_fold))
            rmses.append(np.sqrt(mean_squared_error(y_train[te_idx], y_pred_fold)))
            mapes.append(mape(y_train[te_idx], y_pred_fold))

        st.info(
            f"Cross-validation (5 dobras) — "
            f"MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f} | "
            f"RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f} | "
            f"MAPE: {np.mean(mapes):.2f}% ± {np.std(mapes):.2f}%"
        )

    # Treino final e predição no teste
    pipe.fit(X_train, y_train)
    y_pred_test = pipe.predict(X_test)

    # --- Métricas de teste ---
    mae  = mean_absolute_error(y_test, y_pred_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mape_val = mape(y_test, y_pred_test)

    st.markdown("### 🧮 Métricas (Teste)")
    k1, k2, k3 = st.columns(3)
    k1.metric("MAE", f"{mae:.3f}")
    k2.metric("RMSE", f"{rmse:.3f}")
    k3.metric("MAPE (%)", f"{mape_val:.2f}")

    # --- Real vs Previsto (janela de teste) ---
    st.markdown("### ⏱️ Real vs. Previsto — Janela de Teste")
    plot_df = pd.DataFrame({
        "data": pd.to_datetime(datas_test),
        "Preço (real)": y_test,
        "Preço (previsto)": y_pred_test,
    })

    vmin = float(np.nanmin(plot_df[["Preço (real)", "Preço (previsto)"]].to_numpy()))
    vmax = float(np.nanmax(plot_df[["Preço (real)", "Preço (previsto)"]].to_numpy()))
    pad = 0.1 * (vmax - vmin) if vmax > vmin else 1.0
    domain = [vmin - pad, vmax + pad]

    line_real = alt.Chart(plot_df).mark_line(color="#4FC3F7").encode(
        x=alt.X("data:T", axis=alt.Axis(format="%b.%y", title="")),
        y=alt.Y("Preço (real):Q", title="Preço Real (USD)", scale=alt.Scale(domain=domain), axis=alt.Axis(titleColor="#4FC3F7")),
        tooltip=[alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"), alt.Tooltip("Preço (real):Q", format=".2f")],
    )
    line_pred = alt.Chart(plot_df).mark_line(color="#FFA726").encode(
        x="data:T",
        y=alt.Y("Preço (previsto):Q", title="Preço (previsto)", scale=alt.Scale(domain=domain), axis=alt.Axis(titleColor="#FFA726")),
        tooltip=[alt.Tooltip("data:T", title="Data", format="%d/%m/%Y"), alt.Tooltip("Preço (previsto):Q", format=".2f")],
    )
    st.altair_chart(alt.layer(line_real, line_pred).resolve_scale(y="independent").properties(height=360), use_container_width=True)

    # --- Filtro por ano (interativo) ---
    st.markdown("### 🎚️ Filtro por Ano")
    plot_df["ano"] = plot_df["data"].dt.year
    min_ano, max_ano = int(plot_df["ano"].min()), int(plot_df["ano"].max())

    year_param = alt.param(name="ano_param", value=max_ano, bind=alt.binding_range(name="Ano (filtro):", min=min_ano, max=max_ano, step=1))
    base = alt.Chart(plot_df).add_params(year_param).transform_filter(alt.datum.ano == year_param)

    line_real = base.mark_line(color="#4FC3F7").encode(x=alt.X("data:T", axis=alt.Axis(format="%d.%m.%y", title="")), y=alt.Y("Preço (real):Q", title="Preço (USD)"))
    line_prev = base.mark_line(color="#FFA726").encode(x="data:T", y="Preço (previsto):Q")
    st.altair_chart(alt.layer(line_real, line_prev).properties(height=360, title="Filtro por Ano — Real vs Previsto"), use_container_width=True)

    # --- Resíduos ---
    st.markdown("### 🔎 Análise de Resíduos")
    res_df = plot_df.copy()
    res_df["Resíduo"] = res_df["Preço (real)"] - res_df["Preço (previsto)"]

    res_line = alt.Chart(res_df).mark_line().encode(
        x=alt.X("data:T", axis=alt.Axis(format="%b.%y"), title=""),
        y=alt.Y("Resíduo:Q", title="Resíduo (USD)"),
        tooltip=[alt.Tooltip("data:T", format="%d/%m/%Y"), alt.Tooltip("Resíduo:Q", format=".2f")],
    ).properties(height=220)
    disp = alt.Chart(res_df).mark_circle(opacity=0.5).encode(
        x=alt.X("Preço (real):Q"), y=alt.Y("Preço (previsto):Q"),
        tooltip=[alt.Tooltip("Preço (real):Q", format=".2f"), alt.Tooltip("Preço (previsto):Q", format=".2f")],
    ).properties(height=260)

    st.altair_chart(res_line, use_container_width=True)
    st.altair_chart(disp, use_container_width=True)

    # --- Importância de features (quando disponível) ---
    st.markdown("### 🧠 Importância das Features (Top 5)")
    try:
        importances = pipe.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({"feature": feature_cols, "importancia": importances})
        imp_df["feature_label"] = imp_df["feature"].map(feature_labels_map(feature_cols))
        imp_df = imp_df.sort_values("importancia", ascending=False).head(5)

        bar = alt.Chart(imp_df).mark_bar().encode(
            x=alt.X("importancia:Q", title="Importância"),
            y=alt.Y("feature_label:N", sort="-x", title=""),
            tooltip=["feature_label", alt.Tooltip("importancia:Q", format=".4f")],
        ).properties(height=360)
        st.altair_chart(bar, use_container_width=True)
    except Exception:
        st.info("Importância de features não disponível para este modelo.")
