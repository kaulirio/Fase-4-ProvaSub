# ==========================================================
# Importação de bibliotecas necessárias
# ==========================================================
import streamlit as st
from paginas import data_page, eda_page, normalizacao_page, modelos_preditivos_page

# ==========================================================
# Configurações iniciais
# ==========================================================
st.set_page_config(
    page_title="Modelo_Preditivo_Petroleo_ProvaSub",
    page_icon="🛢️",
    layout="wide",
)

# ----------------------------------------------------------
# Mapeamento de páginas (rótulo -> função .show)
# ----------------------------------------------------------
PAGES = {
    "data": ("📥 Atualizar Dados (IPEA)", data_page.show),
    "eda": ("🔎 Explorar Dados", eda_page.show),
    "normalizacao": ("📐 Normalizar Dados", normalizacao_page.show),
    "modelos_preditivos": ("📊 Modelos Preditivos", modelos_preditivos_page.show),
}

# ----------------------------------------------------------
# Estado inicial (evita KeyError)
# ----------------------------------------------------------
st.session_state.setdefault("page", "home")
st.session_state.setdefault("extrair_dados", False)

# ==========================================================
# Cabeçalho
# ==========================================================
st.title("🛢️ Modelo Preditivo – Preço do Petróleo (USD)")
st.markdown("---")

# ==========================================================
# Navegação por botões (responsivos)
# ==========================================================
cols = st.columns([0.2, 1, 1, 1, 1])  # 4 botões + espaçador
for (key, (label, _)), col in zip(PAGES.items(), cols[1:]):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state["page"] = key

st.markdown("---")

# ==========================================================
# Roteamento de páginas
# ==========================================================
if st.session_state["page"] in PAGES:
    _, render = PAGES[st.session_state["page"]]
    render()
else:
    st.info("👆 Selecione uma das opções acima para começar.")
