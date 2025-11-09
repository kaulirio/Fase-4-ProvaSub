#Import libraries
import streamlit as st
import pandas as pd
import json
import gdown #Use gdown to Access the File
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import gc
import os

#Initialise Streamlit
st.set_page_config(page_title="Sistema de Recomendação de Talentos por Vaga", layout="wide")
# Force garbage collection
gc.collect()

@st.cache_data(ttl=1800) # cache expires after 30 minutes 
def load_json(json_file):    
    df = json.load(json_file)
    return df

# Check whether the JSON files have been loaded into the python application
#if 'df_Vagas' not in globals() and 'df_Applicants' not in the streamlit session():
if 'df_Vagas' not in st.session_state or 'df_Applicants' not in st.session_state:
    # #Imports JSON files from my personal Google Drive (files made public)
    # # Replace with your own FILE_ID
    # file_Prospects  = '1sh88eHjyIp0wXtcRIFozgN064VGOOxEs'
    file_Applicants = '17859ae_Ki5CImI9-1lhJ335GMDW0f2Qr'
    file_Vagas      = '1YKM7yDTzjHJVf82l2RxEx-SuLxFxCxrl'

    # # Download the JSON files
    # gdown.download(f'https://drive.google.com/uc?export=download&id={file_Prospects}', 'prospects.json', quiet=False)
    gdown.download(f'https://drive.google.com/uc?export=download&id={file_Applicants}', 'applicants.json', quiet=False)
    gdown.download(f'https://drive.google.com/uc?export=download&id={file_Vagas}', 'vagas.json', quiet=False)

    # #Load the JSON File into Python
    # with open('prospects.json', 'r') as prospects_file:
    #     data_Prospects = json.load(prospects_file)

    with open('applicants.json', 'r') as applicants_file:
        data_Applicants = load_json(applicants_file) 
        # data_Applicants = json.load(applicants_file)

    with open('vagas.json', 'r') as vagas_file:
        data_Vagas = load_json(vagas_file) 
        # data_Vagas = json.load(vagas_file)


    # # Convert the JSON so that each prospect candidate is represented as a separate row in the DataFrame
    # # -----------------------
    # #prospects.JSON file
    # # -----------------------
    # records = []

    # for prof_id, profile_info in data_Prospects.items():
    #     titulo = profile_info.get("titulo")
    #     modalidade = profile_info.get("modalidade")

    #     for prospect in profile_info.get("prospects", []):
    #         record = {
    #             "id_prospect": prof_id,
    #             "titulo": titulo,
    #             "modalidade": modalidade,
    #             "nome_candidato": prospect.get("nome"),
    #             "codigo_candidato": prospect.get("codigo"),
    #             "situacao_candidado": prospect.get("situacao_candidado"),
    #             "data_candidatura": prospect.get("data_candidatura"),
    #             "ultima_atualizacao": prospect.get("ultima_atualizacao"),
    #             "comentario": prospect.get("comentario"),
    #             "recrutador": prospect.get("recrutador")
    #         }
    #         records.append(record)

    # # Convert to DataFrame
    # df_Prospects = pd.DataFrame(records)


    # -----------------------
    #applicants.JSON file
    # -----------------------
    records = []

    for prof_id, profile_info in data_Applicants.items():
        record = {
            "id_applicant": prof_id
        }

        # Flatten sections
        for section_name, section_data in profile_info.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    record[f"{section_name}__{key}"] = value
            else:
                # Just in case any sections are not dicts (e.g., cv_pt or cv_en directly under profile)
                record[section_name] = section_data

        records.append(record)

    # Convert to DataFrame
    #df_Applicants = pd.DataFrame(records)
    st.session_state.df_Applicants = pd.DataFrame(records)

    #test

    # -----------------------
    #vagas.JSON file
    # -----------------------
    records = []

    for prof_id, profile_info in data_Vagas.items():
        record = {
            "id_vaga": prof_id
        }

        # Flatten sections
        for section_name, section_data in profile_info.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    record[f"{section_name}__{key}"] = value
            else:
                record[section_name] = section_data

        records.append(record)

    # Convert to DataFrame
    #df_Vagas = pd.DataFrame(records)
    st.session_state.df_Vagas = pd.DataFrame(records)


    #Release memory used
    # Clear variables
    del data_Vagas
    del data_Applicants
    # Force garbage collection
    gc.collect()

    
df_Vagas = st.session_state.df_Vagas
df_Applicants = st.session_state.df_Applicants


# Count NaN or empty values per column
empty_counts = (df_Vagas.isnull() | (df_Vagas == '')).sum()

# Identify columns with more than 13.000 missing/empty values
cols_to_drop = empty_counts[empty_counts > 13000].index

# Drop them from the DataFrame
df_Vagas.drop(columns=cols_to_drop, inplace=True)

# Convert date fields to datetime
df_Vagas['informacoes_basicas__data_requicisao'] = pd.to_datetime(df_Vagas['informacoes_basicas__data_requicisao'], format='%d-%m-%Y', errors='coerce' )
df_Vagas['informacoes_basicas__data_inicial'] = pd.to_datetime(df_Vagas['informacoes_basicas__data_inicial'], format='%d-%m-%Y', errors='coerce' )
df_Vagas['informacoes_basicas__data_final'] = pd.to_datetime(df_Vagas['informacoes_basicas__data_final'], format='%d-%m-%Y', errors='coerce' )


#Measure management skills from the applicant database using the "cv_pt" column — 
#applying supervised machine learning with keyword-based scoring via the Scikit-learn (sklearn) library.
#Load and Preprocess Your CV Data
#if 'df_Vagas' not in globals() and 'df_Applicants' not in globals():
if 'df_cvs' not in st.session_state:
    #Load CVs_PT data from the applicants database
    df_cvs = df_Applicants[['id_applicant', 'cv_pt']]  # Keep only the text column

    # Basic text cleaning
    def clean_text(text):
        text = re.sub(r'\n', ' ', text)  # Remove line breaks
        text = re.sub(r'[^a-zA-Z0-9À-ÿ ]', '', text)  # Keep accented characters
        text = text.lower()  # Lowercase

        # Normalize accented characters to plain ASCII (e.g., é → e)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

        return text

    df_cvs['cleaned_cv_pt'] = df_cvs['cv_pt'].apply(clean_text)

    #Define a List of Keywords - Supervised model where the input and output are known
    management_keywords_pt = [
        'gerenciei', 'gerenciou', 'gerenciaram', 'coordenaram', 'coordenou', 'coordenado',
        'supervisionaram', 'supervisionou', 'supervisionado', 'dirigi', 'dirigiu', 'dirigidos',
        'administrei', 'administrou', 'administrados', 'planejei', 'planejou', 'planejados',
        'estruturei', 'estruturou', 'estruturados', 'implementei', 'implementou', 'implementados',
        'desenvolvi', 'desenvolveu', 'desenvolvidos', 'executaram', 'executou', 'executado',
        'avaliai', 'avaliou', 'avaliados', 'treinei', 'treinou', 'treinados', 'recrutei', 'recrutou',
        'recrutados', 'motivou', 'motivados', 'deleguei', 'delegou', 'delegados', 'liderei', 'liderou',
        'liderados', 'inovei', 'inovou', 'inovados', 'conduzi', 'conduziu', 'conduzidos', 'organizamos',
        'organizou', 'organizados', 'resolvi', 'resolveu', 'resolvidos', 'negociei', 'negociou',
        'negociados', 'fiz', 'fez', 'feitos', 'avaliaram', 'avaliou', 'gerou', 'gerados',
        'acompanhamos', 'acompanhou', 'acompanhados', 'formei', 'formou', 'formados', 'controlei',
        'controlou', 'controlados', 'concluí', 'concluiu', 'concluídos', 'comuniquei', 'comunicou',
        'comunicados'
    ]

    #Define a Scoring Function
    def score_management_skills(text):
        # Initialize the score
        score = 0

        # Loop through all management keywords
        for keyword in management_keywords_pt:
            # Check if the keyword appears in the cleaned CV
            if keyword in text:
                # Increment score for each keyword match (can be adjusted to weight different keywords)
                score += 1  # You can customize the score value here (e.g., score += 2 for 'manager')

        return score

    df_cvs['management_score'] = df_cvs['cleaned_cv_pt'].apply(score_management_skills)
    # Normalize the 'management_score' column to a range between 0 and 3
    df_cvs['management_score'] = (df_cvs['management_score'] / 12) * 3

    st.session_state.df_cvs = df_cvs




df_cvs = st.session_state.df_cvs 

# Add management_score column to df_Applicants
df_Applicants = df_Applicants.merge(
    df_cvs[['id_applicant', 'management_score']],
    on='id_applicant',
    how='left'
)

df_Applicants_original = df_Applicants #Used to create the list which populates the Applicants filters





















#Montando a estrutura do dashboard
# -----------------------------
# Título e introdução - HEADER
# -----------------------------
#st.set_page_config(page_title="Sistema de Recomendação de Talentos por Vaga", layout="wide")

# with col1:
#     st.image("https://decision.pt/wp-content/uploads/2019/12/Logo_Decision.png", width=200)

# with col2:
#     st.markdown("## Sistema de Recomendação de Talentos")
#     st.markdown("### Selecione uma vaga na aba filtros para visualizar os candidatos mais compatíveis")

st.markdown("""
          <style>

        
        .stElementContainer{
            /* This is a comment 
background-color: rgb(73,162,252);   */
            }    
        .stSidebar{
           /* background-color: rgb(206,224,254);
            rgb(73, 162, 252)   */

        }
    </style>      
            
<div style="display: flex; align-items: center; background-color:white; color:black;">
    <img src="https://decision.pt/wp-content/uploads/2019/12/Logo_Decision.png" width="150" style="margin-right: 20px;background-color:white">
    <div>
        <h1 style="margin-bottom: 5px;font-color=black;">Sistema de Recomendação de Talentos</h1>
        <h3 style="margin-top: 0;">Selecione uma vaga na aba filtros para visualizar os candidatos mais compatíveis</h3>
    </div>
</div>
""", unsafe_allow_html=True)
    

# st.title("DashVagaoard de Matching entre Vagas e Candidatos")
# st.subheader("Selecione uma vaga na aba filtros para visualizar os candidatos mais compatíveis")

# -----------------------------
# Filtros e seleção - SIDEBAR
# -----------------------------
st.sidebar.header("Selecione a vaga desejada")

#Lista de meses existentes na base de vagas
# Criar nova coluna no formato 'MMM.yyyy'
df_Vagas['mes_ano'] = df_Vagas['informacoes_basicas__data_requicisao'].dt.strftime('%b.%Y')

# Converter para datetime temporariamente (formato: %b.%Y → 'Apr.2019')
lista_meses_ordenada = sorted(
    df_Vagas['mes_ano'].dropna().unique(),
    key=lambda x: pd.to_datetime(x, format='%b.%Y')
)

# Exemplo de seleção de vaga - Lista de vagas
#lista_vagas = sorted(df_Vagas['informacoes_basicas__titulo_vaga'])
#vaga_selecionada = st.sidebar.selectbox("Mês.Ano:", lista_vagas)

# MÊS.ANO FILTER -  Mês.Ano Filtro lateral
mth_selecionado = st.sidebar.selectbox("Mês.Ano:", lista_meses_ordenada)

# Filtrar o dataframe pelo mês selecionado
df_filtrado = df_Vagas[df_Vagas['mes_ano'] == mth_selecionado]
# Gerar lista de vagas com base no mês selecionado
lista_vagas = sorted(df_filtrado['informacoes_basicas__titulo_vaga'].dropna().unique())
# TITULO VAGA FILTER "Titulo da vaga" Filtro lateral
vaga_selecionada = st.sidebar.selectbox("Título da vaga:", lista_vagas)













# -----------------------------
# Análise dos possívels matches da vaga de candidatos da base Applicants 
# -----------------------------
#Load and Preprocess Your CV Data

# Basic text cleaning
def clean_text(text):
    text = re.sub(r'\n', ' ', text)  # Remove line breaks
    text = re.sub(r'[^a-zA-Z0-9À-ÿ ]', '', text)  # Keep accented characters
    text = text.lower()  # Lowercase
    # Normalize accented characters to plain ASCII (e.g., é → e)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

    return text

#lista_vagas = sorted(df_filtrado['informacoes_basicas__titulo_vaga'].dropna().unique())
# TITULO VAGA FILTER "Titulo da vaga" Filtro lateral

# Mostrar o título e o DataFrame da vaga_selecionada
st.markdown(f"### Vaga: {vaga_selecionada}")
# Filtrar o dataframe pelo mês selecionado
df_vaga_display = df_filtrado[df_filtrado['informacoes_basicas__titulo_vaga'] == vaga_selecionada]
# Rename columns for display (adjust as needed)

# List of columns to keep (original column names)
columns_to_keep = [
    "id_vaga",
    "informacoes_basicas__data_requicisao",
    "informacoes_basicas__limite_esperado_para_contratacao",
    "informacoes_basicas__titulo_vaga",
    "informacoes_basicas__vaga_sap",
    "informacoes_basicas__cliente",
    "informacoes_basicas__solicitante_cliente",
    "informacoes_basicas__empresa_divisao",
    "informacoes_basicas__requisitante",
    "informacoes_basicas__prioridade_vaga",
    "informacoes_basicas__origem_vaga",
    "perfil_vaga__estado",
    "perfil_vaga__cidade",
    "perfil_vaga__vaga_especifica_para_pcd",
    "perfil_vaga__nivel_academico",
    "perfil_vaga__nivel_ingles",
    "perfil_vaga__principais_atividades",
    "perfil_vaga__competencia_tecnicas_e_comportamentais"
]

# Select only those columns first
df_vaga_display = df_vaga_display[columns_to_keep].copy()

# Rename the columns
df_vaga_display = df_vaga_display.rename(columns={
    "id_vaga": "ID Vaga",
    "informacoes_basicas__data_requicisao": "Data de Requisição",
    "informacoes_basicas__limite_esperado_para_contratacao": "Limite para Contratação",
    "informacoes_basicas__titulo_vaga": "Título da Vaga",
    "informacoes_basicas__vaga_sap": "É Vaga SAP?",
    "informacoes_basicas__cliente": "Cliente",
    "informacoes_basicas__solicitante_cliente": "Solicitante",
    "informacoes_basicas__empresa_divisao": "Empresa",
    "informacoes_basicas__requisitante": "Requisitante",
    "informacoes_basicas__prioridade_vaga": "Prioridade",
    "informacoes_basicas__origem_vaga": "Origem",
    "perfil_vaga__estado": "Estado",
    "perfil_vaga__cidade": "Cidade",
    "perfil_vaga__vaga_especifica_para_pcd": "PCD",
    "perfil_vaga__nivel_academico": "Nível Acadêmico",
    "perfil_vaga__nivel_ingles": "Inglês",
    "perfil_vaga__principais_atividades": "Principais Atividades",
    "perfil_vaga__competencia_tecnicas_e_comportamentais": "Competências"
})

# Display without the DataFrame index (row number)
st.dataframe(df_vaga_display.reset_index(drop=True), use_container_width=True)



# -----------------------------
# Exibição dos candidatos compatíveis
# -----------------------------
st.markdown(f"### Candidatos compatíveis com a vaga: {vaga_selecionada}")
# if 'local' not in st.session_state:
    # Executa apenas se 'local' possui valor diferente de nulo
    # df_Applicants["match_score"] = 0
    # top_matches = df_Applicants
    # df_Applicants_original = df_Applicants


# LOCAL CANDIDATO FILTER "Titulo da vaga" Filtro acima do gráfico
# Gerar lista com os locais/estados dos candidatos        
lista_local_candidatos = sorted(df_Applicants_original['infos_basicas__local'].dropna().unique())
lista_nivel_academico = sorted(df_Applicants_original['formacao_e_idiomas__nivel_academico'].dropna().unique())
lista_score_gestao = sorted(df_Applicants_original['management_score'].dropna().unique())


col1, col2, col3 = st.columns(3)
with col1:
    # local = st.selectbox("Local candidato:", lista_local_candidatos)
    st.session_state.local = st.selectbox("Local candidato:", lista_local_candidatos)
with col2:
    nivel_academico = st.selectbox("Nível Acadênico:", lista_nivel_academico)
with col3:
    score_gestao = st.selectbox("Score Gestão (0 a 3):", lista_score_gestao)   


if st.session_state.local != "":
    # Executa apenas se 'local' foi declarado e tem valor diferente de vazio    
    df_Vagas['comparison_info'] = df_Vagas['perfil_vaga__competencia_tecnicas_e_comportamentais']
    #Clean special characters
    df_Vagas['comparison_info'] = df_Vagas['comparison_info'].apply(clean_text)

    # Example: compare first job in df_Vagas against all applicants
    job_description = df_Vagas['comparison_info'].iloc[0]
    df_Applicants_original = df_Applicants
    df_Applicants = df_Applicants[df_Applicants['infos_basicas__local'] == st.session_state.local]

    # Combine the job description and all CVs into one list
    texts = [job_description] + df_Applicants['cv_pt'].fillna('').tolist()

    # Convert to TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute cosine similarity between job and each CV
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Add results to df_Applicants
    df_Applicants['match_score'] = similarities

    # Sort to get best-matching candidates
    top_matches = df_Applicants.sort_values(by='match_score', ascending=False)

    top_matches_display = top_matches.rename(columns={
        "id_applicant": "ID Candidato",
        "infos_basicas__telefone_recado": "Telefone Recado",
        "infos_basicas__telefone": "Telefone",
        "infos_basicas__objetivo_profissional": "Objetivo Profissional",
        "infos_basicas__data_criacao": "Data Criação",
        "infos_basicas__inserido_por": "Inserido por",
        "infos_basicas__email": "Email",
        "infos_basicas__local": "Local",
        "infos_basicas__sabendo_de_nos_por": "Soube da vaga por",
        "infos_basicas__data_atualizacao": "Data Atualização",
        "infos_basicas__codigo_profissional": "Código Profissional",
        "infos_basicas__nome": "Nome",
        "informacoes_pessoais__data_aceite": "Data Aceite",
        "informacoes_pessoais__nome": "Nome (Pessoal)",
        "informacoes_pessoais__fonte_indicacao": "Fonte de Indicação",
        "informacoes_pessoais__email": "Email (Pessoal)",
        "informacoes_pessoais__data_nascimento": "Data de Nascimento",
        "informacoes_pessoais__telefone_celular": "Celular",
        "informacoes_pessoais__sexo": "Sexo",
        "informacoes_pessoais__estado_civil": "Estado Civil",
        "informacoes_pessoais__pcd": "PCD",
        "informacoes_pessoais__endereco": "Endereço",
        "informacoes_profissionais__titulo_profissional": "Título Profissional",
        "informacoes_profissionais__area_atuacao": "Área de Atuação",
        "informacoes_profissionais__conhecimentos_tecnicos": "Conhecimentos Técnicos",
        "informacoes_profissionais__remuneracao": "Remuneração",
        "formacao_e_idiomas__nivel_academico": "Nível Acadêmico",
        "formacao_e_idiomas__nivel_ingles": "Inglês",
        "formacao_e_idiomas__nivel_espanhol": "Espanhol",
        "formacao_e_idiomas__outro_idioma": "Outro Idioma",
        "cv_pt": "Currículo",
        "formacao_e_idiomas__cursos": "Cursos",
        "formacao_e_idiomas__ano_conclusao": "Ano de Conclusão",
        "informacoes_pessoais__download_cv": "Download CV",    
        "match_score": "% Compatibilidade Vaga",    
        "management_score": "Score Gestão"
    })

    # List of columns to keep (original column names)
    columns_to_keep = [
        "ID Candidato",
        "Nome",
        "Telefone",
        "Local",    
        "Email",
        "Objetivo Profissional",
        "% Compatibilidade Vaga",
        "Score Gestão",
        "Data Criação",
        "Data Atualização",    
        "Data Aceite",
        "Sexo",    
        "PCD",
        "Título Profissional",
        "Área de Atuação",
        "Nível Acadêmico",
        "Inserido por",
        "Inglês",
        "Currículo"
    ]

    # Select only those columns first
    top_matches_display = top_matches_display[columns_to_keep].copy()

    # Clone original DF to avoid overwriting
    top_matches_display = top_matches_display.copy()

    # 1. Format "% Compatibilidade Vaga" as percentage (e.g., 0.82 → "82.00%")
    if "% Compatibilidade Vaga" in top_matches_display.columns:
        top_matches_display["% Compatibilidade Vaga"] = top_matches_display["% Compatibilidade Vaga"].apply(
            lambda x: f"{x:.2%}" if pd.notnull(x) else ""
        )

    # 2. Format "Data Criação" and "Data Atualização" to "dd.MMM.yyyy"
    for col in ["Data Criação", "Data Atualização", "Data Aceite"]:
        if col in top_matches_display.columns:
            top_matches_display[col] = pd.to_datetime(top_matches_display[col], errors="coerce")
            top_matches_display[col] = top_matches_display[col].dt.strftime('%d.%b.%Y')

    top_matches_display_aux = top_matches_display

    

    # top_matches_display = top_matches_display[top_matches_display['Local'] == local]
    top_matches_display = top_matches_display[top_matches_display['Nível Acadêmico'] == nivel_academico]
    if score_gestao > 0:
        top_matches_display = top_matches_display[top_matches_display['Score Gestão'] == score_gestao]

    # Remove 'local' filter if selected
    if st.session_state.local == "" and nivel_academico == "" and score_gestao == 0:
        top_matches_display = top_matches_display_aux
    # else:
    #     top_matches_display = top_matches_display_aux

    local = st.session_state.local

    
    #Show the dataframe using streamlit object
    st.dataframe(top_matches_display.reset_index(drop=True), use_container_width=True)

