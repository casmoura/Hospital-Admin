#!/usr/bin/env python3 
#-*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:24:22 2026

@author: casmoura@local.hcpa.ufrgs.br
"""

import streamlit as st
import pandas    as pd
import numpy     as np
import plotly.graph_objects as go
import plotly.express       as px
import os
from   datetime import datetime
from   streamlit.runtime.scriptrunner import get_script_run_ctx
from   scipy.stats import gaussian_kde
import unicodedata

# Modules to import data straight from SIGTAP
import requests
from   bs4 import BeautifulSoup
import re
import zipfile
import io
import concurrent.futures
import ftplib
from datetime import datetime

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def strip_accents(text):
    """Removes accents and converts string to lowercase for flexible search."""
    if not text:
        return ""
    return ''.join(
        c for c in unicodedata.normalize('NFD', str(text))
        if unicodedata.category(c) != 'Mn'
    ).lower()

#------------------------------------------------------------------------------
def format_br(val, decimals=2):
    """Formats Brazilian currency and standard numerical locales safely."""
    if pd.isna(val): 
        return "-"
    if isinstance(val, (float, np.floating)) and not np.isfinite(val):
        return "∞"
    
    try:
        val_float = float(val)
    except (ValueError, TypeError):
        return str(val)

    if decimals == 0:
        return f"{int(round(val_float)):,}".replace(",", ".")

    parts = f"{val_float:,.{decimals}f}".split('.')
    int_part = parts[0].replace(',', '.')
    return f"{int_part},{parts[1]}"

def parse_br_currency(val_str: str) -> float:
    """Converts Brazilian currency strings (e.g., 'R$ 1.234,56') to float numbers."""
    if not val_str:
        return 0.0
    # Strip everything except digits and commas, then convert comma to dot
    clean = re.sub(r'[^\d,]', '', val_str).replace(',', '.')
    try:
        return float(clean)
    except ValueError:
        return 0.0


#==============================================================================
# ===============  Logger Function  ===========================================
#==============================================================================

LOG_FILE = "/home/local.hcpa.ufrgs.br/casmoura/HCPA/Admin/access_logs.csv"


def record_access_log(event_type="SESSION_START", details=""):
    """
    Appends access metrics (timestamp, session ID, client IP, selected filters)
    to a centralized CSV file on the server.
    """
    try:
        # 1. Capture unique Streamlit session ID
        ctx = get_script_run_ctx()
        session_id = ctx.session_id if ctx else "unknown"
        
        # 2. Capture Client IP (retrieves proxy IP if running behind Nginx/Apache)
        try:
            headers = st.context.headers
            client_ip = headers.get("X-Forwarded-For", headers.get("Host", "Localhost")).split(',')[0].strip()
        except Exception:
            client_ip = "Unknown"

        # 3. Assemble log payload
        log_entry = pd.DataFrame([{
            "TIMESTAMP":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "SESSION_ID":   session_id,
            "CLIENT_IP":    client_ip,
            "EVENT_TYPE":   event_type,
            "CONVENIO":     option_plan   if 'option_plan'   in locals() else "N/A",
            "PROCEDIMENTO": selected_proc if 'selected_proc' in locals() else "N/A",
            "DETAILS":      details
        }])

        # 4. Thread-safe append to CSV file
        file_exists = os.path.exists(LOG_FILE)
        log_entry.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)
        
    except Exception:
        # Prevent app crash if log writing ever fails due to disk permissions
        pass

# =============================================================================
# PAGE CONFIGURATION & STYLING
# =============================================================================
# Set page configurations to mimic a premium enterprise application

st.set_page_config(page_title="HCPA Analytics Hub", layout="wide", page_icon="👩🏻‍⚕️")

# Custom CSS to reduce font size for st.metric labels and values
st.markdown("""
    <style>
    /* Change KPI Number Font Size */
    div[data-testid="stMetricValue"] > div {
        font-size: 20px !important;
        font-weight: 700 !important;
    }
    /* Change KPI Title Font Size */
    div[data-testid="stMetricLabel"] label {
        font-size: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING & CACHING
# =============================================================================
# Cache data loading so the dashboard remains lightning fast

@st.cache_data
 
def load_sigtap_mapping():
    """Loads HCPA vs SIGTAP procedure correlation table."""
    file_path = '/home/local.hcpa.ufrgs.br/casmoura/HCPA/Admin/HCPAvsSIGTAP.csv'
    if not os.path.exists(file_path):
        file_path = 'HCPAvsSIGTAP.csv'
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.DataFrame()
 
SIGTAP_FWF_NAMES = [
    'CO_PROCEDIMENTO', 'VL_SH', 'VL_SA', 'VL_SP', 'QT_TEMPO_PERMANENCIA', 'DT_COMPETENCIA'
]
 
# DATASUS has changed the tb_procedimento.txt fixed-width record layout and 
# here we work around it."
SIGTAP_LAYOUT_REGISTRY = {
    336: {'VL_SH': (282, 294), 'VL_SA': (294, 306)},
    330: {'VL_SH': (284, 292), 'VL_SA': (292, 300)},
}
 

def _sigtap_lines_from(file_obj):
    """Reads a path or file-like object as latin1 text and returns its non-blank lines."""
    if isinstance(file_obj, (str, os.PathLike)):
        with open(file_obj, 'rb') as f:
            data = f.read()
    else:
        data = file_obj.read()
    text = data.decode('latin1') if isinstance(data, bytes) else data
    return [ln for ln in text.splitlines() if ln.strip()]
 
 
def _parse_sigtap_fixed_width(file_obj):
    """
    Parses a SIGTAP tb_procedimento.txt fixed-width file into a standardized
    DataFrame. Shared by load_tb_procedimento() (static local file on disk)
    and fetch_sigtap_competence() (freshly downloaded competence zip from
    DATASUS), so both sources are parsed identically and stay comparable.
 
    Detects which known record layout a file uses by its line length (see
    SIGTAP_LAYOUT_REGISTRY) rather than assuming one fixed set of column
    positions for every competence — DATASUS has changed this layout before.
    CO_PROCEDIMENTO, QT_TEMPO_PERMANENCIA, VL_SP and DT_COMPETENCIA are
    stable across the known layouts and are read positionally/from-the-end;
    only VL_SH/VL_SA vary and are looked up per layout.
 
    `file_obj` can be a path (str) or a file-like object (e.g. io.BytesIO).
    """
    lines = _sigtap_lines_from(file_obj)
    if not lines:
        return pd.DataFrame(columns=SIGTAP_FWF_NAMES)
 
    line_length = len(lines[0])
    layout = SIGTAP_LAYOUT_REGISTRY.get(line_length)
    if layout is None:
        st.warning(
            f"Layout do SIGTAP não reconhecido (linhas de {line_length} caracteres — "
            f"esperado {', '.join(str(k) for k in SIGTAP_LAYOUT_REGISTRY)}). "
            "VL_SH e VL_SA podem sair incorretos para esta competência; DT_COMPETENCIA "
            "e VL_SP usam posições relativas ao fim da linha e devem continuar corretos. "
            "Use o diagnóstico de layout para calibrar as novas posições e adicioná-las "
            "a SIGTAP_LAYOUT_REGISTRY."
        )
 
    records = []
    for line in lines:
        sh_raw = line[layout['VL_SH'][0]:layout['VL_SH'][1]] if layout else ""
        sa_raw = line[layout['VL_SA'][0]:layout['VL_SA'][1]] if layout else ""
        records.append((
            line[0:10],
            sh_raw,
            sa_raw,
            line[-30:-18],   # VL_SP — stable across known layouts
            line[266:270],   # QT_TEMPO_PERMANENCIA — stable across known layouts
            line[-6:],       # DT_COMPETENCIA — stable across known layouts
        ))
 
    df = pd.DataFrame(records, columns=SIGTAP_FWF_NAMES)
    df['CO_PROCEDIMENTO'] = (
        df['CO_PROCEDIMENTO'].astype(str).str.strip().str.replace(r'\D', '', regex=True).str.zfill(10)
    )
    for col in ['VL_SH', 'VL_SA', 'VL_SP']:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r'\D', '', regex=True), errors='coerce'
        ).fillna(0) / 100.0
    df['DT_COMPETENCIA'] = df['DT_COMPETENCIA'].astype(str).str.strip().str.replace(r'\D', '', regex=True)
    return df
 
 
@st.cache_data(show_spinner=False)
def load_tb_procedimento():
    """
    Loads local tb_procedimento.txt from SIGTAP.
    """
 
    file_path = (
        '/home/local.hcpa.ufrgs.br/casmoura/HCPA/Admin/tb_procedimento.txt'
    )
 
    if not os.path.exists(file_path):
        file_path = 'tb_procedimento.txt'
 
    if not os.path.exists(file_path):
        return pd.DataFrame()
 
    try:
        return _parse_sigtap_fixed_width(file_path)
    except Exception as e:
        st.warning(
            f"Não foi possível carregar o tb_procedimento.txt local: {e}"
        )
        return pd.DataFrame()    
    
def load_data():
    df = pd.read_csv('/home/local.hcpa.ufrgs.br/casmoura/HCPA/Admin/Cirurgias_2026-v2.csv',
                     parse_dates=['CIRURGIA_DATA', 'DTHR_INICIO_CIRG', 'DTHR_FIM_CIRG', 
                                  'DTHR_INICIO_ANEST', 'DTHR_FIM_ANEST',
                                  'DT_INT_ADMINISTRATIVA', 'DT_ALTA_ADMINISTRATIVA'])

    datetime_cols = ['CIRURGIA_DATA', 'DTHR_INICIO_CIRG', 'DTHR_FIM_CIRG', 
                     'DTHR_INICIO_ANEST', 'DTHR_FIM_ANEST', 
                     'DT_INT_ADMINISTRATIVA', 'DT_ALTA_ADMINISTRATIVA']
    
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

    financial_columns = ['VALOR_CONTA', 'VALOR_OPM','VALOR_TOTAL_NOTA_CONSUMO']
    for col in financial_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('.', '',  regex=False)
                .str.replace(',', '.', regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df['DURACAO_MINUTOS'] = (df['DTHR_FIM_CIRG'] - df['DTHR_INICIO_CIRG']).dt.total_seconds() / 60.0
    df['DURACAO_ANESTESIA_MINUTOS'] = (df['DTHR_FIM_ANEST'] - df['DTHR_INICIO_ANEST']).dt.total_seconds() / 60.0
    #Calculate Hospitalization Duration in Days
    df['TEMPO_INTERNACAO_DIAS'] = (df['DT_ALTA_ADMINISTRATIVA'] - df['DT_INT_ADMINISTRATIVA']).dt.total_seconds() / 86400.0
    df['ANO_MES'] = df['CIRURGIA_DATA'].dt.strftime('%Y-%m')
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Erro ao carregar o arquivo CSV: {e}")
    st.stop()
 
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_sigtap_procedure_details(code: str, month: str, year: str) -> dict:
    """
    Looks out procedure straight from tb_procedimento.txt
    for the selected competence.
    Does not scraps SIGTAP's HTML page
    """

    def _fail(reason: str):
        st.session_state["sigtap_last_error"] = reason
        return None

    # 1. Normalize code
    clean_code = (
        re.sub(r'\D', '', str(code))
        .strip()
        .zfill(10)
    )
    if len(clean_code) != 10:
        return _fail(
            f"Código de procedimento inválido: '{code}'."
        )
    # 2. Normalizes competence
    month = str(month).zfill(2)
    year = str(year)

    if not re.fullmatch(r'\d{4}', year):
        return _fail(
            f"Ano inválido: '{year}'."
        )
    if not re.fullmatch(r'(0[1-9]|1[0-2])', month):
        return _fail(
            f"Mês inválido: '{month}'."
        )
    competence = f"{year}{month}"

    # 3. Download full comptence
    try:
        df = fetch_sigtap_competence(competence)
    except Exception as e:
        return _fail(
            f"Erro ao carregar a competência {competence}: {e}"
        )
    if df is None or df.empty:
        return _fail(
            f"Não foi possível carregar o tb_procedimento.txt "
            f"da competência {month}/{year}."
        )

    # 4. Guarantee code standarzation
    df = df.copy()
    df['CO_PROCEDIMENTO'] = (
        df['CO_PROCEDIMENTO']
        .astype(str)
        .str.strip()
        .str.replace(r'\D', '', regex=True)
        .str.zfill(10)
    )

    # 5. Localize procedure
    match = df[
        df['CO_PROCEDIMENTO'] == clean_code
    ]
    if match.empty:
        return _fail(
            f"O procedimento {clean_code} não foi encontrado "
            f"na competência {month}/{year}."
        )
    row = match.iloc[0]

    # 6. Retrieve values 
    def safe_float(value):
        try:
            value = float(value)
            if np.isfinite(value):
                return value
            return 0.0
        except (TypeError, ValueError):
            return 0.0
    vl_sh = safe_float(row.get('VL_SH', 0.0))
    vl_sa = safe_float(row.get('VL_SA', 0.0))
    vl_sp = safe_float(row.get('VL_SP', 0.0))

    qt_max = row.get(
        'QT_TEMPO_PERMANENCIA',
        0
    )
    try:
        qt_max = int(float(qt_max))
    except (TypeError, ValueError):
        qt_max = 0

    # 7. Success
    st.session_state["sigtap_last_error"] = None
    return {
        "CO_PROCEDIMENTO": clean_code,
        "NO_PROCEDIMENTO": "Obtido do tb_procedimento.txt",
        "VL_SH": vl_sh,
        "VL_SA": vl_sa,
        "VL_SP": vl_sp,
        "QT_MAXIMA_EXECUCAO": qt_max,
        "DT_COMPETENCIA": competence
    }

def fetch_sigtap_batch(procedure_list: list, month: str, year: str) -> list:
    """Executes multi-threaded HTTP scraping to dramatically speed up execution."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_sigtap_procedure_details, code, month, year): code for code in procedure_list}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
    return results

@st.cache_data(show_spinner=False, ttl=3600)
def _download_sigtap_zip_bytes(competence_str):
    """
    Locates and downloads, from the DATASUS anonymous FTP, the official
    TabelaUnificada zip for a given competence (format YYYYMM). DATASUS
    versions these filenames (e.g. TabelaUnificada_202608_v2608141139.zip),
    so this lists the directory instead of guessing the filename.
 
    Returns (zip_buffer: io.BytesIO | None, zip_filename: str | None, error: str | None).
    """
    competence_str = str(competence_str).strip()
 
    if not re.fullmatch(r'\d{6}', competence_str):
        return None, None, f"Competência inválida: '{competence_str}'. Use o formato YYYYMM, por exemplo 202608."
 
    year = competence_str[:4]
    month = competence_str[4:]
    if not (1 <= int(month) <= 12):
        return None, None, f"Mês inválido na competência {competence_str}."
 
    ftp_host = "ftp2.datasus.gov.br"
    ftp_directory = "/public/sistemas/tup/downloads"
    ftp = None
    try:
        ftp = ftplib.FTP()
        ftp.connect(host=ftp_host, port=21, timeout=60)
        ftp.login(user="anonymous", passwd="anonymous@")
        ftp.set_pasv(True)
        ftp.cwd(ftp_directory)
        filenames = ftp.nlst()
 
        pattern = re.compile(rf"^TabelaUnificada_{competence_str}_v(\d+)\.zip$", re.IGNORECASE)
        matching_files = []
        for filename in filenames:
            filename_only = os.path.basename(filename.strip())
            m = pattern.match(filename_only)
            if m:
                matching_files.append((int(m.group(1)), filename_only))
 
        if not matching_files:
            pattern_fallback = re.compile(rf"^TabelaUnificada_{competence_str}.*\.zip$", re.IGNORECASE)
            for filename in filenames:
                filename_only = os.path.basename(filename.strip())
                if pattern_fallback.match(filename_only):
                    matching_files.append((0, filename_only))
 
        if not matching_files:
            return None, None, (
                f"O DATASUS não disponibilizou um arquivo TabelaUnificada "
                f"para a competência {month}/{year}."
            )
 
        matching_files.sort(key=lambda x: x[0], reverse=True)
        zip_filename = matching_files[0][1]
 
        zip_buffer = io.BytesIO()
        ftp.retrbinary(f"RETR {zip_filename}", zip_buffer.write, blocksize=1024 * 1024)
        zip_buffer.seek(0)
 
        if not zipfile.is_zipfile(zip_buffer):
            return None, zip_filename, f"O arquivo recebido do DATASUS ({zip_filename}) não é um ZIP válido."
 
        return zip_buffer, zip_filename, None
 
    except ftplib.all_errors as e:
        return None, None, f"Erro de conexão com o FTP do DATASUS ao baixar a competência {month}/{year}: {e}"
    except Exception as e:
        return None, None, f"Erro ao baixar a competência {month}/{year}: {e}"
    finally:
        if ftp is not None:
            try:
                ftp.quit()
            except Exception:
                try:
                    ftp.close()
                except Exception:
                    pass
 
 
def _find_tb_procedimento_member(z):
    return next(
        (n for n in z.namelist() if n.replace("\\", "/").lower().endswith("tb_procedimento.txt")),
        None
    )
  

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_sigtap_competence(competence_str):
    """
    Downloads a single SIGTAP competence from DATASUS FTP and parses
    tb_procedimento.txt using the SAME fixed-width layout as
    load_tb_procedimento(), via _parse_sigtap_fixed_width().
    competence_str: format YYYYMM, example 202608.
    """
    competence_str = str(competence_str).strip()
    year = competence_str[:4] if len(competence_str) == 6 else "?"
    month = competence_str[4:] if len(competence_str) == 6 else "?"
 
    zip_buffer, zip_filename, error = _download_sigtap_zip_bytes(competence_str)
    if error:
        st.error(error)
        return pd.DataFrame()
 
    try:
        with zipfile.ZipFile(zip_buffer) as z:
            target_file = _find_tb_procedimento_member(z)
            if target_file is None:
                st.error(f"O arquivo {zip_filename} não contém tb_procedimento.txt.")
                return pd.DataFrame()
            with z.open(target_file) as f:
                df = _parse_sigtap_fixed_width(io.BytesIO(f.read()))
 
        if df.empty:
            st.error(f"O tb_procedimento.txt da competência {month}/{year} está vazio.")
            return pd.DataFrame()
 
        df['DT_COMPETENCIA'] = competence_str
        return df
 
    except zipfile.BadZipFile:
        st.error(f"O arquivo da competência {month}/{year} não pôde ser aberto como ZIP.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao processar a competência {month}/{year}: {e}")
        return pd.DataFrame()
            
  
# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.image("/home/local.hcpa.ufrgs.br/casmoura/HCPA/Admin/logo.png", width=200)
st.sidebar.title("Filtros Executivos")

# 1. HEALTH PLAN FILTER
st.sidebar.subheader("💳 Selecione o tipo de convênio")
option_plan = st.sidebar.radio(
    label="Selecione o tipo de convênio", 
    options=["Todos", "SUS", "Bradesco", "CABERGS", "CCG","HCPA-UFRGS","FUNDMEDRS",
             "IPERGS", "MEDSENIOR","NPCRGS","Particular", "Partic. Pesq.",
             "PAS", "RBS", "SULMED" ],
    index=1,
    horizontal=True,
    label_visibility="collapsed"
)

# Apply Health Plan Filter
if option_plan == "SUS":
    df = df_raw[df_raw['CONVENIO'].str.upper() == 'SUS'].copy()
elif option_plan == "Particular":
    df = df_raw[df_raw['CONVENIO'].str.upper() == 'PARTICULAR'].copy()
elif option_plan == "CABERGS":
    df = df_raw[df_raw['CONVENIO'].str.upper() == 'CABERGS'].copy()   
elif option_plan == "CCG":
    df = df_raw[df_raw['CONVENIO'].str.upper() == "CENTRO CLINICO GAUCHO"].copy()   
elif option_plan == "HCPA-UFRGS":
    df = df_raw[df_raw['CONVENIO'].str.upper() == "CONV.FUNCIONARIOS HCPA/UFRGS"].copy()   
elif option_plan == "FUNDMEDRS":
    df = df_raw[df_raw['CONVENIO'].str.upper() == "FUNDACAO MEDICA DO RS"].copy()   
elif option_plan == "IPERGS":
    df = df_raw[df_raw['CONVENIO'].str.upper() == "IPE"].copy()   
elif option_plan == "MEDSENIOR":
    df = df_raw[df_raw['CONVENIO'].str.upper() == "MEDSENIOR"].copy()   
elif option_plan == "NPCRGS":
    df = df_raw[df_raw['CONVENIO'].str.upper() == "NUCLEO DE PESQUISA CLINICA DO RGS LTDA"].copy()   
elif option_plan == "Partic. Pesq.":
    df = df_raw[df_raw['CONVENIO'].str.upper() == "PARTICULAR PESQUISA"].copy()   
elif option_plan == "RBS":
    df = df_raw[df_raw['CONVENIO'].str.upper() == "RBS PARTICIPACOES S.A."].copy()  
elif option_plan == "Bradesco":
    df = df_raw[df_raw['CONVENIO'].str.upper() == "SAUDE BRADESCO"].copy()  
elif option_plan == "PAS":
    df = df_raw[df_raw['CONVENIO'].str.upper() == "SAUDE PAS MEDICINA E ODONTO"].copy()  
elif option_plan == "SULMED":
    df = df_raw[df_raw['CONVENIO'].str.upper() == "SULMED - ASSISTENCIA MEDICA"].copy()  
else:
    df = df_raw.copy()

# ==========================================
# 2. HOSPITAL UNIT FILTER (UNF_SEQ)
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("🏥 Unidade")

# Map friendly UI Acronyms -> Dataset values (numeric codes & string names)
UNIT_MAPPING = {
    "🔪 Bloco Cir.":   [126, "126", "BLOCO CIRÚRGICO"],
    "🩹 CCA":         [131, "131", "C.C.A. - CENTRO CIRÚRGICO AMBULATORIAL", "CENTRO CIRÚRGICO AMBULATORIAL"],
    "🤰🏻 CO":          [  123, "130", "CENTRO DE PROCEDIMENTOS OBSTÉTRICOS"],
    "❤️ UDT 2° andar": [130, "130", "UNID DE DIAGN TERAP CARDIOVASC - EXAMES - BL B 2 ANDAR"]
}

# Display friendly acronyms on the UI
unf_options = ["Todos"] + list(UNIT_MAPPING.keys())

selected_unf = st.sidebar.radio(
    label="Selecione a Unidade:",
    options=unf_options,
    index=0
)

# Apply Unit Filter matching the corresponding dataset codes
if selected_unf != "Todos":
    target_values = UNIT_MAPPING[selected_unf]
    df = df[df['UNF_SEQ'].isin(target_values)].copy()



# =============================================================================
# 3. SIDEBAR CONTROLS - PROCEDURE FILTER & SEARCH
# =============================================================================
all_procedures = sorted(df['PROCEDIMENTO'].dropna().unique().tolist())

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Localizar Procedimento")

# 1. Search Box for filtering the procedure list
search_query = st.sidebar.text_input(
    label            = "Buscar por nome/termo:",
    value            = "",
    placeholder      = "Ex: Apendicectomia, Colecistectomia...",
    label_visibility = "visible"
)

clean_query = strip_accents(search_query.strip())

# 2. Filter procedure list dynamically with accent-insensitive search
if clean_query:
    matching_procs = [
        proc for proc  in all_procedures 
        if clean_query in strip_accents(proc)
    ]
else:
    matching_procs = all_procedures

proc_options = ["Visão Geral (Sistema Completo)"] + matching_procs
    
    
# --- SELECTBOX & DYNAMIC COLOR LOGIC ---
selected_proc = st.sidebar.selectbox(
    "Selecione o Procedimento:",
    options=proc_options
)        
# Define dynamic highlight colors based on selection
if selected_proc != "Visão Geral (Sistema Completo)":
    accent_color = "#1ABC9C"  # Vibrant Teal for active procedure
    bg_color     = "#E8F8F5"  # Light background tint
    border_color = "#16A085"
else:
    accent_color = "#2C3E50"  # Default Slate Navy
    bg_color     = "#F8F9F9"
    border_color = "#BDC3C7"
        
# Inject CSS to style the sidebar selectbox container
st.markdown(f"""
    <style>
    div[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color:               {bg_color} !important;
        border:           2px solid {border_color} !important;
        border-radius:                         8px !important;
    }}
    div[data-testid="stSidebar"] div[data-baseweb="select"] span {{
        color:       {accent_color} !important;
        font-weight:            600 !important;
    }}
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Módulos do Sistema")

# Left panel tab selectors
nav_options = [
    "💰 Análise Financeira",
    "⏱️ Tempo Cirúrgico",
    "💉 Tempo de Anestesia",
    "🛏️ Tempo de Internação",
    "👨‍⚕️ Equipe",
    "🦿 OPME",
    "🛋️ Frequência de Pacientes",
    "📍 Origem dos Pacientes"
]

selected_tab = st.sidebar.radio(
    label   = "Selecione o Módulo Visualizado:",
    options = nav_options,
    index   = 0
)

 
# =============================================================================
# ACCESS LOGGING CONTROLLER
# =============================================================================

# 1. Log New Session / Unique Visit (Fires only once per browser connection)
if "session_logged" not in st.session_state:
    st.session_state["session_logged"] = True
    record_access_log(event_type="NEW_VISIT", details="User initialized dashboard session")

# 2. Log Procedure Filter Usage (Fires whenever a user changes selected procedure)
if "last_selected_proc" not in st.session_state:
     st.session_state["last_selected_proc"]  = selected_proc
elif st.session_state["last_selected_proc"] != selected_proc:
     st.session_state["last_selected_proc"]  = selected_proc
     record_access_log(event_type = "FILTER_CHANGE", 
                       details    = f"User selected: {selected_proc}")
    
# =============================================================================
# MAIN INTERACTIVE DASHBOARD - HEADER & TOP FILTERS
# =============================================================================

filtered_df = df.copy()

col_icon, col_txt = st.columns([0.4, 1.96])
  
with col_icon:
  st.image("/home/local.hcpa.ufrgs.br/casmoura/HCPA/Admin/Logo-CIA.png", width=250)
with col_txt:
    st.markdown(
        "<div style='line-height: 40px; font-size: 30px;'><b>Central de Inteligência Administrativa<b><br> Diretoria Administrativa - HCPA", 
        unsafe_allow_html=True
    )
    

st.markdown(f"**Fonte de Dados: Cirurgias_2006.xls**")
st.markdown(f"**Período Abrangido: jan/2025 - maio/2026**")

col_icon, col_txt = st.columns([0.04, 0.96])
with col_icon:
    st.image("/home/local.hcpa.ufrgs.br/casmoura/HCPA/Admin/page-under-construction.png", width=50)
with col_txt:
    st.markdown(
        "<div style='line-height: 40px; font-size: 15px;'><b>Página em construção.</b> Atualizações ocorrem a cada instante. Contato: <a href='mailto:casmoura@hcpa.edu.br'>casmoura@hcpa.edu.br</a></div>", 
        unsafe_allow_html=True
    )

# --- 1. Filtering & Calculations (Logic Only) ---
if selected_proc  == "Visão Geral (Sistema Completo)":
    title_context  = "Geral (Todos os Procedimentos)"
    kpi_label      = "Valor Total (Geral)"
    consump_metric = filtered_df['VALOR_TOTAL_NOTA_CONSUMO'].sum()
else:
    filtered_df     = filtered_df[filtered_df['PROCEDIMENTO'] == selected_proc]
    title_context   = selected_proc
    kpi_label       = f"Tíquete Médio da Nota de Consumo"
    # Average: Sum of column for selected procedure divided by total execution count
    execution_count = len(filtered_df)
    consump_metric  = (
        filtered_df['VALOR_TOTAL_NOTA_CONSUMO'].sum() / execution_count 
        if execution_count > 0 else 0.0
    )
    
# Formatted currency string (Brazilian format)
formated_value = f"R$ {format_br(consump_metric)}"
# -----------------------------------------------------------------------------

st.markdown("---")

# Dynamic Context Banner on Main Page
st.markdown(
    f"""
    <div style="
        background-color: {bg_color};
        border-left:      6px solid {accent_color};
        padding:          12px 18px;
        border-radius:    6px;
        margin-top:       10px;
        margin-bottom:    20px;
    ">
        <h4 style="
            color:   {accent_color};
            margin:  0; 
            padding: 0;
            ">
            🎯 Contexto Atual: {title_context}
        </h4>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 1. METRIC KPIS (Top Blocks)
# ==========================================

total_surgeries = len(filtered_df)

# -----------------------------------------------------------------------------
# Calculate Margin per Inpatient Day:
#    (VALOR_CONTA - VALOR_TOTAL_NOTA_CONSUMO) / TEMPO_INTERNACAO_DIAS
# Filter for valid stays (> 0 days) to prevent division by zero
valid_int = filtered_df[
    (filtered_df['TEMPO_INTERNACAO_DIAS'] > 0) & 
    (filtered_df['VALOR_CONTA'].notna())
].copy()

if len(valid_int) > 0:
    row_margin_per_day = (
        (valid_int['VALOR_CONTA'].fillna(0) - valid_int['VALOR_TOTAL_NOTA_CONSUMO'].fillna(0)) 
        / valid_int['TEMPO_INTERNACAO_DIAS']
    )
    mean_margin_per_day = row_margin_per_day.mean()
else:
    mean_margin_per_day = 0.0
    
# 2. Calculate Percentage Margin: 
#    ((VALOR_CONTA - VALOR_TOTAL_NOTA_CONSUMO) / VALOR_CONTA) * 100
valid_margin_pct = filtered_df[
    (filtered_df['VALOR_CONTA'] > 0) & 
    (filtered_df['VALOR_CONTA'].notna())
].copy()

if len(valid_margin_pct) > 0:
    row_margin_pct = (
        (valid_margin_pct['VALOR_CONTA'] - valid_margin_pct['VALOR_TOTAL_NOTA_CONSUMO'].fillna(0)) 
        / valid_margin_pct['VALOR_CONTA']
    ) * 100.0
    mean_margin_pct = row_margin_pct.mean()
else:
    mean_margin_pct = 0.0
    
# -----------------------------------------------------------------------------
# KPI Calculations (Row 1: Filtered > 0 | Row 2: All Procedures incl. VALOR_CONTA = 0)
# -----------------------------------------------------------------------------

# --- Row 2 Metrics (KP21-KP26: FULL Population Including VALOR_CONTA = 0) ---
total_surgeries_all = len(filtered_df)

# Count of unbilled procedures (VALOR_CONTA = 0 or NaN)
unbilled_count = len(filtered_df[filtered_df['VALOR_CONTA'].fillna(0) == 0])

# Mean account value considering 0 cost entries
mean_val_all = filtered_df['VALOR_CONTA'].fillna(0).mean() if total_surgeries_all > 0 else 0.0

# Mean OPME considering 0 cost entries
mean_opme_all = filtered_df['VALOR_OPM'].fillna(0).mean() if total_surgeries_all > 0 else 0.0

# Margin per inpatient day across all valid hospital stays (including VALOR_CONTA = 0)
valid_int_all = filtered_df[filtered_df['TEMPO_INTERNACAO_DIAS'] > 0].copy()
if len(valid_int_all) > 0:
    row_margin_per_day_all = (
        (valid_int_all['VALOR_CONTA'].fillna(0) - valid_int_all['VALOR_TOTAL_NOTA_CONSUMO'].fillna(0)) 
        / valid_int_all['TEMPO_INTERNACAO_DIAS']
    )
    mean_margin_per_day_all = row_margin_per_day_all.mean()
else:
    mean_margin_per_day_all = 0.0

# Portfolio Margin % across all procedures (Total Revenue vs Total Consumption)
tot_rev_all = filtered_df['VALOR_CONTA'].fillna(0).sum()
tot_exp_all = filtered_df['VALOR_TOTAL_NOTA_CONSUMO'].fillna(0).sum()
if tot_rev_all > 0:
    mean_margin_pct_all = ((tot_rev_all - tot_exp_all) / tot_rev_all) * 100.0
else:
    mean_margin_pct_all = 0.0

# Consumption metric divided by full count of procedures
if selected_proc == "Visão Geral (Sistema Completo)":
    kpi_label_all = "Valor Total (Geral Incl. R$ 0)"
    consump_metric_all = filtered_df['VALOR_TOTAL_NOTA_CONSUMO'].fillna(0).sum()
else:
    kpi_label_all = "Tíquete Médio Consumo (Geral Incl. R$ 0)"
    consump_metric_all = (
        filtered_df['VALOR_TOTAL_NOTA_CONSUMO'].fillna(0).sum() / total_surgeries_all 
        if total_surgeries_all > 0 else 0.0
    )

formated_value_all = f"R$ {format_br(consump_metric_all)}"


# 0 ---------------------------------------------------------------------------    
# Render 6 KPI cards (Filtered > 0)
kp01, kp02, kp03, kp04, kp05, kp06 = st.columns(6)
with kp01:
    st.metric(
        "Total de Procedimentos", 
        format_br(total_surgeries).split(",")[0],
        help="Coluna de origem: PROCEDIMENTO"
    )
with kp02:
   st.metric(
        "   ", 
        f"   " ,
        help=" "
    )    
    
with kp03:
   st.metric(
        "   ", 
        f"   " ,
        help=" "
    )    
    
with kp04:
   st.metric(
        "   ", 
        f"   " ,
        help=" "
    )    

with kp05:
   st.metric(
        "   ", 
        f"   " ,
        help=" "
    )        
    
with kp06:
   st.metric(
        "   ", 
        f"   " ,
        help=" "
    )        

# 1 ---------------------------------------------------------------------------    
# Render 6 KPI cards (Filtered > 0)
kp11, kp12, kp13, kp14, kp15, kp16 = st.columns(6)

with kp11:
    st.metric(
        "Procedimentos Faturados", 
        format_br(total_surgeries-unbilled_count).split(",")[0],
        help="Coluna de origem: PROCEDIMENTO"
    )

with kp12:
    mean_val = filtered_df[filtered_df['VALOR_CONTA'] > 0]['VALOR_CONTA'].mean()
    st.metric(
        "Tíquete Médio por AIH", 
        f"R$ {format_br(mean_val)}",
        help="Coluna de origem: VALOR_CONTA (Somente > R$ 0)"
    )

with kp13:
    mean_opme = filtered_df[filtered_df['VALOR_OPM'] > 0]['VALOR_OPM'].mean()
    st.metric(
        "Tíquete Médio Faturado de OPME", 
        f"R$ {format_br(mean_opme)}",
        help="Coluna de origem: VALOR_OPM (Somente > R$ 0)"
    )

with kp14:
    st.metric(
        "Margem / Dia de Internação", 
        f"R$ {format_br(mean_margin_per_day)}",
        help="(VALOR_CONTA - VALOR_TOTAL_NOTA_CONSUMO) / TEMPO_INTERNACAO_DIAS (Somente VALOR_CONTA > 0)"
    )
    
with kp15:
    st.metric(
        "Margem (%)", 
        f"{format_br(mean_margin_pct, decimals=1)}%",
        help="((VALOR_CONTA - VALOR_TOTAL_NOTA_CONSUMO) / VALOR_CONTA) * 100 (Somente VALOR_CONTA > 0)"
    )

with kp16:  
    st.metric(
        label=kpi_label, 
        value=formated_value,
        help="VALOR_TOTAL_NOTA_CONSUMO"
    )

# 2 ---------------------------------------------------------------------------
# Render 6 KPI cards (KP21 - KP26: FULL Population Including VALOR_CONTA = 0)
kp21, kp22, kp23, kp24, kp25, kp26 = st.columns(6)

with kp21:
    st.metric(
        "Procedimentos não-faturados", 
        format_br(unbilled_count).split(",")[0],
        help="Quantidade total de procedimentos com VALOR_CONTA igual a R$ 0 (ou sem valor)"
    )

with kp22:
    st.metric(
        "Tíquete Médio AIH (Geral)", 
        f"R$ {format_br(mean_val_all)}",
        help="Média considerando todos os procedimentos (incluindo os com VALOR_CONTA = 0)"
    )

with kp23:
    st.metric(
        "Tíquete Médio OPME (Geral)", 
        f"R$ {format_br(mean_opme_all)}",
        help="Média faturada de OPME dividida pelo total de procedimentos (incluindo zerados)"
    )

with kp24:
    st.metric(
         "   ", 
         f"   " ,
         help=" "
     )

with kp25:
    st.metric(
         "   ", 
         f"   " ,
         help=" "
     )

with kp26:  
    st.metric(
         "   ", 
         f"   " ,
         help=" "
     )
    
    
    
    
# 3 ---------------------------------------------------------------------------
# Render 2 KPI cards across the top row
kp31, kp32, kp33, kp34, kp35, kp36 = st.columns(6)

with kp31:
    mean_dur = filtered_df[filtered_df['DURACAO_MINUTOS'] > 0]['DURACAO_MINUTOS'].mean()
    val_dur = format_br(mean_dur).split(",")[0]
    st.metric(
        "Tempo Médio de Cirurgia", 
        f"{val_dur} min" if val_dur != "-" else "-",
        help=" DTHR_FIM_CIRG - DTHR_INICIO_CIRG"
    )

with kp32:
    mean_anest = filtered_df[filtered_df['DURACAO_ANESTESIA_MINUTOS'] > 0]['DURACAO_ANESTESIA_MINUTOS'].mean()
    val_anest = format_br(mean_anest).split(",")[0]
    st.metric(
        "Tempo Médio de Anestesia", 
        f"{val_anest} min" if val_anest != "-" else "-",
        help=" DTHR_FIM_ANEST - DTHR_INICIO_ANEST"
    )

with kp33:
   st.metric(
        "   ", 
        f"   " ,
        help=" "
    )    
    
with kp34:
   st.metric(
        "   ", 
        f"   " ,
        help=" "
    )    
    
with kp35:
   st.metric(
        "   ", 
        f"   " ,
        help=" "
    )    
    
with kp36:
   st.metric(
        "   ", 
        f"   " ,
        help=" "
    )    
    
   
st.markdown("---")


# =============================================================================
# SIGTAP PROCEDURE MAPPING LOOKUP & DETAILS LOOKUP
# =============================================================================

st.markdown("---")
st.markdown("### 📋 Correspondência e Valores SIGTAP")

if selected_proc == "Visão Geral (Sistema Completo)":
    st.info("Selecione um procedimento na barra lateral para consultar sua correspondência e os valores SIGTAP.")
else:
    df_sigtap = load_sigtap_mapping()

    if df_sigtap.empty:
        st.warning("Não foi possível carregar a tabela HCPAvsSIGTAP.csv.")
    else:
        col_hcpa = df_sigtap.columns[0]
        col_code = df_sigtap.columns[1]
        col_name = df_sigtap.columns[2]

        sigtap_matches = df_sigtap[
            df_sigtap[col_hcpa].astype(str).str.strip().str.upper()
            == selected_proc.strip().upper()
        ].copy()

        if sigtap_matches.empty:
            st.info(
                f"Nenhuma correspondência SIGTAP encontrada para o procedimento "
                f"**{selected_proc}**."
            )
        else:          
            # Competence selector.
            competence_options = [
                f"{year}-{month:02d}"
                for year in [2026, 2025, 2024, 2023]
                for month in range(12, 0, -1)  # Counts down from 12 to 1
            ]

            # Get current competence in "YYYY-MM" format
            current_competence = datetime.now().strftime("%Y-%m")

            # Fallback to index 0 if current date falls outside options
            default_index = (
                competence_options.index(current_competence)
                if current_competence in competence_options
                else 0
            )        
            selected_sigtap_competence = st.selectbox(
                "Selecione abaixo a Competência dos valores SIGTAP",
                options     = competence_options,
#                index       = competence_options.index("2026-08"),
                index=default_index,
                format_func = lambda x: f"{x[5:7]}/{x[:4]}",
                help="Escolha a competência que será usada para buscar os valores SH, SA e SP."
            )

            competence_year = selected_sigtap_competence[:4]
            competence_month = selected_sigtap_competence[5:7]

            sigtap_matches[col_code] = (
                sigtap_matches[col_code]
                .astype(str)
                .str.strip()
                .str.replace(r"\D", "", regex=True)
                .str.zfill(10)
            )

            # Carrega a competência uma única vez para todos os códigos relacionados.
            competence_df = fetch_sigtap_competence(
                f"{competence_year}{competence_month}"
            )

 
            if competence_df is None or competence_df.empty:
                st.warning(
                    f"Não foi possível carregar os dados SIGTAP da competência "
                    f"{competence_month}/{competence_year}."
                )
            else:
                competence_df = competence_df.copy()
                competence_df["CO_PROCEDIMENTO"] = (
                    competence_df["CO_PROCEDIMENTO"]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\D", "", regex=True)
                    .str.zfill(10)
                )
 
                values_df = competence_df[
                    competence_df["CO_PROCEDIMENTO"].isin(sigtap_matches[col_code])
                ].copy()
 
                merged = pd.merge(
                    sigtap_matches,
                    values_df[
                        [
                            "CO_PROCEDIMENTO",
                            "VL_SH",
                            "VL_SA",
                            "VL_SP",
                            "QT_TEMPO_PERMANENCIA"
                        ]
                    ],
                    left_on=col_code,
                    right_on="CO_PROCEDIMENTO",
                    how="left"
                )
 
                for col in ["VL_SH", "VL_SA", "VL_SP"]:
                    merged[col] = pd.to_numeric(
                        merged[col], errors="coerce"
                    ).fillna(0.0)
 
                merged["QT_TEMPO_PERMANENCIA"] = pd.to_numeric(
                    merged["QT_TEMPO_PERMANENCIA"], errors="coerce"
                )
 
                sigtap_display = merged[
                    [
                        col_hcpa,
                        col_code,
                        col_name,
                        "VL_SH",
                        "VL_SA",
                        "VL_SP",
                        "QT_TEMPO_PERMANENCIA"
                    ]
                ].copy()
 
                # Competence is located on the last column.
                sigtap_display["Competência"] = (
                    f"{competence_month}/{competence_year}"
                )
 
                sigtap_display = sigtap_display.rename(columns={
                    col_hcpa: "Procedimento HCPA",
                    col_code: "Código SIGTAP",
                    col_name: "Descrição do Procedimento SIGTAP",
                    "VL_SH": "Serviço Hospitalar (R$)",
                    "VL_SA": "Serviço Ambulatorial (R$)",
                    "VL_SP": "Serviço Profissional (R$)",
                    "QT_TEMPO_PERMANENCIA": "Tempo de Permanência"
                })
 
                st.dataframe(
                    sigtap_display,
                    column_config={
                        "Procedimento HCPA": st.column_config.TextColumn("Procedimento HCPA"),
                        "Código SIGTAP": st.column_config.TextColumn("Código SIGTAP"),
                        "Descrição do Procedimento SIGTAP": st.column_config.TextColumn(
                            "Descrição do Procedimento SIGTAP"
                        ),
                        "Serviço Hospitalar (R$)": st.column_config.NumberColumn(
                            "Serviço Hospitalar (R$)", format="localized" #"R$ %.2f"
                        ),
                        "Serviço Ambulatorial (R$)": st.column_config.NumberColumn(
                            "Serviço Ambulatorial (R$)", format="localized" #"R$ %.2f"
                        ),
                        "Serviço Profissional (R$)": st.column_config.NumberColumn(
                            "Serviço Profissional (R$)", format="localized" #"R$ %.2f"
                        ),
                        "Tempo de Permanência": st.column_config.NumberColumn(
                            "Tempo de Permanência", format="%d"
                        ),
                        "Competência": st.column_config.TextColumn("Competência")
                    },
                    hide_index=True,
                    use_container_width=True
                )
 
                st.caption(
                    "Os valores SH, SA e SP são carregados da competência selecionada. "
                    ) 

# ===========================================================================
# INTERACTIVE PLOTLY HELPER FUNCTIONS
# ===========================================================================

def generate_stats_df(series, col_name='Duração (Minutos)'):
    if len(series) == 0: return None
    return pd.DataFrame({
        'Métrica Estatística': ['Média', 'Mediana', 'Desvio Padrão', 'Percentil 25', 'Percentil 75', 'Percentil 90'],
        col_name: [series.mean(), series.median(), series.std(), series.quantile(0.25), series.quantile(0.75), series.quantile(0.90)]
    })

def render_distribution_plot_interactive(df_target, value_column, title_label, x_label, dist_color, x_limits=None, y_limit=None):
    """Generates an interactive log-scale histogram with exact hover tooltips and an overlaid smooth KDE density curve."""
    df_positive = df_target[
        np.isfinite(df_target[value_column]) & 
        (df_target[value_column] > 0)
    ].copy()
    
    if len(df_positive) > 0:
        df_positive['LOG_VAL'] = np.log10(df_positive[value_column])
        
        if x_limits is not None:
            min_log, max_log = x_limits
        else:
            min_log = np.floor(df_positive['LOG_VAL'].min())
            max_log = np.ceil(df_positive[ 'LOG_VAL'].max())
            
        if min_log == max_log:
            max_log += 1

        bin_edges     = np.linspace(min_log, max_log, 41)
        bin_width     = bin_edges[1] - bin_edges[0]
        counts, edges = np.histogram(df_positive['LOG_VAL'], bins=bin_edges)
        bin_centers   = (edges[:-1] + edges[1:]) / 2

        hover_texts = []
        has_prontuario = 'PRONTUARIO' in df_positive.columns
        
        for i in range(len(counts)):
            low_val = 10**edges[i]
            high_val = 10**edges[i+1]
            
            
             # Identify patients belonging to this specific bin range
            if i == len(counts) - 1:
                bin_mask = (df_positive['LOG_VAL'] >= edges[i]) & (df_positive['LOG_VAL'] <= edges[i+1])
            else:
                bin_mask = (df_positive['LOG_VAL'] >= edges[i]) & (df_positive['LOG_VAL'] < edges[i+1])

            # Extract unique patient IDs if column exists
            if has_prontuario:
                pront_list = df_positive.loc[bin_mask, 'PRONTUARIO'].dropna().astype(str).unique().tolist()
                
                # Truncate if there are too many IDs to keep tooltip readable
                MAX_DISPLAY = 15
                if len(pront_list) > MAX_DISPLAY:
                    displayed_p = pront_list[:MAX_DISPLAY]
                    extra_count = len(pront_list) - MAX_DISPLAY
                    chunks      = [", ".join(displayed_p[k:k+5]) for k in range(0, len(displayed_p), 5)]
                    pront_str   = "<br>".join(chunks) + f"<br><i>(+ {extra_count} outros)</i>"
                elif len(pront_list) > 0:
                    chunks      = [", ".join(pront_list[k:k+5]) for k in range(0, len(pront_list), 5)]
                    pront_str   = "<br>".join(chunks)
                else:
                    pront_str = "Nenhum"
            else:
                pront_str = "N/A"

           
            hover_texts.append(
                f"<b>Faixa de Valor:</b> R$ {format_br(low_val)} - R$ {format_br(high_val)}<br>"
                f"<b>Ocorrências:</b> {counts[i]:,.0f}<br><br>"
                f"<b>Prontuários:</b><br>{pront_str}"
            )

        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x                 = bin_centers,
            y                 = counts,
            width             = bin_width,
            marker_color      = dist_color,
            marker_line_color = "#154360",
            marker_line_width = 1,
            opacity           = 0.70,
            name              = 'Frequência',
            hoverinfo         = 'text',
            hovertext         = hover_texts
        ))

        if len(df_positive) > 1 and df_positive['LOG_VAL'].nunique() > 1:
            try:
                kde   = gaussian_kde(df_positive['LOG_VAL'])
                x_kde = np.linspace(min_log, max_log, 200)
                y_kde = kde(x_kde) * len(df_positive) * bin_width
                
                fig.add_trace(go.Scatter(
                    x         = x_kde,
                    y         = y_kde,
                    mode      = 'lines',
                    name      = 'Curva de Densidade',
                    line      = dict(color='#2C3E50', width=2.5, shape='spline'),
                    hoverinfo = 'skip'
                ))
            except Exception:
                pass

        tick_values = np.arange(min_log, max_log + 1)
        if len(tick_values) > 8:
            tick_values = np.linspace(min_log, max_log, 8)
        tick_texts = [f"R$ {format_br(10**t)}" for t in tick_values]

        fig.update_layout(
            title = dict(text=f" {title_label}", font=dict(size=26, color='#2C3E50')),
            xaxis = dict(
                title    = f"{x_label} (R$ - Escala Logarítmica)",
                tickmode = 'array',
                tickvals = tick_values,
                ticktext = tick_texts,
                range    = [min_log, max_log]
            ),
            yaxis=dict(
                title = "Quantidade de Ocorrências (Frequência)",
                range = [0, y_limit] if y_limit is not None else None
            ),
            template   = 'plotly_white',
            height     = 380,
            margin     = dict(l=20, r=20, t=50, b=20),
            showlegend = False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dados insuficientes (valores finitos maiores que zero) para gerar o gráfico de distribuição.")

def render_trend_plot_interactive(df_target, value_column, bar_color, title_label):
    """Generates an interactive monthly projection graph with hover tooltips using Plotly."""
    monthly_counts = df_target['ANO_MES'].value_counts().sort_index()
    
    if len(monthly_counts) >= 3:
        start_month   = monthly_counts.index[0]
        full_timeline = pd.date_range(start=start_month, end='2026-12', freq='MS').strftime('%Y-%m').tolist()
        
        historical_counts = monthly_counts.values
        num_historical    = len(historical_counts)
        num_total         = len(full_timeline)
        
        x_historical      = np.arange(num_historical)
        x_all             = np.arange(num_total)
        slope, intercept  = np.polyfit(x_historical, historical_counts, 1)
        trend_line        = np.clip(slope * x_all + intercept, a_min=0, a_max=None)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x             = full_timeline[:num_historical],
            y             = historical_counts,
            name          = 'Volume Histórico Real',
            marker_color  = bar_color,
            hovertemplate = '<b>Mês:</b> %{x}<br><b>Procedimentos:</b> %{y:,.0f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x             = full_timeline,
            y             = trend_line,
            mode          = 'lines+markers',
            name          = 'Projeção de Tendência',
            line          = dict(color='#E74C3C', width=3, dash='dash'),
            marker        = dict(size=6),
            hovertemplate = '<b>Mês:</b> %{x}<br><b>Tendência Estimada:</b> %{y:,.1f}<extra></extra>'
        ))

        fig.update_layout(
            title       = dict(text=f'Volume Mensal e Projeção de Tendência - {title_label}', font=dict(size=26, color='#2C3E50')),
            xaxis_title = 'Mês / Ano',
            yaxis       = dict(
                title = 'Quantidade de Procedimentos',
                dtick = 1 if historical_counts.max() <= 10 else None
            ),
            template  = 'plotly_white',
            hovermode = 'x unified',
            height    = 380,
            margin    = dict(l=20, r=20, t=50, b=20),
            legend    = dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dados históricos insuficientes para gerar a projeção estatística deste contexto.")

def render_percentile_bar_interactive(p25, p50, p75, p90, mean_val, 
                                     title="Duração", xlabel="Unidade", 
                                     color_palette=None, is_currency=False, 
                                     unit_label="min", x_max_limit=None,
                                     decimals=0):
    """
    Renders interactive percentile bar chart.
    - Formats time metrics as rounded integers (0 decimals) on annotations and tooltips.
    - Accepts x_max_limit to support fixed scales across modules (e.g., AIH views).
    - 'Média' annotation is vertically centered directly INSIDE the bar strip.
    """
    if color_palette is None:
        color_palette = ['#A3E4D7', '#48C9B0', '#1ABC9C', '#16A085']
        
    fig = go.Figure()

    # Helper function to format as clean integers for time or R$ for currency
    def fmt(val):
        if is_currency:
            return f"R$ {format_br(val, decimals=2)}"
        return f"{int(round(val))}" if decimals == 0 else format_br(val, decimals=decimals)
    
    # Segments tuple: (x0, x1, color, hover_text, legend_name)
    segments = [
        (0,   p25, color_palette[0], f"Até P25: {fmt(p25)} {unit_label}", f"0 - P25 ({fmt(p25)})"),
        (p25, p50, color_palette[1], f"P25 a P50 (Mediana): {fmt(p50)} {unit_label}", f"P25 - P50 ({fmt(p50)})"),
        (p50, p75, color_palette[2], f"P50 a P75: {fmt(p75)} {unit_label}", f"P50 - P75 ({fmt(p75)})"),
        (p75, p90, color_palette[3], f"P75 a P90: {fmt(p90)} {unit_label}", f"P75 - P90 ({fmt(p90)})")
    ]

    # Add each percentile range as a distinct trace with showlegend=True
    for x0, x1, color, text, leg_name in segments:
        fig.add_trace(go.Bar(
            y           = [''],
            x           = [x1 - x0],
            base        = x0,
            orientation = 'h',
            name        = leg_name,
            marker      = dict(color=color),
            hoverinfo   = 'text',
            hovertext   = text,
            showlegend  = True
        ))

   # 1. Add Mean vertical dashed line
    fig.add_vline(
        x          = mean_val, 
        line_dash  = "dash", 
        line_color = "#E74C3C", 
        line_width = 2
    )

    # 2. Add 'Média' badge vertically centered INSIDE the bar strip
    fig.add_annotation(
        x           = mean_val,
        y           = '',
        text        = f"<b>Média: {fmt(mean_val)} {unit_label}</b>",
        showarrow   = False,
        xanchor     = "left",
        yanchor     = "middle",
        xshift      = 5,
        font        = dict(color="#C0392B", size=11),
        bgcolor     = "rgba(255, 255, 255, 0.88)",
        bordercolor = "#E74C3C",
        borderwidth = 1,
        borderpad   = 3
    )

    layout_kwargs = dict(
        title       = title,
        xaxis_title = xlabel,
        barmode     = 'stack',
        template    = 'plotly_white',
        height      = 210,
        margin      = dict(l=20, r=20, t=40, b=20),
        legend      = dict(
                        orientation = "h",
                        yanchor     = "top",
                        y           = -0.4,
                        xanchor     = "center",
                        x           = 0.5,
                        font        = dict(size=11)
                        )
    )

    # Respect x_max_limit if supplied (used in AIH/Comparative views)
    if x_max_limit is not None:
        layout_kwargs['xaxis'] = dict(title=xlabel, range=[0, x_max_limit])

    fig.update_layout(**layout_kwargs)
    st.plotly_chart(fig, use_container_width=True)
    
def render_time_distribution_interactive(df_target, value_column='DURACAO_MINUTOS', title_label='', bar_color='#1ABC9C', unit_label='min'):
    """Generates an interactive duration histogram with exact hover tooltips and an overlaid smooth KDE density curve."""
    s_dur = df_target[
        np.isfinite(df_target[value_column]) & 
        (df_target[value_column] > 0)
    ][value_column]
    
    if len(s_dur) > 0:
        max_dur = float(s_dur.max())
        min_dur = float(s_dur.min())
        
        span = max_dur - min_dur
        if span <= 60:
            bin_width = 5
        elif span <= 180:
            bin_width = 10
        elif span <= 360:
            bin_width = 15
        else:
            bin_width = 30

        bin_edges = np.arange(0, max_dur + bin_width, bin_width)
        if len(bin_edges) < 2:
            bin_edges = np.array([0, max_dur + 10])
            
        counts, edges = np.histogram(s_dur, bins=bin_edges)
        bin_centers = (edges[:-1] + edges[1:]) / 2

        hover_texts = []
        for i in range(len(counts)):
            low_val  = edges[i]
            high_val = edges[i+1]
            hover_texts.append(
                f"<b>Faixa de Duração:</b> {int(low_val)} a {int(high_val)} {unit_label}<br>"
                f"<b>Quantidade de Ocorrências:</b> {counts[i]:,.0f}"
            )

        fig = go.Figure()
        
        # 1. Frequency Histogram Bars
        fig.add_trace(go.Bar(
            x                 = bin_centers,
            y                 = counts,
            width             = bin_width * 0.88,
            marker_color      = bar_color,
            marker_line_color = "#148F77",
            marker_line_width = 1,
            opacity           = 0.70,
            name              = 'Histograma',
            hoverinfo         = 'text',
            hovertext         = hover_texts
        ))

        # 2. Smooth Kernel Density Estimation (KDE) Trend Curve Overlay
        if len(s_dur) > 1 and s_dur.nunique() > 1:
            try:
                kde   = gaussian_kde(s_dur)
                x_kde = np.linspace(0, max_dur, 200)
                y_kde = kde(x_kde) * len(s_dur) * bin_width
                
                fig.add_trace(go.Scatter(
                    x         = x_kde,
                    y         = y_kde,
                    mode      = 'lines',
                    name      = 'Curva de Suavização',
                    line      = dict(color='#2C3E50', width=2.5, shape='spline'),
                    hoverinfo = 'skip'
                ))
            except Exception:
                pass

        fig.update_layout(
            title = dict(
                text=f"Distribuição de Frequência do Tempo - {title_label} - 01/25 a 05/26", 
                font=dict(size=26, color='#2C3E50')
            ),
            xaxis = dict(
                title=f"Duração (em {unit_label.capitalize()})",
                dtick=bin_width * 2 if bin_width * 2 >= 10 else 10,
                range=[0, max_dur * 1.02]
            ),
            yaxis = dict(
                title="Quantidade de Procedimentos (Frequência)"
            ),
            template   = 'plotly_white',
            height     = 380,
            margin     = dict(l=20, r=20, t=50, b=20),
            showlegend = False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dados insuficientes para gerar a distribuição de tempo cirúrgico.")
               
def render_day_integer_distribution_interactive(df_target, value_column='TEMPO_INTERNACAO_DIAS', title_label='', bar_color='#8E44AD'):
    """Generates a histogram with explicit 1-day integer intervals (0-1, 1-2, 2-3... days)."""
    df_clean = df_target[
        np.isfinite(df_target[value_column]) & 
        (df_target[value_column] >= 0)
    ].copy()
    
    if len(df_clean) > 0:
        max_val = df_clean[value_column].max()
        max_day = max(1, int(np.ceil(max_val)))
            
        bin_edges     = np.arange(0, max_day + 1, 1)
        counts, edges = np.histogram(df_clean[value_column], bins=bin_edges)
        bin_centers   = (edges[:-1] + edges[1:]) / 2

        hover_texts = []
        for i in range(len(counts)):
            low_val  = int(edges[i])
            high_val = int(edges[i+1])
            hover_texts.append(
                f"<b>Intervalo:</b> {low_val} a {high_val} dia(s)<br>"
                f"<b>Quantidade de Ocorrências:</b> {counts[i]:,.0f}"
            )

        fig = go.Figure()
        
        # 3. Histogram Bars centered on 1-day intervals
        fig.add_trace(go.Bar(
            x                 = bin_centers,
            y                 = counts,
            width             = 0.9,
            marker_color      = bar_color,
            marker_line_color = "#4A235A",
            marker_line_width = 1,
            opacity           = 0.85,
            name              = 'Frequência',
            hoverinfo         = 'text',
            hovertext         = hover_texts
        ))

        # 4. Overlaid Density Curve
        if len(df_clean) > 1 and df_clean[value_column].nunique() > 1:
            try:
                kde   = gaussian_kde(df_clean[value_column])
                x_kde = np.linspace(0, max_day, 200)
                y_kde = kde(x_kde) * len(df_clean) * 1.0
                
                fig.add_trace(go.Scatter(
                    x         = x_kde,
                    y         = y_kde,
                    mode      = 'lines',
                    name      = 'Curva de Densidade',
                    line      = dict(color='#2C3E50', width=2.5, shape='spline'),
                    hoverinfo = 'skip'
                ))
            except Exception:
                pass

        # 5. Dynamic tick mark spacing (prevents overlap for long stays)
        tick_step = 1 if max_day <= 15 else (2 if max_day <= 30 else 5)
        tick_vals = list(range(0, max_day + 1, tick_step))

        fig.update_layout(
            title = dict(
                text = f"Distribuição do Tempo de Internação (Intervalos de 1 Dia) - {title_label} - 01/25 a 05/26", 
                font = dict(size = 26, color = '#2C3E50')
            ),
            xaxis = dict(
                title    = "Tempo de Internação (em Dias)",
                tickmode = 'array',
                tickvals = tick_vals,
                ticktext = [f"{v} d" for v in tick_vals],
                range    = [-0.5, max_day + 0.5]
            ),
            yaxis      = dict(title="Quantidade de Procedimentos (Frequência)"),
            template   = 'plotly_white',
            height     = 380,
            margin     = dict(l=20, r=20, t=50, b=20),
            showlegend = False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Nenhum dado de internação disponível para renderizar o gráfico.")
        
def render_top_patients_module(df):
    """
    Renders the Patient Frequency & Volume Module:
    1. Key Metrics / KPIs
    2. Distribution Histogram: Number of Uses (X) vs. Patient Frequency (Y)
    3. Top 10 High-Utilizer Patients Bar Chart
    4. Data table expander
    """
    st.markdown("### 🛋️ Frequência e Perfil dos Pacientes")
    st.markdown("Análise de distribuição da quantidade de usos por paciente e ranking de utilização.")
    
    # Validation
    if 'PRONTUARIO' not in df.columns:
        st.error("⚠️ A coluna **PRONTUARIO** não foi encontrada no conjunto de dados.")
        return
    
    # Drop missing IDs and convert to string (prevents Plotly from treating IDs as continuous numbers)
    df_clean               = df.dropna(subset=['PRONTUARIO']).copy()
    df_clean['PRONTUARIO'] = df_clean['PRONTUARIO'].astype(str).str.replace('.0', '', regex=False)
    
    # 1. Calculate frequency table
    patient_counts = (
        df_clean['PRONTUARIO']
        .value_counts()
        .reset_index()
    )
    patient_counts.columns = ['PRONTUARIO', 'Frequência']

    # Get Top 10
    top_10 = patient_counts.head(10)

    # 2. Key Metrics / KPIs
    total_unique    = df_clean['PRONTUARIO'].nunique()
    avg_per_patient = len(df_clean) / total_unique if total_unique > 0 else 0
    max_use         = top_10['Frequência'].max() if not top_10.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Pacientes Únicos", f"{total_unique:,}".replace(",", "."))
    col2.metric("Média de Procedimentos / Paciente", f"{avg_per_patient:.1f}")
    col3.metric("Maior Utilização Individual", f"{max_use} usos")

    st.markdown("---")

    # ==========================================
    # 3. HISTOGRAM: NUMBER OF USES (X) vs FREQUENCY (Y)
    # ==========================================
    use_distribution = (
        patient_counts['Frequência']
        .value_counts()
        .reset_index()
    )
    use_distribution.columns = ['Numero_de_Usos', 'Qtd_Pacientes']
    use_distribution = use_distribution.sort_values(by='Numero_de_Usos').reset_index(drop=True)

    hover_texts = [
        f"<b>Número de Usos:</b> {row['Numero_de_Usos']}<br>"
        f"<b>Quantidade de Pacientes:</b> {row['Qtd_Pacientes']:,}".replace(",", ".")
        for _, row in use_distribution.iterrows()
    ]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        x                 = use_distribution['Numero_de_Usos'],
        y                 = use_distribution['Qtd_Pacientes'],
        marker_color      = '#16A085',
        marker_line_color = '#0E6655',
        marker_line_width = 1,
        opacity           = 0.85,
        name              = 'Pacientes',
        hoverinfo         = 'text',
        hovertext         = hover_texts,
        text              = use_distribution['Qtd_Pacientes'],
        textposition      = 'outside'
    ))

    max_x = use_distribution['Numero_de_Usos'].max()
    fig_hist.update_layout(
        title = dict(
            text = "<b>Distribuição de Frequência: Quantidade de Usos por Paciente - 01/25 a 05/26</b>",
            font = dict(size=26, color='#2C3E50')
        ),
        xaxis = dict(
            title = "Número de Usos (Procedimentos / Atendimentos por Paciente)",
            dtick = 1 if max_x <= 30 else None,
            tick0 = 1
        ),
        yaxis = dict(
            title = "Quantidade de Pacientes (Frequência)"
        ),
        template   ='plotly_white',
        height     = 380,
        margin     = dict(l=20, r=20, t=50, b=20),
        showlegend = False
    )

    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("---")
    
# 3. Horizontal Bar Chart Setup
    # Sort ascending so the #1 top patient appears at the TOP of the horizontal bar chart
    top_10_sorted = top_10.sort_values(by='Frequência', ascending=True)

    fig_top10 = px.bar(
        top_10_sorted,
        x           = 'Frequência',
        y           = 'PRONTUARIO',
        orientation = 'h',
        title       = "<b>Top 10 Prontuários por Volume de Utilização</b>",
        labels      = {'Frequência': 'Quantidade de Ocorrências', 'PRONTUARIO': 'Prontuário'},
        text        = 'Frequência',
        color       = 'Frequência',
        color_continuous_scale = 'Teal'
    )

    fig_top10.update_traces(
        textposition  = 'outside',
        hovertemplate = "<b>Prontuário:</b> %{y}<br><b>Frequência:</b> %{x}<extra></extra>"
    )

    fig_top10.update_layout(
        template='plotly_white',
        height              = 420,
        xaxis_title         = "Número de Procedimentos / Atendimentos",
        yaxis_title         = "Nº do Prontuário",
        coloraxis_showscale = False,
        margin              = dict(l=20, r=40, t=50, b=20)
    )

    # Render Plotly Chart
    st.plotly_chart(fig_top10, use_container_width=True)

    # 5. Detailed Table View
    with st.expander("📋 Ver Tabela do Top 10 Pacientes"):
        st.dataframe(
            top_10.rename(columns={'PRONTUARIO': 'Nº Prontuário', 'Frequência': 'Total de Procedimentos'}),
            use_container_width = True,
            hide_index          = True
        )

#==========================================
# GEOGRAPHIC COORDINATES & ORIGIN MAP MODULE
# ==========================================

@st.cache_data
def load_brazil_cities_coords():
    """
    Loads IBGE coordinates for Brazilian municipalities.
    Prioritizes Rio Grande do Sul (IBGE codigo_uf = 43) when duplicate city names exist across Brazil.
    """
    url = "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv"
    try:
        cities_df               = pd.read_csv(url)
        cities_df['NOME_CLEAN'] = cities_df['nome'].apply(strip_accents)
        cities_df['IS_RS']      = cities_df['codigo_uf'].astype(str) == '43'
        
        unique_cities = (
            cities_df.sort_values(by = ['NOME_CLEAN', 'IS_RS'], ascending=[True, False])
            .drop_duplicates(subset  = ['NOME_CLEAN'], keep='first')
        )
        
        return unique_cities[['nome', 'latitude', 'longitude', 'NOME_CLEAN']]
    except Exception:
        return pd.DataFrame()
    

def render_origin_map_module(df_target):
    """
    Renders an interactive Mapbox plot of Brazil showing solid proportional circles
    for patient origin municipalities based on the PROCEDENCIA column.
    """
    st.markdown("### 📍 Procedência e Origem dos Pacientes")
    st.markdown("Mapeamento geográfico da origem dos pacientes por município brasileiro.")
    
    if 'PROCEDENCIA' not in df_target.columns:
        st.error("⚠️ A coluna **PROCEDENCIA** não foi encontrada no conjunto de dados.")
        return

    # 1. Clean PROCEDENCIA column and group counts
    df_clean                    = df_target.dropna(subset=['PROCEDENCIA']).copy()
    df_clean['PROCEDENCIA_RAW'] = df_clean['PROCEDENCIA'].astype(str).str.strip()
    
    # Extract base city name (handles cases like "PORTO ALEGRE - RS" or "CANOAS/RS")
    df_clean['PROCEDENCIA_BASE'] = (
        df_clean['PROCEDENCIA_RAW']
        .str.split('-').str[0]
        .str.split('/').str[0]
        .str.strip()
    )
    df_clean['PROCEDENCIA_CLEAN'] = df_clean['PROCEDENCIA_BASE'].apply(strip_accents)

    if len(df_clean) == 0:
        st.warning("Nenhum registro de procedência disponível para este contexto.")
        return

    # 2. Group patient frequency by city
    origin_counts = (
        df_clean['PROCEDENCIA_CLEAN']
        .value_counts()
        .reset_index()
    )
    origin_counts.columns = ['NOME_CLEAN', 'Frequência']

    # 3. Match with RS-prioritized IBGE Lat/Lon Coordinates
    coords_df = load_brazil_cities_coords()
    
    if coords_df.empty:
        st.error("Não foi possível carregar a base de coordenadas geográficas dos municípios.")
        return

    map_data = pd.merge(origin_counts, coords_df, on='NOME_CLEAN', how='inner')

    if map_data.empty:
        st.warning("Nenhum município foi localizado no mapa para o filtro selecionado.")
        return

    # 4. KPI Metrics Summary
    total_cities = len(map_data)
    total_mapped_patients = map_data['Frequência'].sum()
    top_city_row = map_data.sort_values(by='Frequência', ascending=False).iloc[0]

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Municípios Atendidos", f"{total_cities:,}".replace(",", "."))
    kpi2.metric("Total de Pacientes Mapeados", f"{total_mapped_patients:,}".replace(",", "."))
    kpi3.metric("Maior Origem", f"{top_city_row['nome']}", f"{top_city_row['Frequência']:,} pac.".replace(",", "."))

    st.markdown("---")

    # 5. Interactive Mapbox Scatter Plot with Crisp Purple Gradient
    fig_map = px.scatter_mapbox(
        map_data,
        lat        = "latitude",
        lon        = "longitude",
        size       = "Frequência",
        color      = "Frequência",
        hover_name = "nome",
        hover_data = {"Frequência": True, "latitude": False, "longitude": False, "NOME_CLEAN": False},
        color_continuous_scale=["#b07dc5", "#9257ab", "#7a3996", "#5a1678"],  # Crisp Purple Gradient
        size_max=38
    )

    fig_map.update_traces(
        marker=dict(opacity=0.90)
    )

    fig_map.update_layout(
        title = dict(
            text = "<b>Distribuição Geográfica da Procedência dos Pacientes (Brasil) - 01/25 a 05/26</b>",
            font = dict(size=26, color='#2C3E50')
        ),
        mapbox = dict(
            style  = "open-street-map",  
            center = {"lat": -30.0346, "lon": -51.2177},
            zoom   = 6
        ),
        template = 'plotly_white',
        height   = 580,
        margin   = dict(l=10, r=10, t=40, b=10)
    )

    # Render with scrollZoom enabled in Streamlit's display config
    st.plotly_chart(
        fig_map, 
        use_container_width=True, 
        config={'scrollZoom': True}
    )

    # 6. Detailed Table View
    with st.expander("📋 Ver Top 10 - Municípios de Origem"):
        table_df = (
            map_data[['nome', 'Frequência']]
            .sort_values(by='Frequência', ascending=False)
            .rename(columns={'nome': 'Município de Origem', 'Frequência': 'Quantidade de Pacientes'})
        )
        st.dataframe(table_df, use_container_width=True, hide_index=True)

# ===========================================================================
# 4. MAIN NAVIGATION CONTROLLER (SIDEBAR DRIVEN)
# ===========================================================================

# --- MODULE 1: ACCOUNT VALUE 💰 ---
if selected_tab == "💰 Análise Financeira":
    st.markdown("### 💰 Análise Financeira")
    st.markdown("#### Volumetria e Parâmetros de Custo Hospitalar (Faturamento vs. Gastos) - 01/25 a 05/26")
    
    df_clean_val = filtered_df[filtered_df['VALOR_CONTA'] > 0].dropna(subset=['VALOR_CONTA'])
    
    s_rec = df_clean_val[df_clean_val['VALOR_CONTA'] > 0]['VALOR_CONTA']
    s_exp = df_clean_val[df_clean_val['VALOR_TOTAL_NOTA_CONSUMO'] > 0]['VALOR_TOTAL_NOTA_CONSUMO']

    if len(s_rec) > 0 or len(s_exp) > 0:
        log_rec = np.log10(s_rec) if len(s_rec) > 0 else pd.Series([], dtype=float)
        log_exp = np.log10(s_exp) if len(s_exp) > 0 else pd.Series([], dtype=float)
        
        all_logs = pd.concat([log_rec, log_exp])
        min_log  = np.floor(all_logs.min())
        max_log  = np.ceil(all_logs.max())
        if min_log == max_log:
            max_log += 1

        global_x_limits = (min_log, max_log)
        
        bin_edges     = np.linspace(min_log, max_log, 41)
        counts_rec, _ = np.histogram(log_rec, bins=bin_edges) if len(log_rec) > 0 else (np.array([0]), None)
        counts_exp, _ = np.histogram(log_exp, bins=bin_edges) if len(log_exp) > 0 else (np.array([0]), None)
        
        global_max_y = max(counts_rec.max() if len(counts_rec) > 0 else 0, 
                           counts_exp.max() if len(counts_exp) > 0 else 0) * 1.15
    else:
        global_x_limits = None
        global_max_y    = None

    st.markdown("### Distribuição de AIH")
    render_distribution_plot_interactive(
        df_clean_val, 'VALOR_CONTA', title_context, 
        x_label    = 'Recebido pelo HCPA', 
        dist_color = '#52BE80', 
        x_limits   = global_x_limits,        
        y_limit    = global_max_y
    )
    
    st.markdown("### Distribuição de Custos")
    render_distribution_plot_interactive(
        df_clean_val, 'VALOR_TOTAL_NOTA_CONSUMO', title_context, 
        x_label    = 'Gasto pelo HCPA', 
        dist_color = '#E74C3C', 
        x_limits   = global_x_limits,      
        y_limit    = global_max_y
    )
    
    if selected_proc != "Visão Geral (Sistema Completo)" and len(df_clean_val) > 0:
        st.markdown(f"##### Distribuição de Percentis do Valor de Faturamento e Gastos para: {selected_proc}")
        
        vp25      = df_clean_val['VALOR_CONTA'].quantile(0.25)
        vp50      = df_clean_val['VALOR_CONTA'].median()
        vp75      = df_clean_val['VALOR_CONTA'].quantile(0.75)
        vp90      = df_clean_val['VALOR_CONTA'].quantile(0.90)
        vmean_val = df_clean_val['VALOR_CONTA'].mean()
        
        vp25_exp      = s_exp.quantile(0.25) if len(s_exp) > 0 else 0
        vp50_exp      = s_exp.median()       if len(s_exp) > 0 else 0
        vp75_exp      = s_exp.quantile(0.75) if len(s_exp) > 0 else 0
        vp90_exp      = s_exp.quantile(0.90) if len(s_exp) > 0 else 0
        vmean_val_exp = s_exp.mean()         if len(s_exp) > 0 else 0

        max_x_limit = max(vp90, vp90_exp, vmean_val, vmean_val_exp) * 1.10
        
        render_percentile_bar_interactive(
            vp25, vp50, vp75, vp90, vmean_val,
            title         = "Valores Recebidos pelo HCPA (R$)",
            xlabel        = "Valor Recebido (R$)",
            color_palette = ['#A9DFBF', '#7DCEA0', '#52BE80', '#27AE60'],
            is_currency   = True,
            x_max_limit   = max_x_limit
        )

        render_percentile_bar_interactive(
            vp25_exp, vp50_exp, vp75_exp, vp90_exp, vmean_val_exp,
            title         = "Valores Gastos pelo HCPA (R$)",
            xlabel        = "Valor Gasto (R$)",
            color_palette = ['#F72C02', '#FF5757', '#FF1C1C', '#FF0000'],
            is_currency   = True,
            x_max_limit   = max_x_limit
        )

    st.markdown("##### Detalhamento Estatístico de Faturamento e Consumo")
    
    if len(df_clean_val) > 0:
        s_income  = df_clean_val['VALOR_CONTA']
        s_outcome = df_clean_val[df_clean_val['VALOR_TOTAL_NOTA_CONSUMO'] > 0]['VALOR_TOTAL_NOTA_CONSUMO']

        stats_val = pd.DataFrame({
            'Métrica Estatística': ['Média', 'Mediana', 'Desvio Padrão', 'Percentil 25 (Q1 - 25%)', 'Percentil 75 (Q3 - 75%)', 'Percentil 90'],
            'Recebido pelo HCPA (R$)': [s_income.mean(), s_income.median(), s_income.std(), s_income.quantile(0.25), s_income.quantile(0.75), s_income.quantile(0.90)],
            'Gasto pelo HCPA (R$)': [s_outcome.mean() if len(s_outcome) > 0 else np.nan, s_outcome.median() if len(s_outcome) > 0 else np.nan, s_outcome.std() if len(s_outcome) > 0 else np.nan, s_outcome.quantile(0.25) if len(s_outcome) > 0 else np.nan, s_outcome.quantile(0.75) if len(s_outcome) > 0 else np.nan, s_outcome.quantile(0.90) if len(s_outcome) > 0 else np.nan]
        })

        stats_val['Recebido pelo HCPA (R$)'] = stats_val['Recebido pelo HCPA (R$)'].apply(lambda x: f"{format_br(x)}" if pd.notna(x) else "-")
        stats_val['Gasto pelo HCPA (R$)'   ] = stats_val['Gasto pelo HCPA (R$)'   ].apply(lambda x: f"{format_br(x)}" if pd.notna(x) else "-")

        stats_val_indexed = stats_val.set_index('Métrica Estatística')
        financial_cols    = ['Recebido pelo HCPA (R$)', 'Gasto pelo HCPA (R$)']

        styled_stats_val = (
            stats_val_indexed.style
            .set_properties(subset=financial_cols, **{'text-align': 'right !important'})
            .set_table_styles([
                {'selector': 'th.col0, th.col1', 'props': [('text-align', 'right !important')]},
                {'selector': 'th.index_name', 'props': [('text-align', 'left !important')]}
            ])
        )
        st.table(styled_stats_val)
    else:
        st.info("Nenhum registro de faturamento ou consumo limpo encontrado para este cruzamento.")

# --- MODULE 2: SURGERY ⏱️ ---
elif selected_tab == "⏱️ Tempo Cirúrgico":
    st.markdown("### ⏱️ Tempo Cirúrgico")
    st.markdown("### 📈 Tendências de Produtividade e Projeções (Até Dez/2026)")
    st.markdown("#### Volumetria e Parâmetros de Tempo Cirúrgico")
    df_clean_dur = filtered_df[filtered_df['DURACAO_MINUTOS'] > 0].dropna(subset=['DURACAO_MINUTOS'])
    
    render_trend_plot_interactive(df_clean_dur, 'DURACAO_MINUTOS', '#38cf91', title_context)
    render_time_distribution_interactive(df_clean_dur, 'DURACAO_MINUTOS', title_context, bar_color='#1ABC9C')
    
    if selected_proc != "Visão Geral (Sistema Completo)" and len(df_clean_dur) > 0:
        st.markdown(f"##### Distribuição de Percentis para: {selected_proc}")
        
        p25      = df_clean_dur['DURACAO_MINUTOS'].quantile(0.25)
        p50      = df_clean_dur['DURACAO_MINUTOS'].median()
        p75      = df_clean_dur['DURACAO_MINUTOS'].quantile(0.75)
        p90      = df_clean_dur['DURACAO_MINUTOS'].quantile(0.90)
        mean_val = df_clean_dur['DURACAO_MINUTOS'].mean()
        
        render_percentile_bar_interactive(
            p25, p50, p75, p90, mean_val,
            title         = "Duração Cirúrgica (Minutos)",
            xlabel        = "Minutos",
            color_palette = ['#A2E8DD', '#76D7C4', '#1ABC9C', '#148F77'],
            is_currency   = False
        )

    stats_dur = generate_stats_df(df_clean_dur['DURACAO_MINUTOS'])
    if stats_dur is not None:
        stats_dur['Duração (Minutos)'] = stats_dur['Duração (Minutos)'].apply(lambda x: format_br(x, decimals=0))
        
        # Display table with hidden index column
        st.dataframe(
            stats_dur, 
            hide_index          = True, 
            use_container_width = True
        )
    else:
        st.info("Nenhum registro de tempo cirúrgico limpo encontrado para este cruzamento.")

# --- MODULE 3: ANESTHESIA 💉 ---
elif selected_tab == "💉 Tempo de Anestesia":
    st.markdown("### 💉 Tempo de Anestesia")
    st.markdown("### 📈 Tendências de Produtividade e Projeções (Até Dez/2026)")
    st.markdown("#### Volumetria e Parâmetros de Tempo de Anestesia")
    df_clean_anest = filtered_df[filtered_df['DURACAO_ANESTESIA_MINUTOS'] > 0].dropna(subset=['DURACAO_ANESTESIA_MINUTOS'])
    
    render_trend_plot_interactive(df_clean_anest, 'DURACAO_ANESTESIA_MINUTOS', '#5DADE2', title_context)
    render_time_distribution_interactive(df_clean_anest, value_column='DURACAO_ANESTESIA_MINUTOS', title_label=title_context, bar_color='#5DADE2')
    
    if selected_proc != "Visão Geral (Sistema Completo)" and len(df_clean_anest) > 0:
        st.markdown(f"##### Distribuição de Percentis de Anestesia para: {selected_proc}")
        
        ap25      = df_clean_anest['DURACAO_ANESTESIA_MINUTOS'].quantile(0.25)
        ap50      = df_clean_anest['DURACAO_ANESTESIA_MINUTOS'].median()
        ap75      = df_clean_anest['DURACAO_ANESTESIA_MINUTOS'].quantile(0.75)
        ap90      = df_clean_anest['DURACAO_ANESTESIA_MINUTOS'].quantile(0.90)
        amean_val = df_clean_anest['DURACAO_ANESTESIA_MINUTOS'].mean()
        
        render_percentile_bar_interactive(
            ap25, ap50, ap75, ap90, amean_val,
            title         = "Duração de Anestesia (Minutos)",
            xlabel        = "Minutos",
            color_palette = ['#AED6F1', '#85C1E9', '#5DADE2', '#2E86C1'],
            is_currency   = False
        )

    stats_anest = generate_stats_df(df_clean_anest['DURACAO_ANESTESIA_MINUTOS'])
    if stats_anest is not None:
        stats_anest['Duração (Minutos)'] = stats_anest['Duração (Minutos)'].apply(lambda x: format_br(x, decimals=0))
        
        # Display table with hidden index column
        st.dataframe(
            stats_anest, 
            hide_index          = True, 
            use_container_width = True
        )
    else:
        st.info("Nenhum registro de tempo anestésico limpo encontrado para este cruzamento.")

# --- MODULE 4: PHYSICIAN PERFORMANCE 👨‍⚕️ ---
elif selected_tab == "👨‍⚕️ Equipe":
    st.markdown("### 👨‍⚕️ Equipe")
    st.markdown("#### Tempo Cirúrgico Mediano e Variabilidade (P25-P75) por Equipe Executante - 01/25 a 05/26")
    
    df_clean_med = filtered_df[filtered_df['DURACAO_MINUTOS'] > 0].dropna(subset=['MEDICO', 'DURACAO_MINUTOS'])
    
    if len(df_clean_med) > 0:
        # Group by Team using Median, P25, and P75 on Duration
        df_med_grouped = df_clean_med.groupby('MEDICO')['DURACAO_MINUTOS'].agg(
            Mediana = 'median',
            P25     = lambda x: x.quantile(0.25),
            P75     = lambda x: x.quantile(0.75),
            Volume  = 'count'
        ).reset_index()

        # Calculate asymmetric error range relative to the Median
        df_med_grouped['err_upper'] = df_med_grouped['P75'] - df_med_grouped['Mediana']
        df_med_grouped['err_lower'] = df_med_grouped['Mediana'] - df_med_grouped['P25']

        df_top10 = df_med_grouped.sort_values(by='Volume', ascending=False).head(10).reset_index(drop=True)
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        df_top10['EQUIPE_ANON'] = [f"Equipe {letters[i]}" for i in range(len(df_top10))]
        df_med_plot = df_top10.sort_values(by='Mediana', ascending=True).reset_index(drop=True)

        fig_med = go.Figure()
        fig_med.add_trace(go.Bar(
            y            = df_med_plot['EQUIPE_ANON'],
            x            = df_med_plot['Mediana'],
            orientation  = 'h',
            marker_color = '#2C3E50',
            opacity      = 0.85,
            error_x      = dict(
                type       = 'data',
                array      = df_med_plot['err_upper'],       # Extends to P75
                arrayminus = df_med_plot['err_lower'],  # Extends down to P25
                color      = '#E74C3C',
                thickness  = 2,
                width      = 5
            ),
            hovertemplate='<b>%{y}</b><br>Tempo Mediano: %{x:.0f} min<br>Faixa (P25-P75): %{customdata[0]:.0f} a %{customdata[1]:.0f} min<extra></extra>',
            customdata=np.stack((df_med_plot['P25'], df_med_plot['P75']), axis=-1)
        ))

        fig_med.update_layout(
            title    = dict(text=f'Top {len(df_med_plot)} Equipes com Maior Volume (Ordenadas por Tempo Mediano)', font=dict(size=14, color='#2C3E50')),
            xaxis    = dict(title='Duração da Cirurgia (em Minutos)', range=[0, None]),
            template = 'plotly_white',
            height   = max(350, len(df_med_plot) * 45),
            margin   = dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_med, use_container_width=True)
        
        st.markdown("##### Detalhamento Estatístico Consolidado")
        df_med_table = df_top10[['EQUIPE_ANON', 'Mediana', 'P25', 'P75', 'Volume']].copy()
        
        df_med_table['Mediana'] = df_med_table['Mediana'].apply(lambda x: f"{format_br(x, decimals=0)} min")
        df_med_table['P25']     = df_med_table['P25'].apply(lambda x: f"{format_br(x, decimals=0)} min")
        df_med_table['P75']     = df_med_table['P75'].apply(lambda x: f"{format_br(x, decimals=0)} min")
        df_med_table['Volume']  = df_med_table['Volume'].apply(lambda x: format_br(x, decimals=0))
        
        df_med_table.columns = ['Equipe', 'Tempo Mediano (P50)', 'Percentil 25 (P25)', 'Percentil 75 (P75)', 'Volume de Procedimentos']
        
        # Table rendered without index column
        st.dataframe(
            df_med_table, 
            hide_index          = True, 
            use_container_width = True
        )
    else:
        st.info("Nenhum registro com informações médicas válidas encontrado para este contexto.")

# --- MODULE 5: OPME PERFORMANCE 🦿 ---

elif selected_tab == "🦿 OPME":
    st.markdown("### 🦿 OPME")
    st.markdown("#### Custo Mediano de OPME e Variabilidade (P25-P75) por Equipe Executante - 01/25 a 05/26")
    
    df_clean_opme = filtered_df[filtered_df['VALOR_OPM'] > 0].dropna(subset=['MEDICO', 'VALOR_OPM'])
    
    if len(df_clean_opme) > 0:
        # Group by Team using Median, P25, and P75 on OPME Values
        df_med_grouped = df_clean_opme.groupby('MEDICO')['VALOR_OPM'].agg(
            Mediana = 'median',
            P25     = lambda x: x.quantile(0.25),
            P75     = lambda x: x.quantile(0.75),
            Volume  = 'count'
        ).reset_index()

        # Calculate asymmetric error range relative to the Median
        df_med_grouped['err_upper'] = df_med_grouped['P75'] - df_med_grouped['Mediana']
        df_med_grouped['err_lower'] = df_med_grouped['Mediana'] - df_med_grouped['P25']

        df_top10 = df_med_grouped.sort_values(by='Volume', ascending=False).head(10).reset_index(drop=True)
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        df_top10['EQUIPE_ANON'] = [f"Equipe {letters[i]}" for i in range(len(df_top10))]
        df_med_plot = df_top10.sort_values(by='Mediana', ascending=True).reset_index(drop=True)

        fig_med = go.Figure()
        fig_med.add_trace(go.Bar(
            y            = df_med_plot['EQUIPE_ANON'],
            x            = df_med_plot['Mediana'],
            orientation  = 'h',
            marker_color = '#2C3E50',
            opacity      = 0.85,
            error_x      = dict(
                type       = 'data',
                array      = df_med_plot['err_upper'],       # Extends to P75
                arrayminus = df_med_plot['err_lower'],  # Extends down to P25
                color      = '#E74C3C',
                thickness  = 2,
                width      = 5
            ),
            hovertemplate='<b>%{y}</b><br>Mediana: R$ %{x:,.2f}<br>Faixa (P25-P75): R$ %{customdata[0]:,.2f} a R$ %{customdata[1]:,.2f}<extra></extra>',
            customdata=np.stack((df_med_plot['P25'], df_med_plot['P75']), axis=-1)
        ))

        fig_med.update_layout(
            title    = dict(text=f'Top {len(df_med_plot)} Equipes em Volume de OPME (Ordenadas por Custo Mediano)', font=dict(size=14, color='#2C3E50')),
            xaxis    = dict(title='Custo de OPME (em R$)', range=[0, None]),
            template = 'plotly_white',
            height   = max(350, len(df_med_plot) * 45),
            margin   = dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_med, use_container_width=True)
        
        st.markdown("##### Detalhamento Estatístico Consolidado")
        df_opme_table = df_top10[['EQUIPE_ANON', 'Mediana', 'P25', 'P75', 'Volume']].copy()
        
        df_opme_table['Mediana'] = df_opme_table['Mediana'].apply(lambda x: f"R$ {format_br(x, decimals=2)}")
        df_opme_table['P25']     = df_opme_table['P25'].apply(lambda x: f"R$ {format_br(x, decimals=2)}")
        df_opme_table['P75']     = df_opme_table['P75'].apply(lambda x: f"R$ {format_br(x, decimals=2)}")
        df_opme_table['Volume']  = df_opme_table['Volume'].apply(lambda x: format_br(x, decimals=0))
        
        df_opme_table.columns = ['Equipe', 'Custo Mediano (P50)', 'Percentil 25 (P25)', 'Percentil 75 (P75)', 'Volume de Procedimentos']
        
        # Display table without the index column
        st.dataframe(
            df_opme_table, 
            hide_index          = True, 
            use_container_width = True
        )
    else:
        st.info("Nenhum registro com informações de OPME válidas encontrado para este contexto.")       

# --- MODULE 6: HOSPITAL STAY / INTERNAÇÃO 🛏️ ---
elif selected_tab == "🛏️ Tempo de Internação":
    st.markdown("### 🛏️ Tempo de Internação (Permanência Hospitalar)")
    st.markdown("### 📈 Tendências de Permanência e Projeções (Até Dez/2026)")
    st.markdown("#### Volumetria e Parâmetros de Tempo de Internação em Dias")
    
    df_clean_int = filtered_df[filtered_df['TEMPO_INTERNACAO_DIAS'] >= 0].dropna(subset=['TEMPO_INTERNACAO_DIAS'])
    
    # 1. Trend Line
    render_trend_plot_interactive(df_clean_int, 'TEMPO_INTERNACAO_DIAS', '#8E44AD', title_context)
    
    # 2. Integer Day Distribution
    render_day_integer_distribution_interactive(
        df_clean_int, 
        value_column = 'TEMPO_INTERNACAO_DIAS', 
        title_label  = title_context, 
        bar_color    = '#8E44AD'
    )
    
    # 3. Percentile Bar Chart
    if selected_proc != "Visão Geral (Sistema Completo)" and len(df_clean_int) > 0:
        st.markdown(f"##### Distribuição de Percentis de Internação para: {selected_proc}")
        
        ip25      = df_clean_int['TEMPO_INTERNACAO_DIAS'].quantile(0.25)
        ip50      = df_clean_int['TEMPO_INTERNACAO_DIAS'].median()
        ip75      = df_clean_int['TEMPO_INTERNACAO_DIAS'].quantile(0.75)
        ip90      = df_clean_int['TEMPO_INTERNACAO_DIAS'].quantile(0.90)
        imean_val = df_clean_int['TEMPO_INTERNACAO_DIAS'].mean()
        
        # Explicitly pass unit_label="dias"
        render_percentile_bar_interactive(
            ip25, ip50, ip75, ip90, imean_val,
            title         = "Duração da Internação (Dias)",
            xlabel        = "Dias",
            color_palette = ['#D7BDE2', '#BB8FCE', '#8E44AD', '#6C3483'],
            is_currency   = False,
            unit_label    = "dias"  # 👈 Hover tooltips now display "dias"
        )

    # 4. Statistics Table
    stats_int = generate_stats_df(df_clean_int['TEMPO_INTERNACAO_DIAS'], col_name='Tempo (Dias)')
    if stats_int is not None:
        # Format numbers as integer days (no decimals)
        stats_int['Tempo (Dias)'] = stats_int['Tempo (Dias)'].apply(lambda x: f"{int(round(x))}")
        # Hide table index column cleanly
        st.dataframe(stats_int, hide_index=True, use_container_width=True)
    else:
        st.info("Nenhum registro de tempo de internação limpo encontrado para este cruzamento.")
        
# --- MODULE 7: FREQUENCY  ---        
elif selected_tab == "🛋️ Frequência de Pacientes":
    render_top_patients_module(df)
    
# --- MODULE 8: PATIENT ORIGIN MAP 📍 ---
elif selected_tab == "📍 Origem dos Pacientes":
    render_origin_map_module(filtered_df)    
    
# =============================================================================
# SUMMARY TABLE BY SECTOR UNIT (WITH FINANCIAL BALANCE & ADVANCED STYLING)
# =============================================================================
if selected_unf == "Todos" and selected_proc == "Visão Geral (Sistema Completo)":
    st.subheader("📊 Resumo por Unidade")

    # 1. Helper function to calculate raw numeric KPIs for any dataframe slice
    def get_kpis_raw(df_sub):
        if df_sub.empty:
            return {
                "Total Recebido (R$)":     np.nan,
                "Total Gasto (R$)":        np.nan,
                "Balanço Financeiro (R$)": np.nan,
                "Total Procedimentos":     0,
                "Tempo Cirurgia (min)":    np.nan,
                "Tempo Anestesia (min)":   np.nan,
                "Tíquete Médio AIH":       np.nan,
                "Tíquete Médio OPME":      np.nan,
                "Margem / Dia (R$)":       np.nan,
                "Margem (%)":              np.nan
            }
    
        # Safely extract numeric series
        dur         = pd.to_numeric(df_sub['DURACAO_MINUTOS'],           errors='coerce')
        anest       = pd.to_numeric(df_sub['DURACAO_ANESTESIA_MINUTOS'], errors='coerce')
        valor_conta = pd.to_numeric(df_sub['VALOR_CONTA'],               errors='coerce')
        valor_opm   = pd.to_numeric(df_sub['VALOR_OPM'], errors='coerce') if 'VALOR_OPM' in df_sub.columns else pd.Series(dtype=float)
        valor_gasto = pd.to_numeric(df_sub['VALOR_TOTAL_NOTA_CONSUMO'],  errors='coerce')
        tempo_int   = pd.to_numeric(df_sub['TEMPO_INTERNACAO_DIAS'],     errors='coerce')

        # 1. Consolidate Financial Totals (bal_col1, bal_col2, bal_col3)
        total_received    = valor_conta.sum(skipna=True)
        total_spent       = valor_gasto.sum(skipna=True)
        financial_balance = total_received - total_spent

        # 2. Volumes and Operational Averages
        total_proc   = len(df_sub)
        mean_dur     = dur[dur                 > 0].mean()
        mean_anest   = anest[anest             > 0].mean()
        mean_val     = valor_conta[valor_conta > 0].mean()
        mean_consumo = valor_gasto[valor_gasto > 0].mean()
        mean_opme    = valor_opm[valor_opm     > 0].mean() if not valor_opm.empty else np.nan
    
        # 3. Margins
        valid_int_mask  = (tempo_int > 0) & valor_conta.notna()
        margin_per_day  = (valor_conta[valid_int_mask] - valor_gasto[valid_int_mask].fillna(0)) / tempo_int[valid_int_mask]
        mean_margin_day = margin_per_day.mean()
    
        valid_pct_mask  = (valor_conta > 0) & valor_conta.notna()
        margin_pct      = ((valor_conta[valid_pct_mask] - valor_gasto[valid_pct_mask].fillna(0)) / valor_conta[valid_pct_mask]) * 100
        mean_margin_pct = margin_pct.mean()

        return {
            "Total Recebido (R$)":        total_received,
            "Total Gasto (R$)":           total_spent,
            "Balanço Financeiro (R$)":    financial_balance,
            "Total Procedimentos":        total_proc,
            "Tempo Cirurgia (min)":       mean_dur,
            "Tempo Anestesia (min)":      mean_anest,
            "Tíquete Médio AIH":          mean_val,
            "Tíquete Médio Nota Consumo": mean_consumo,
            "Tíquete Médio OPME":         mean_opme,
            "Margem / Dia (R$)":          mean_margin_day,
            "Margem (%)":                 mean_margin_pct
        }

    # 2. Build summary dataframe across units
    summary_rows = []
    unit_list = ["Todos"] + list(UNIT_MAPPING.keys())

    for unit_label in unit_list:
        if unit_label == "Todos":
            df_unit = filtered_df
        else:
            target_codes = UNIT_MAPPING[unit_label]
            df_unit      = filtered_df[filtered_df['UNF_SEQ'].isin(target_codes)]

        row_data    = get_kpis_raw(df_unit)
        clean_label = unit_label.replace("🔪 ", "").replace("🩹 ", "").replace("🤰🏻 ", "").replace("❤️ ", "")
        row_data["Unidade"] = clean_label
        summary_rows.append(row_data)

    summary_df = pd.DataFrame(summary_rows).set_index("Unidade")

    # Explicitly ensure column order (Unidade / Setor is index)
    column_order = [
        "Total Recebido (R$)", 
        "Total Gasto (R$)", 
        "Balanço Financeiro (R$)", 
        "Total Procedimentos", 
        "Tempo Cirurgia (min)", 
        "Tempo Anestesia (min)", 
        "Tíquete Médio AIH", 
        "Tíquete Médio Nota Consumo",
        "Tíquete Médio OPME", 
        "Margem / Dia (R$)", 
        "Margem (%)"
    ]
    summary_df = summary_df[column_order]


    # =============================================================================
    # STYLING FUNCTIONS
    # =============================================================================

    def style_rows(row):
        # 1. Distinct styling for the total summary row ("Todos")
        if row.name == "Todos":
            return ['background-color: #E2E6EA; font-weight: bold; border-bottom: 2px solid #2C3E50;'] * len(row)
        
        # 2. Alternating light gray background for zebra striping
        row_idx = summary_df.index.get_loc(row.name)
        if row_idx % 2 == 1:
            return ['background-color: #F4F6F7;'] * len(row)  # Soft light gray
        
        return [''] * len(row)

    def color_financials(series):
        styles = []
        for val in series:
            if pd.isna(val):
                styles.append('')
            elif val > 0:
                styles.append('color: #1E8449; font-weight: bold;')
            elif val < 0:
                styles.append('color: #C0392B; font-weight: bold;')
            else:
                styles.append('')
        return styles

    format_dict = {
        "Total Recebido (R$)":        lambda x: f"R$ {format_br(x)}"           if pd.notna(x) else "-",
        "Total Gasto (R$)":           lambda x: f"R$ {format_br(x)}"           if pd.notna(x) else "-",
        "Balanço Financeiro (R$)":    lambda x: f"R$ {format_br(x)}"           if pd.notna(x) else "-",
        "Total Procedimentos":        lambda x: format_br(x, decimals=0)       if pd.notna(x) else "-",
        "Tempo Cirurgia (min)":       lambda x: f"{int(round(x))} min"         if pd.notna(x) else "-",
        "Tempo Anestesia (min)":      lambda x: f"{int(round(x))} min"         if pd.notna(x) else "-",
        "Tíquete Médio AIH":          lambda x: f"R$ {format_br(x)}"           if pd.notna(x) else "-",
        "Tíquete Médio Nota Consumo": lambda x: f"R$ {format_br(x)}"           if pd.notna(x) else "-",
        "Tíquete Médio OPME":         lambda x: f"R$ {format_br(x)}"           if pd.notna(x) else "-",
        "Margem / Dia (R$)":          lambda x: f"R$ {format_br(x)}"           if pd.notna(x) else "-",
        "Margem (%)":                 lambda x: f"{format_br(x, decimals=1)}%" if pd.notna(x) else "-"
    }

    styled_summary = (
        summary_df.style
        .apply(style_rows, axis = 1)
        .apply(color_financials, subset=["Balanço Financeiro (R$)", "Margem / Dia (R$)", "Margem (%)"], axis=0)
        .format(format_dict)
    )

    st.dataframe(styled_summary, use_container_width=True)
        
# =============================================================================    
#               BAR GRAPH
# =============================================================================

if selected_proc == "Visão Geral (Sistema Completo)":
    # Only render macro benchmarking if "Todos" is selected
    if selected_unf == "Todos":
        st.header("⏱️ Indicador de Desempenho de Tempo por Procedimento e Percentis")
        
        df_global_clean = df[df['DURACAO_MINUTOS'] > 0].dropna(subset=['DURACAO_MINUTOS', 'PROCEDIMENTO'])
        
        min_surgeries_threshold = 5
        procedure_counts = df_global_clean['PROCEDIMENTO'].value_counts()
        valid_procedures = procedure_counts[procedure_counts >= min_surgeries_threshold].index
        
        if len(valid_procedures) > 0:
            bench_data = []
            for proc in valid_procedures:
                proc_series = df_global_clean[df_global_clean['PROCEDIMENTO'] == proc]['DURACAO_MINUTOS']
                bench_data.append({
                    'Procedimento':  proc,
                    'Vol':           len(proc_series),
                    'P25':           proc_series.quantile(0.25),
                    'Mediana (P50)': proc_series.median(),
                    'P75':           proc_series.quantile(0.75),
                    'P90':           proc_series.quantile(0.90)
                    })
        
            df_bench = pd.DataFrame(bench_data)
            
            # 1. Filter Top 10 by volume
            df_bench = df_bench.sort_values(by='Vol', ascending=False).head(10)
            # 2. Sort by Volume ascending so highest volume sits at the top of the horizontal chart
            df_bench = df_bench.sort_values(by='Vol', ascending=True).reset_index(drop=True)
        
            fig_bench = go.Figure()
            
            fig_bench.add_trace(go.Bar(
                y             = df_bench['Procedimento'], 
                x             = df_bench['P90'], 
                orientation   = 'h',
                name          = 'Percentil 90 (Caso Complexo)', 
                marker_color  = '#A2E8DD', 
                opacity       = 0.5,
                text          = [f"  <b>N = {v}</b>" for v in df_bench['Vol']],
                textposition  = 'outside',
                hovertemplate = '<b>%{y}</b><br>P90: %{x:.0f} min<extra></extra>'
            ))
            fig_bench.add_trace(go.Bar(
                y             = df_bench['Procedimento'], x=df_bench['P75'], orientation='h',
                name          = 'Percentil 75', marker_color='#76D7C4', opacity=0.7,
                hovertemplate = '<b>%{y}</b><br>P75: %{x:.0f} min<extra></extra>'
            ))
            fig_bench.add_trace(go.Bar(
                y             = df_bench['Procedimento'], x=df_bench['Mediana (P50)'], orientation='h',
                name          = 'Mediana (Tempo Padrão)', marker_color='#1ABC9C', opacity=1.0,
                hovertemplate = '<b>%{y}</b><br>Mediana (P50): %{x:.0f} min<extra></extra>'
            ))
            fig_bench.add_trace(go.Bar(
                y             = df_bench['Procedimento'], x=df_bench['P25'], orientation='h',
                name          = 'Percentil 25 (Caso Rápido)', marker_color='#148F77', opacity=1.0,
                hovertemplate = '<b>%{y}</b><br>P25: %{x:.0f} min<extra></extra>'
            ))

            fig_bench.update_layout(
                barmode     = 'overlay',
                title       = dict(text='Top 10 Procedimentos mais Frequentes (Ordenados por Volume)', font=dict(size=14, color='#2C3E50')),
                xaxis_title = 'Duração do Procedimento (em Minutos)',
                template    = 'plotly_white',
                height      = max(400, len(df_bench) * 45),
                margin      = dict(l=20, r=20, t=50, b=20),
                legend      = dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig_bench, use_container_width=True)
        else:
            st.info("Dados insuficientes para gerar o gráfico comparativo de percentis por procedimento.")    
            
