
# streamlit_app.py — Classificador N1/N2/N3 (VFleets) — UI simplificada + Modo avançado
# ------------------------------------------------------------------------------------------
# Destaques:
# - Leitura robusta de CSV (encoding/separador).
# - Mapeamento de colunas (TIPO, PLACA, MOTORISTA, DATA+HORA ou DATAHORA, e NIVEL).
# - TIPOS canônicos: EXCESSO_VELOCIDADE (filtra NIVEL MEDIO/ALTO), FADIGA_MOTORISTA,
#   SEM_CINTO, BOCEJO, MANUSEIO_CELULAR, POSSIVEL_COLISAO, ULTRAPASSAGEM_ILEGAL, CAMERA_OBSTRUIDA.
# - Janela configurável, threshold configurável, escalonamento diário N1/N2/N3.
# - UI "legível" (datas DD/MM/YYYY HH:MM:SS, nomes amigáveis, Sim/Não).
# - Toggle "Modo avançado" para exibir colunas técnicas e fazer download "raw".
#
# Execução local: streamlit run streamlit_app.py
# ------------------------------------------------------------------------------------------

import io
import unicodedata
from collections import defaultdict, deque
from datetime import timedelta

import numpy as np
import unicodedata
import pandas as pd
import streamlit as st

# -----------------------
# Utilitários
# -----------------------

def try_read_csv(uploaded_file):
    if uploaded_file is None:
        return None
    content = uploaded_file.getvalue()
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    seps = [",", ";", "\t", "|"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc, sep=sep)
                if df.shape[1] >= 3:
                    st.session_state["read_meta"] = {"encoding": enc, "sep": sep}
                    return df
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"Não consegui ler o CSV com encodings/separadores comuns. Último erro: {last_err}")

def strip_accents(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def norm(s: str) -> str:
    s = strip_accents(s).upper().strip()
    s = s.replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "")
    while "__" in s:
        s = s.replace("__", "_")
    return s

CANON_TYPES = [
    "EXCESSO_VELOCIDADE",
    "FADIGA_MOTORISTA",
    "SEM_CINTO",
    "BOCEJO",
    "MANUSEIO_CELULAR",
    "POSSIVEL_COLISAO",
    "ULTRAPASSAGEM_ILEGAL",
    "CAMERA_OBSTRUIDA",
]

ALIASES = {
    "EXCESSO_VELOCIDADE": "EXCESSO_VELOCIDADE",
    "EXCESSO_DE_VELOCIDADE": "EXCESSO_VELOCIDADE",
    "EXCESSO_VEL": "EXCESSO_VELOCIDADE",
    "FADIGA_MOTORISTA": "FADIGA_MOTORISTA",
    "FADIGA": "FADIGA_MOTORISTA",
    "SEM_CINTO": "SEM_CINTO",
    "SEM_CINTO_SEGURANCA": "SEM_CINTO",
    "BOCEJO": "BOCEJO",
    "MANUSEIO_CELULAR": "MANUSEIO_CELULAR",
    "USO_CELULAR": "MANUSEIO_CELULAR",
    "USO_DE_CELULAR": "MANUSEIO_CELULAR",
    "USO_DO_CELULAR": "MANUSEIO_CELULAR",
    "POSSIVEL_COLISAO": "POSSIVEL_COLISAO",
    "POSSIVEL_COLISSAO": "POSSIVEL_COLISAO",
    "ULTRAPASSAGEM_ILEGAL": "ULTRAPASSAGEM_ILEGAL",
    "CAMERA_OBSTRUIDA": "CAMERA_OBSTRUIDA",
}

DEFAULT_PESOS = {
    "FADIGA_MOTORISTA": 100,
    "CAMERA_OBSTRUIDA": 100,
    "EXCESSO_VELOCIDADE": 100,
    "SEM_CINTO": 40,
    "BOCEJO": 10,
    "MANUSEIO_CELULAR": 40,
    "ULTRAPASSAGEM_ILEGAL": 100,
    "POSSIVEL_COLISAO": 100,
}

FRIENDLY_TIPO = {
    "EXCESSO_VELOCIDADE": "Excesso de Velocidade",
    "FADIGA_MOTORISTA": "Fadiga do Motorista",
    "SEM_CINTO": "Sem Cinto",
    "BOCEJO": "Bocejo",
    "MANUSEIO_CELULAR": "Uso de Celular",
    "POSSIVEL_COLISAO": "Possível Colisão",
    "ULTRAPASSAGEM_ILEGAL": "Ultrapassagem Ilegal",
    "CAMERA_OBSTRUIDA": "Câmera Obstruída",
}

def tipo_to_canon(tipo: str):
    t = norm(tipo)
    if t in ALIASES:
        return ALIASES[t]
    if t in CANON_TYPES:
        return t
    for suf in ["_ALTO", "_MEDIO", "_BAIXO"]:
        if t.endswith(suf) and t[:-len(suf)] in CANON_TYPES:
            return t[:-len(suf)]
    return None

def normalize_tipo_weights(df_tipos: pd.DataFrame) -> dict:
    out = {}
    for _, r in df_tipos.iterrows():
        tipo = str(r.get("Tipo", "")).strip()
        try:
            peso = int(r.get("Peso", 0))
        except Exception:
            peso = 0
        canon = tipo_to_canon(tipo) or norm(tipo)
        if canon:
            out[canon] = peso
    return out

def is_missing_motorista(val) -> bool:
    """
    Considera ausente quando:
    - vazio/None/NaN
    - strings comuns: "sem motorista", "sem_motorista", "sem condutor", "-", "s/d", "null", "none"
    """
    if val is None:
        return True
    s = str(val).strip().lower()
    if s in {"", "nan", "none", "null", "-", "s/d"}:
        return True
    s_norm = s.replace("_", " ").replace("-", " ")
    return s_norm in {"sem motorista", "sem condutor", "sem driver"}




from collections import deque
from datetime import timedelta

def recompute_windowed(base: pd.DataFrame, dt_col: str, window_minutes: int, threshold: int, prefer_motorista: bool=True) -> pd.DataFrame:
    """
    - Ordena por data/hora crescente
    - Faz janela deslizante por entidade (preferindo motorista; fallback placa)
    - Zera pontos após cada crítica e continua a contagem (N1->N2->N3...)
    - Preenche 'detalhe' com os eventos que contribuíram para cruzar o limiar
    """
    df = base.copy()
    if dt_col not in df.columns:
        raise KeyError(f"Coluna de data/hora não encontrada: {dt_col}")
    df[dt_col] = pd.to_datetime(df[dt_col], dayfirst=True, errors="coerce")
    df = df.sort_values(dt_col).reset_index(drop=True)

    # Campos de trabalho
    df["soma_na_janela_(min)"] = 0
    if "crítico?" not in df.columns:
        df["crítico?"] = "Não"
    else:
        df["crítico?"] = "Não"
    df["nível"] = ""
    if "detalhe" not in df.columns:
        df["detalhe"] = ""
    else:
        df["detalhe"] = ""
    if "janela_início" not in df.columns:
        df["janela_início"] = pd.NaT
        df["janela_fim"] = pd.NaT

    window = timedelta(minutes=window_minutes)

    # Estado por entidade
    states = {}

    for idx, row in df.iterrows():
        ent = choose_entity(row.get("motorista", ""), row.get("placa",""), prefer_motorista=prefer_motorista)
        t = row[dt_col]
        try:
            peso = int(row.get("peso", 0)) if pd.notna(row.get("peso", 0)) else 0
        except Exception:
            try:
                peso = int(float(row.get("peso", 0)))
            except Exception:
                peso = 0
        tipo = str(row.get("tipo",""))

        if ent not in states:
            states[ent] = {
                "dq": deque(),  # (timestamp, peso, tipo, idx)
                "sum": 0,
                "level": 0,
            }
        st = states[ent]

        # expirar eventos fora da janela
        t_min = t - window
        while st["dq"] and st["dq"][0][0] < t_min:
            old_t, old_w, _, _ = st["dq"].popleft()
            st["sum"] -= old_w

        # adicionar evento atual
        st["dq"].append((t, peso, tipo, idx))
        st["sum"] += peso

        # limites da janela
        win_start = st["dq"][0][0] if st["dq"] else t
        win_end = win_start + window

        df.at[idx, "soma_na_janela_(min)"] = st["sum"]
        df.at[idx, "janela_início"] = win_start
        df.at[idx, "janela_fim"] = win_end

        if st["sum"] >= threshold:
            st["level"] += 1
            df.at[idx, "crítico?"] = "Sim"
            df.at[idx, "nível"] = f"N{st['level']}"
            # montar detalhe com os eventos que compõem a janela
            detalhes = []
            for t_ev, w_ev, tipo_ev, idx_ev in list(st["dq"]):
                try:
                    ts = t_ev.strftime("%H:%M:%S")
                except Exception:
                    ts = str(t_ev)
                detalhes.append(f"{ts} {tipo_ev} [{placa_ev}] (+{w_ev})")
            df.at[idx, "detalhe"] = " | ".join(detalhes)
            # reset após crítica
            st["dq"].clear()
            st["sum"] = 0
        else:
            df.at[idx, "detalhe"] = ""

    # formato final
    if pd.api.types.is_datetime64_any_dtype(df["janela_início"]):
        df["janela_início"] = df["janela_início"].dt.strftime("%d/%m/%Y %H:%M:%S")
    if pd.api.types.is_datetime64_any_dtype(df["janela_fim"]):
        df["janela_fim"] = df["janela_fim"].dt.strftime("%d/%m/%Y %H:%M:%S")
    return df


def choose_entity(motorista: str, placa: str, prefer_motorista=True) -> str:
    mot = (motorista or "").strip()
    pla = (placa or "").strip()
    if prefer_motorista and not is_missing_motorista(mot):
        return f"MOTORISTA::{mot}"
    # Fallback explícito para PLACA quando o motorista está ausente/é "Sem motorista"
    if not is_missing_motorista(pla):
        return f"PLACA::{pla}"
    return "DESCONHECIDO"

def guess_col(df: pd.DataFrame, candidates: list[str]):
    if df is None:
        return None
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in df.columns:
        for cand in candidates:
            if c.replace(" ", "").lower() == cand.replace(" ", "").lower():
                return c
    return None

# -----------------------
# Núcleo de classificação
# -----------------------

def classificar_eventos(
    df: pd.DataFrame,
    col_tipo: str,
    col_data: str | None,
    col_hora: str | None,
    col_datahora: str | None,
    col_placa: str,
    col_motorista: str,
    col_nivel: str | None,
    pesos: dict,
    janela_min: int = 30,
    threshold: int = 100,
    prefer_motorista: bool = True,
):
    # 1) Base alinhada (inclui NIVEL antes de filtros)
    base = pd.DataFrame({
        "tipo_raw": df[col_tipo].astype(str),
        "placa": df[col_placa].astype(str).str.strip(),
        "motorista": df[col_motorista].astype(str).str.strip(),
    })
    if col_nivel and col_nivel in df.columns:
        base["nivel_evento"] = df[col_nivel].astype(str).str.upper().str.strip()
    else:
        base["nivel_evento"] = None

    # 2) Datetime completo
    if col_datahora:
        dt = pd.to_datetime(df[col_datahora], errors="coerce", dayfirst=True, infer_datetime_format=True)
    else:
        data = df[col_data].astype(str).str.strip() if col_data else ""
        hora = df[col_hora].astype(str).str.strip() if col_hora else ""
        dt = pd.to_datetime(data + " " + hora, errors="coerce", dayfirst=True, infer_datetime_format=True)
    base["data_hora"] = dt

    # 3) Tipo canônico e filtros
    base["tipo_canon"] = [tipo_to_canon(x) for x in base["tipo_raw"]]
    base = base[base["tipo_canon"].notna()].copy()
    base = base[~base["data_hora"].isna()].copy()

    # Filtro nível para excesso de velocidade
    mask_vel = base["tipo_canon"].eq("EXCESSO_VELOCIDADE")
    if "nivel_evento" in base.columns:
        mask_nivel_ok = (~mask_vel) | (base["nivel_evento"].isin(["MEDIO", "MÉDIO", "ALTO"]))
        base = base[mask_nivel_ok].copy()

    # 4) Pesos e entidade
    base["peso"] = base["tipo_canon"].apply(lambda c: int(pesos.get(c, DEFAULT_PESOS.get(c, 0))))
    base["entidade"] = base.apply(lambda r: choose_entity(r["motorista"], r["placa"], prefer_motorista), axis=1)
    # Garantia extra: quando Motorista == "Sem Identificação", força fallback por PLACA
    mask_sem_id = base["motorista"].astype(str).str.strip() == "Sem Identificação"
    base.loc[mask_sem_id, "entidade"] = "PLACA::" + base.loc[mask_sem_id, "placa"].astype(str).str.strip()


    # 5) Ordenar
    base = base.sort_values(["entidade", "data_hora"]).reset_index(drop=True)

    # 6) Janela e disparos
    base["window_total"] = np.nan
    base["critico"] = False
    base["nivel"] = None
    base["janela_inicio"] = pd.NaT
    base["janela_fim"] = pd.NaT
    base["ocorrencia_idx_no_dia"] = np.nan
    base["detalhe"] = None

    window = timedelta(minutes=janela_min)
    deques = defaultdict(deque)
    sum_in_window = defaultdict(int)
    block_until = defaultdict(lambda: pd.Timestamp.min)
    counts_per_day = defaultdict(int)

    for idx, row in base.iterrows():
        ent = row["entidade"]
        t = row["data_hora"]
        w = int(row["peso"]) if not pd.isna(row["peso"]) else 0

        is_blocked = t <= block_until[ent]

        deques[ent].append((t, w, row['tipo_canon'], row['placa'], row['motorista']))
        sum_in_window[ent] += w

        t_min = t - window
        while deques[ent] and deques[ent][0][0] < t_min:
            old_t, old_w, _, _, _ = deques[ent].popleft()
            sum_in_window[ent] -= old_w

        win_start = deques[ent][0][0] if deques[ent] else t
        win_end = win_start + window
        base.at[idx, "window_total"] = float(sum_in_window[ent])
        base.at[idx, "janela_inicio"] = win_start
        base.at[idx, "janela_fim"] = win_end

        
        if not is_blocked and sum_in_window[ent] >= threshold:
            # incrementa nível por dia/entidade
            day_key = (ent, t.date())
            counts_per_day[day_key] += 1
            occ_n = counts_per_day[day_key]
            nivel = "N1" if occ_n == 1 else ("N2" if occ_n == 2 else "N3")

            base.at[idx, "critico"] = True
            base.at[idx, "nivel"] = nivel
            base.at[idx, "ocorrencia_idx_no_dia"] = occ_n

            # detalhar eventos que contribuíram (conteúdo atual do deque)
            detalhes = []
            for t_ev, w_ev, tipo_ev, placa_ev, mot_ev in list(deques[ent]):
                try:
                    ts = t_ev.strftime("%H:%M:%S")
                except Exception:
                    ts = str(t_ev)
                detalhes.append(f"{ts} {tipo_ev} [{placa_ev}] (+{w_ev})")
            base.at[idx, "detalhe"] = " | ".join(detalhes)

            # reset imediato dos pontos após crítica (deques e soma)
            deques[ent].clear()
            sum_in_window[ent] = 0
            # não manter bloqueio; começamos a contar imediatamente
            block_until[ent] = pd.Timestamp.min


    resumo = (
        base.loc[base["critico"]]
        .assign(data=lambda x: x["data_hora"].dt.date)
        .groupby(["data", "entidade", "nivel"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values(["data", "entidade"])
    )

    return base, resumo

# -----------------------
# Helpers de apresentação
# -----------------------

FRIENDLY_COLS_RESULTS = [
    "Data/Hora","Tipo","Peso","Motorista","Placa",
    "Soma na Janela (min)","Crítico?","Nível","Janela Início","Janela Fim","Detalhe"
]

def entidade_simplificada(ent: str) -> str:
    if not isinstance(ent, str):
        return ent
    if ent.startswith("MOTORISTA::"):
        return ent.replace("MOTORISTA::", "Motorista: ")
    if ent.startswith("PLACA::"):
        return ent.replace("PLACA::", "Placa: ")
    return ent

def format_datetime_series(s: pd.Series) -> pd.Series:
    return s.dt.strftime("%d/%m/%Y %H:%M:%S")

def format_resultados_for_display(df: pd.DataFrame, janela_minutos: int) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("data_hora").reset_index(drop=True)
    # Ordenar por data/hora crescente SEMPRE
    out = out.sort_values("data_hora").reset_index(drop=True)
    out["Data/Hora"] = format_datetime_series(out["data_hora"])
    out["Janela Início"] = format_datetime_series(out["janela_inicio"])
    out["Janela Fim"] = format_datetime_series(out["janela_fim"])
    out["Tipo"] = out["tipo_canon"].map(FRIENDLY_TIPO).fillna(out.get("tipo_canon"))
    out["Peso"] = out["peso"].astype("Int64")
    out["Motorista"] = out["motorista"].apply(lambda x: "" if is_missing_motorista(x) else str(x).strip())
    out["Placa"] = out["placa"].replace({"nan": ""})
    out["Soma na Janela (min)"] = out["window_total"].round(0).astype("Int64")
    out["Crítico?"] = out["critico"].map({True: "Sim", False: "Não"})
    out["Nível"] = out["nivel"].fillna("")
    out["Detalhe"] = out["detalhe"].fillna("")
    cols = [
        "Data/Hora","Tipo","Peso","Motorista","Placa",
        "Soma na Janela (min)","Crítico?","Nível","Janela Início","Janela Fim","Detalhe"
    ]
    return out[cols]

def format_resumo_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Data"] = pd.to_datetime(out["data"]).dt.strftime("%d/%m/%Y")
    out["Entidade"] = out["entidade"].apply(entidade_simplificada)
    for col in ["N1","N2","N3"]:
        if col not in out.columns:
            out[col] = 0
    cols = ["Data","Entidade","N1","N2","N3"]
    return out[cols]

# -----------------------
# UI
# -----------------------

st.set_page_config(page_title="Classificador N1/N2/N3 (VFleets)", layout="wide")
st.title("Classificador de Situações (N1/N2/N3) — VFleets (UI simples + modo avançado)")

with st.sidebar:
    st.header("1) Importar CSV")
    up = st.file_uploader("Arraste o CSV exportado do VFleets", type=["csv"])
    st.caption("Detecção automática de encoding e separador (UTF-8/Latin-1; vírgula/ponto-e-vírgula/etc.).")

    st.header("2) Parâmetros")
    janela_min = st.number_input("Janela (minutos)", min_value=1, max_value=240, value=30, step=1)
    threshold = st.number_input("Threshold (soma de pesos)", min_value=1, max_value=10000, value=100, step=10)
    prefer_motorista = st.checkbox("Preferir Motorista como chave (fallback para Placa se vazio)", value=True)

    st.header("3) Pesos por Tipo (editar se desejar)")
    default_pesos_df = pd.DataFrame({
        "Tipo": CANON_TYPES,
        "Peso": [DEFAULT_PESOS[t] for t in CANON_TYPES],
    })
    pesos_edit = st.data_editor(default_pesos_df, num_rows="fixed", use_container_width=True)

    st.header("4) Exibição")
    modo_avancado = st.toggle("Modo avançado (mostrar colunas técnicas)", value=False)

    run_btn = st.button("Executar classificação", type="primary", use_container_width=True)

st.markdown(
    """
    **Regras especiais**:
    - Para **EXCESSO_VELOCIDADE**, apenas linhas com **NIVEL = MEDIO ou ALTO** entram no cálculo.
    - Demais tipos não usam o campo NIVEL.
    """
)

if up is not None:
    try:
        raw = try_read_csv(up)
        st.success(f"CSV lido (encoding={st.session_state['read_meta']['encoding']}, sep='{st.session_state['read_meta']['sep']}').")
        with st.expander("Prévia do arquivo (10 linhas)", expanded=False):
            st.dataframe(raw.head(10), use_container_width=True)

        st.subheader("Mapeamento de colunas")
        st.caption("Selecione as colunas equivalentes no seu CSV. Use Data+Hora **ou** uma coluna única DataHora. Informe também a coluna NIVEL (para EXCESSO_VELOCIDADE).")

        col1, col2, col3 = st.columns(3)
        with col1:
            col_tipo = st.selectbox("Coluna TIPO", options=list(raw.columns), index=(list(raw.columns).index(guess_col(raw, ["TIPO", "tipo"])) if guess_col(raw, ["TIPO", "tipo"]) else 0))
            col_placa = st.selectbox("Coluna PLACA", options=list(raw.columns), index=(list(raw.columns).index(guess_col(raw, ["PLACA", "placa"])) if guess_col(raw, ["PLACA", "placa"]) else 0))
        with col2:
            col_motorista = st.selectbox("Coluna MOTORISTA", options=list(raw.columns), index=(list(raw.columns).index(guess_col(raw, ["MOTORISTA", "motorista"])) if guess_col(raw, ["MOTORISTA", "motorista"]) else 0))
            col_data = st.selectbox("Coluna DATA (opcional se tiver DataHora)", options=[""] + list(raw.columns), index=(0 if guess_col(raw, ["DATA", "data"]) is None else list([""] + list(raw.columns)).index(guess_col(raw, ["DATA", "data"]))))
        with col3:
            col_hora = st.selectbox("Coluna HORA (opcional se tiver DataHora)", options=[""] + list(raw.columns), index=(0 if guess_col(raw, ["HORA", "hora"]) is None else list([""] + list(raw.columns)).index(guess_col(raw, ["HORA", "hora"]))))
            col_datahora = st.selectbox("Coluna DATAHORA (opcional)", options=[""] + list(raw.columns), index=(0 if guess_col(raw, ["DATAHORA", "datahora"]) is None else list([""] + list(raw.columns)).index(guess_col(raw, ["DATAHORA", "datahora"]))))
        col_nivel = st.selectbox("Coluna NIVEL (obrigatória para EXCESSO_VELOCIDADE)", options=[""] + list(raw.columns), index=(0 if guess_col(raw, ["NIVEL", "nivel"]) is None else list([""] + list(raw.columns)).index(guess_col(raw, ["NIVEL", "nivel"]))))

        pesos_map = normalize_tipo_weights(pesos_edit)

        if run_btn:
            if not col_tipo or not col_placa or not col_motorista:
                st.error("Mapeie pelo menos: TIPO, PLACA e MOTORISTA.")
            else:
                _col_data = col_data if col_data != "" else None
                _col_hora = col_hora if col_hora != "" else None
                _col_datahora = col_datahora if col_datahora != "" else None
                _col_nivel = col_nivel if col_nivel != "" else None
                if _col_datahora is None and (not _col_data or not _col_hora):
                    st.error("Informe 'DATA' e 'HORA' ou selecione uma coluna única 'DATAHORA'.")
                else:
                    with st.spinner("Processando..."):
                        resultados, resumo = classificar_eventos(
                            raw,
                            col_tipo=col_tipo,
                            col_data=_col_data,
                            col_hora=_col_hora,
                            col_datahora=_col_datahora,
                            col_placa=col_placa,
                            col_motorista=col_motorista,
                            col_nivel=_col_nivel,
                            pesos=pesos_map,
                            janela_min=int(janela_min),
                            threshold=int(threshold),
                            prefer_motorista=bool(prefer_motorista),
                        )

                    st.success("Classificação concluída!")

                    # Exibição legível
                    legiveis = format_resultados_for_display(resultados, janela_minutos=int(janela_min))
                    st.subheader("Resultados (legíveis)")
                    st.dataframe(legiveis, use_container_width=True, height=420)

                    st.subheader("Resumo por dia / entidade / nível (legível)")
                    st.dataframe(format_resumo_for_display(resumo), use_container_width=True)

                    # Exibição modo avançado
                    if modo_avancado:
                        st.subheader("Resultados (avançado)")
                        cols_adv = [
                            "data_hora","tipo_canon","peso","motorista","placa","entidade",
                            "window_total","janela_inicio","janela_fim","critico","nivel","ocorrencia_idx_no_dia","detalhe","nivel_evento"
                        ]
                        st.dataframe(resultados[cols_adv], use_container_width=True)

                    # Downloads
                    st.subheader("Downloads")
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="XlsxWriter") as writer:
                        # Raw e legível
                        resultados.to_excel(writer, sheet_name="Resultados (raw)", index=False)
                        resumo.to_excel(writer, sheet_name="Resumo (raw)", index=False)
                        legiveis.to_excel(writer, sheet_name="Resultados (legíveis)", index=False)
                        format_resumo_for_display(resumo).to_excel(writer, sheet_name="Resumo (legível)", index=False)
                        pd.DataFrame({"Tipo": list(pesos_map.keys()), "Peso": list(pesos_map.values())}).to_excel(writer, sheet_name="Pesos (referência)", index=False)
                    st.download_button(
                        label="Baixar Excel (raw + legível)",
                        data=buffer.getvalue(),
                        file_name="resultado_classificacao_alertas_UI_toggle.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

                    st.download_button(
                        label="Baixar Resultados (CSV legível)",
                        data=legiveis.to_csv(index=False).encode("utf-8-sig"),
                        file_name="resultados_legiveis.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.download_button(
                        label="Baixar Resumo (CSV legível)",
                        data=format_resumo_for_display(resumo).to_csv(index=False).encode("utf-8-sig"),
                        file_name="resumo_legivel.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

    except Exception as e:
        st.error(f"Erro ao processar o CSV: {e}")
else:
    st.info("Faça o upload do CSV no painel lateral para começar.")
