import io
from collections import defaultdict, deque
from datetime import timedelta, time

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="VFleets - Janelas Críticas", layout="wide")

DEFAULT_PESOS = {
    "EXCESSO_VELOCIDADE_BAIXO": 0,  # Ignora nível baixo (até 20%) para pontuação
    "EXCESSO_VELOCIDADE_MEDIO": 7,
    "EXCESSO_VELOCIDADE_ALTO": 13,
    "FADIGA_DO_MOTORISTA": 50,
    "SEM_CINTO": 50,
    "BOCEJO": 10,
    "USO_DE_CELULAR": 50,
    "POSSIVEL_COLISAO": 100,
    "ULTRAPASSAGEM_ILEGAL": 100,
    "CAMERA_OBSTRUIDA": 50,
}

FRIENDLY_TIPO = {
    "EXCESSO_VELOCIDADE_BAIXO": "Excesso de Velocidade Baixo (até 20%)",
    "EXCESSO_VELOCIDADE": "Excesso de Velocidade",
    "EXCESSO_VELOCIDADE_MEDIO": "Excesso de Velocidade Médio (20–30%)",
    "EXCESSO_VELOCIDADE_ALTO": "Excesso de Velocidade Alto (Acima de 30%)",
    "FADIGA_DO_MOTORISTA": "Fadiga do Motorista",
    "FADIGA_MOTORISTA": "Fadiga do Motorista",
    "SEM_CINTO": "Sem Cinto",
    "BOCEJO": "Bocejo",
    "USO_DE_CELULAR": "Uso de Celular",
    "MANUSEIO_CELULAR": "Uso de Celular",
    "POSSIVEL_COLISAO": "Possível Colisão",
    "ULTRAPASSAGEM_ILEGAL": "Ultrapassagem Ilegal",
    "CAMERA_OBSTRUIDA": "Câmera Obstruída",
}

EXCLUDE_ROTULOS = {
    "Falso Positivo",
    "Uso de celular",
    "Diversos Painel/Retrovisor/Comendoo",
}


def norm(s: str) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .strip()
        .upper()
        .replace("Á", "A")
        .replace("Â", "A")
        .replace("Ã", "A")
        .replace("À", "A")
        .replace("É", "E")
        .replace("Ê", "E")
        .replace("Í", "I")
        .replace("Ó", "O")
        .replace("Ô", "O")
        .replace("Õ", "O")
        .replace("Ú", "U")
        .replace("Ç", "C")
    )


def tipo_to_canon(tipo: str) -> str | None:
    t = norm(tipo)
    if not t:
        return None
    if "BOCEJO" in t:
        return "BOCEJO"
    if "FADIGA" in t:
        return "FADIGA_DO_MOTORISTA"
    if "SEM" in t and "CINTO" in t:
        return "SEM_CINTO"
    if ("CELULAR" in t) or ("MANUSEIO" in t and "CEL" in t):
        return "USO_DE_CELULAR"
    if ("EXCESSO" in t) or ("VELOCIDADE" in t):
        return "EXCESSO_VELOCIDADE"
    if "POSSIVEL" in t and "COLISAO" in t:
        return "POSSIVEL_COLISAO"
    if "ULTRAPASSAGEM" in t and "ILEGAL" in t:
        return "ULTRAPASSAGEM_ILEGAL"
    if "CAMERA" in t and ("OBSTRUIDA" in t or "OBSTRU" in t):
        return "CAMERA_OBSTRUIDA"
    return t


def is_missing_motorista(val) -> bool:
    return str(val).strip() == "Sem Identificação"


def choose_entity(motorista: str, placa: str, prefer_motorista=True) -> str:
    mot = (motorista or "").strip()
    pla = (placa or "").strip()
    if prefer_motorista and not is_missing_motorista(mot):
        return f"MOTORISTA::{mot}"
    if pla != "":
        return f"PLACA::{pla}"
    return "DESCONHECIDO"


def entidade_simplificada(ent: str) -> str:
    if not isinstance(ent, str):
        return ent
    if ent.startswith("MOTORISTA::"):
        return ent.replace("MOTORISTA::", "Motorista: ")
    if ent.startswith("PLACA::"):
        return ent.replace("PLACA::", "Placa: ")
    return ent


def format_datetime_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%d/%m/%Y %H:%M:%S")


def get_recommended_action(tipo_canon: str | None, nivel: str | None) -> str:
    """
    Retorna o texto de ação recomendada para a Situação Crítica,
    baseada no tipo de evento (tipo_canon) e no nível (N1/N2/N3).
    """
    if not tipo_canon or not nivel:
        return ""

    tipo = str(tipo_canon).upper()
    nivel = str(nivel).upper()

    # Fadiga
    if tipo == "FADIGA_DO_MOTORISTA":
        if nivel == "N1":
            return (
                "Fazer contato com o motorista, perguntar se está tudo bem e acompanhar por 1 minuto. "
                "Informar no grupo de Whatsapp Situação Crítica N1."
            )
        if nivel == "N2":
            return (
                "Entrar em contato e informar que o motorista é reincidente em fadiga e perguntar se está tudo bem. "
                "Acompanhar por 2 minutos. Informar no grupo de Whatsapp Situação Crítica N2."
            )
        if nivel == "N3":
            return (
                "Acompanhar o motorista com comunicação durante 5 minutos. "
                "Informar no grupo de Whatsapp Situação Crítica N3."
            )

    # Câmera Obstruída
    if tipo == "CAMERA_OBSTRUIDA":
        if nivel == "N1":
            return (
                "Fazer contato com o motorista e solicitar a remoção de qualquer objeto que esteja obstruindo a câmera. "
                "Informar no grupo de Whatsapp Situação Crítica N1."
            )
        if nivel in ("N2", "N3"):
            return (
                "Fazer novo contato com o motorista e solicitar a remoção do objeto que está obstruindo a câmera. "
                "Informar que o gestor será comunicado. Informar no grupo de Whatsapp Situação Crítica N3."
            )

    # Excesso de Velocidade Médio e Alto
    if tipo in ("EXCESSO_VELOCIDADE_MEDIO", "EXCESSO_VELOCIDADE_ALTO"):
        if nivel == "N1":
            return (
                "Contato com o motorista e solicitar a redução imediata do excesso de velocidade. "
                "Informar no grupo de Whatsapp Situação Crítica N1."
            )
        if nivel == "N2":
            return (
                "Contato com o motorista e solicitar a redução imediata do excesso reincidente de velocidade. "
                "Informar no grupo de Whatsapp Situação Crítica N2."
            )
        if nivel == "N3":
            return (
                "Fazer novo contato com o motorista e solicitar a redução da velocidade imediata. "
                "Informar que o gestor será comunicado. Informar no grupo de Whatsapp Situação Crítica N3."
            )

    # Sem Cinto
    if tipo == "SEM_CINTO":
        if nivel == "N1":
            return (
                "Contato com o motorista e avisar para uso imediato do cinto. "
                "Informar no grupo de Whatsapp Situação Crítica N1."
            )
        if nivel == "N2":
            return (
                "Novo contato com o motorista e aviso reincidente para uso imediato do cinto. "
                "Informar no grupo de Whatsapp Situação Crítica N2."
            )
        if nivel == "N3":
            return (
                "Novo contato com o motorista e pedir novamente a utilização do cinto. "
                "Informar que o gestor será comunicado. Informar no grupo de Whatsapp Situação Crítica N3."
            )

    # Bocejo
    if tipo == "BOCEJO":
        if nivel == "N1":
            return (
                "Fazer contato com o motorista. Perguntar se está tudo bem e acompanhar por 1 minuto. "
                "Informar no grupo de Whatsapp Situação Crítica N1."
            )
        if nivel == "N2":
            return (
                "Entrar em contato e informar que o motorista é reincidente em fadiga e perguntar se está tudo bem. "
                "Acompanhar por 2 minutos. Informar no grupo de Whatsapp Situação Crítica N2."
            )
        if nivel == "N3":
            return (
                "Acompanhar o motorista com comunicação durante 5 minutos. "
                "Informar no grupo de Whatsapp Situação Crítica N3."
            )

    # Uso de Celular
    if tipo == "USO_DE_CELULAR":
        if nivel == "N1":
            return (
                "Contato com o motorista e aviso para guardar o celular imediatamente. "
                "Informar no grupo de Whatsapp Situação Crítica N1."
            )
        if nivel == "N2":
            return (
                "Contato com o motorista e avisar que ele é reincidente e pedir para guardar o celular imediatamente. "
                "Informar no grupo de Whatsapp Situação Crítica N2."
            )
        if nivel == "N3":
            return (
                "Novo contato com o motorista e pedir novamente para guardar o celular. "
                "Informar que o gestor será comunicado. Informar no grupo de Whatsapp Situação Crítica N3."
            )

    # Possível Colisão
    if tipo == "POSSIVEL_COLISAO":
        return (
            "Tentar contato com o motorista e verificar se está tudo bem. "
            "Caso necessário acionar ajuda e informar os gestores no grupo do Whatsapp."
        )

    # Ultrapassagem Ilegal
    if tipo == "ULTRAPASSAGEM_ILEGAL":
        if nivel == "N1":
            return (
                "Contato com o motorista e solicitar que não faça ultrapassagem em local proibido. "
                "Informar no grupo de Whatsapp Situação Crítica N1."
            )
        if nivel == "N2":
            return (
                "Contato com o motorista e informar que ele é reincidente e reforçar que não deve fazer ultrapassagem em local proibido. "
                "Informar no grupo de Whatsapp Situação Crítica N2."
            )
        if nivel == "N3":
            return (
                "Fazer novo contato com o motorista e reforçar que aquele não é um local adequado para ultrapassagem. "
                "Informar que o gestor será comunicado. Informar no grupo de Whatsapp Situação Crítica N3."
            )

    return ""


def classificar_eventos(
    df: pd.DataFrame,
    col_tipo: str,
    col_data: str | None,
    col_hora: str | None,
    col_datahora: str | None,
    col_placa: str,
    col_motorista: str,
    col_nivel: str | None,
    col_rotulo: str | None,
    janela_min: int = 30,
    threshold: int = 100,
    prefer_motorista: bool = True,
):
    base = pd.DataFrame(
        {
            "tipo_raw": df[col_tipo],
            "placa": df[col_placa].astype(str),
            "motorista": df[col_motorista].astype(str),
        }
    )
    base["nivel_evento"] = (
        df[col_nivel].astype(str).str.upper().str.strip()
        if col_nivel and col_nivel in df.columns
        else None
    )
    base["rotulo"] = (
        df[col_rotulo].astype(str).str.strip()
        if col_rotulo and col_rotulo in df.columns
        else ""
    )

    if col_datahora and col_datahora in df.columns and col_datahora != "":
        dt = pd.to_datetime(
            df[col_datahora],
            errors="coerce",
            dayfirst=True,
            infer_datetime_format=True,
        )
    else:
        data = df[col_data].astype(str).str.strip() if col_data else ""
        hora = df[col_hora].astype(str).str.strip() if col_hora else ""
        dt = pd.to_datetime(
            data + " " + hora,
            errors="coerce",
            dayfirst=True,
            infer_datetime_format=True,
        )
    base["data_hora"] = dt
    base = base.dropna(subset=["data_hora"]).copy()

    # Canonização
    base["tipo_canon"] = base["tipo_raw"].map(tipo_to_canon)

    # Especialização por nível de excesso de velocidade
    if isinstance(base["nivel_evento"], pd.Series):
        mask_vel = base["tipo_canon"].eq("EXCESSO_VELOCIDADE")
        mask_baixo = mask_vel & base["nivel_evento"].isin(["BAIXO"])
        mask_medio = mask_vel & base["nivel_evento"].isin(["MEDIO", "MÉDIO"])
        mask_alto = mask_vel & base["nivel_evento"].isin(["ALTO"])
        base.loc[mask_baixo, "tipo_canon"] = "EXCESSO_VELOCIDADE_BAIXO"
        base.loc[mask_medio, "tipo_canon"] = "EXCESSO_VELOCIDADE_MEDIO"
        base.loc[mask_alto, "tipo_canon"] = "EXCESSO_VELOCIDADE_ALTO"

    # Entidade
    base["entidade"] = base.apply(
        lambda r: choose_entity(r["motorista"], r["placa"], prefer_motorista), axis=1
    )
    mask_sem_id = base["motorista"].astype(str).str.strip() == "Sem Identificação"
    base.loc[mask_sem_id, "entidade"] = (
        "PLACA::" + base.loc[mask_sem_id, "placa"].astype(str).str.strip()
    )

    # Peso por tipo
    base["peso"] = base["tipo_canon"].apply(
        lambda c: int(DEFAULT_PESOS.get(c, 0)) if c else 0
    )

    # Excluir por rótulo
    if isinstance(base["rotulo"], pd.Series):
        base.loc[base["rotulo"].isin(EXCLUDE_ROTULOS), "peso"] = 0

    # Ordenação temporal
    base = base.sort_values(["entidade", "data_hora"]).reset_index(drop=True)

    # Campos auxiliares
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
    counts_per_day = defaultdict(int)

    for idx, row in base.iterrows():
        ent = row["entidade"]
        t = row["data_hora"]
        w = int(row["peso"]) if not pd.isna(row["peso"]) else 0
        tipo = row["tipo_canon"]
        placa = row["placa"]

        deques[ent].append((t, w, tipo, placa))
        sum_in_window[ent] += w

        t_min = t - window
        while deques[ent] and deques[ent][0][0] < t_min:
            old_t, old_w, _, _ = deques[ent].popleft()
            sum_in_window[ent] -= old_w

        win_start = deques[ent][0][0] if deques[ent] else t
        win_end = win_start + window
        base.at[idx, "window_total"] = float(sum_in_window[ent])
        base.at[idx, "janela_inicio"] = win_start
        base.at[idx, "janela_fim"] = win_end

        if sum_in_window[ent] >= threshold:
            day_key = (ent, t.date())
            counts_per_day[day_key] += 1
            occ_n = counts_per_day[day_key]
            nivel = "N1" if occ_n == 1 else ("N2" if occ_n == 2 else "N3")

            base.at[idx, "critico"] = True
            base.at[idx, "nivel"] = nivel
            base.at[idx, "ocorrencia_idx_no_dia"] = occ_n

            detalhes = []
            for t_ev, w_ev, tipo_ev, placa_ev in list(deques[ent]):
                ts = pd.to_datetime(t_ev).strftime("%d/%m %H:%M:%S")
                friendly = FRIENDLY_TIPO.get(tipo_ev, tipo_ev)
                detalhes.append(f"{ts} - {friendly} [{placa_ev}] (+{w_ev})")
            base.at[idx, "detalhe"] = " | ".join(detalhes)

            deques[ent].clear()
            sum_in_window[ent] = 0

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


def format_resultados_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("data_hora").reset_index(drop=True)
    out["Data/Hora"] = format_datetime_series(out["data_hora"])
    out["Janela Início"] = format_datetime_series(out["janela_inicio"])
    out["Janela Fim"] = format_datetime_series(out["janela_fim"])
    out["Tipo"] = out["tipo_canon"].map(FRIENDLY_TIPO).fillna(out.get("tipo_canon"))
    out["Peso"] = out["peso"].astype("Int64")
    out["Motorista"] = out["motorista"].mask(
        out["motorista"].eq("Sem Identificação"), ""
    )
    out["Placa"] = out["placa"]
    out["Soma na Janela (min)"] = out["window_total"].round(0).astype("Int64")
    out["Situação Crítica?"] = out["critico"].map({True: "Sim", False: "Não"})
    out["Nível"] = out["nivel"].fillna("")
    out["Detalhe"] = out["detalhe"].fillna("")
    cols = [
        "Data/Hora",
        "Tipo",
        "Peso",
        "Motorista",
        "Placa",
        "Soma na Janela (min)",
        "Situação Crítica?",
        "Nível",
        "Janela Início",
        "Janela Fim",
        "Detalhe",
    ]
    return out[cols]


def ranking_criticos(df_resultados: pd.DataFrame):
    crit = df_resultados.loc[df_resultados["critico"]].copy()
    if crit.empty:
        return pd.DataFrame(
            columns=["Entidade", "Tipo de Entidade", "Último Nível", "Data/Hora"]
        )
    map_val = {"N1": 1, "N2": 2, "N3": 3}
    crit["nivel_val"] = crit["nivel"].map(map_val).fillna(0).astype(int)
    crit = crit.sort_values(
        ["nivel_val", "data_hora"], ascending=[False, False]
    ).copy()
    best = crit.drop_duplicates(subset=["entidade"], keep="first").copy()
    best["Tipo de Entidade"] = np.where(
        best["entidade"].str.startswith("MOTORISTA::"), "Motorista", "Placa"
    )
    best["Entidade"] = best["entidade"].apply(entidade_simplificada)
    best["Último Nível"] = best["nivel"]
    best["Data/Hora"] = format_datetime_series(best["data_hora"])
    order_val = {"N3": 3, "N2": 2, "N1": 1}
    best["ord"] = best["Último Nível"].map(order_val).fillna(0)
    best = best.sort_values(["ord", "Data/Hora"], ascending=[False, False])
    return best[["Entidade", "Tipo de Entidade", "Último Nível", "Data/Hora"]]


def _parse_datetime_from_raw(
    raw: pd.DataFrame, col_datahora: str | None, col_data: str | None, col_hora: str | None
) -> pd.Series:
    if col_datahora and col_datahora in raw.columns and col_datahora != "":
        return pd.to_datetime(
            raw[col_datahora],
            errors="coerce",
            dayfirst=True,
            infer_datetime_format=True,
        )
    else:
        data = (
            raw[col_data].astype(str).str.strip()
            if col_data and col_data in raw.columns
            else ""
        )
        hora = (
            raw[col_hora].astype(str).str.strip()
            if col_hora and col_hora in raw.columns
            else ""
        )
        return pd.to_datetime(
            data + " " + hora,
            errors="coerce",
            dayfirst=True,
            infer_datetime_format=True,
        )


def _is_in_work_window(dt: pd.Timestamp, start_t: time, end_t: time) -> bool:
    if pd.isna(dt):
        return False
    t = dt.time()
    if start_t <= end_t:
        return start_t <= t <= end_t
    else:
        return (t >= start_t) or (t <= end_t)


def _traduz_status(s):
    s = str(s).strip().upper()
    if s == "RESOLVIDO":
        return "Já tratado"
    if s in ("NOVO", "ATRIBUIDO"):
        return "Pendente Tratativa"
    return "(Desconhecido)"


def tabela_tratativas(
    resultados: pd.DataFrame,
    raw: pd.DataFrame,
    col_motorista_raw: str | None,
    col_datahora: str | None,
    col_data: str | None,
    col_hora: str | None,
    work_start: time | None,
    work_end: time | None,
    only_last: bool = True,
) -> pd.DataFrame:
    # Coluna STATUS
    status_col = None
    for c in raw.columns:
        if str(c).strip().upper() == "STATUS":
            status_col = c
            break
    if status_col is None:
        return pd.DataFrame()

    crit = resultados.loc[resultados["critico"]].copy()
    if crit.empty:
        return pd.DataFrame()

    # Somente motoristas identificados
    crit = crit[crit["entidade"].astype(str).str.startswith("MOTORISTA::")].copy()
    if crit.empty:
        return pd.DataFrame()

    # Nome do motorista
    crit["motorista_nome"] = crit["entidade"].astype(str).str.replace(
        "^MOTORISTA::", "", regex=True
    )

    # Preparar raw
    raw_dt = _parse_datetime_from_raw(raw, col_datahora, col_data, col_hora)
    raw_safe = raw.copy()
    raw_safe["_data_hora_raw"] = raw_dt
    if col_motorista_raw and col_motorista_raw in raw.columns:
        raw_safe["_motorista_raw"] = raw[col_motorista_raw].astype(str).str.strip()
    else:
        guess_col = None
        for c in raw.columns:
            if str(c).strip().lower() == "motorista":
                guess_col = c
                break
        if guess_col:
            raw_safe["_motorista_raw"] = raw[guess_col].astype(str).str.strip()
        else:
            raw_safe["_motorista_raw"] = ""

    if only_last:
        crit_last = (
            crit.sort_values("data_hora", ascending=False)
            .drop_duplicates(subset=["motorista_nome"], keep="first")
            .copy()
        )
        crit_last["_motorista_raw"] = crit_last["motorista_nome"].astype(str).str.strip()
        merged = crit_last.merge(
            raw_safe[[status_col, "_motorista_raw", "_data_hora_raw"]],
            left_on=["_motorista_raw", "data_hora"],
            right_on=["_motorista_raw", "_data_hora_raw"],
            how="left",
        )
    else:
        crit_all = crit.copy()
        crit_all["_motorista_raw"] = crit_all["motorista_nome"].astype(str).str.strip()
        merged = crit_all.merge(
            raw_safe[[status_col, "_motorista_raw", "_data_hora_raw"]],
            left_on=["_motorista_raw", "data_hora"],
            right_on=["_motorista_raw", "_data_hora_raw"],
            how="left",
        )

    # Campos principais
    merged["Status"] = merged[status_col].apply(_traduz_status)
    merged["Motorista"] = merged["motorista_nome"]
    merged["Data/Hora Último Alerta"] = merged["data_hora"]
    merged["Nível Recomendado"] = merged["nivel"]

    # Eventos que compõem a situação crítica (detalhe -> quebra de linha)
    merged["Eventos da Situação Crítica"] = (
        merged.get("detalhe", "")
        .fillna("")
        .astype(str)
        .str.replace(" | ", "\n", regex=False)
    )

    # Ação recomendada (com quebras de linha para reduzir scroll horizontal)
    merged["Ação Recomendada"] = merged.apply(
        lambda r: get_recommended_action(r.get("tipo_canon"), r.get("Nível Recomendado")),
        axis=1,
    )
    merged["Ação Recomendada"] = (
        merged["Ação Recomendada"]
        .fillna("")
        .astype(str)
        .str.replace(". ", ".\n", regex=False)
    )

    # Prioridade por expediente
    if work_start is None:
        work_start = time(8, 0)
    if work_end is None:
        work_end = time(17, 0)
    merged["Dentro do Expediente?"] = merged["Data/Hora Último Alerta"].apply(
        lambda d: _is_in_work_window(pd.to_datetime(d), work_start, work_end)
    )

    # Formatar datas
    merged["Data/Hora Último Alerta"] = format_datetime_series(
        pd.to_datetime(merged["Data/Hora Último Alerta"])
    )

    merged["Prioridade"] = np.where(
        (merged["Status"] == "Pendente Tratativa")
        & (merged["Dentro do Expediente?"] == True),
        1,
        2,
    )

    out = (
        merged[
            [
                "Motorista",
                "Data/Hora Último Alerta",
                "Status",
                "Nível Recomendado",
                "Eventos da Situação Crítica",
                "Ação Recomendada",
                "Dentro do Expediente?",
                "Prioridade",
            ]
        ]
        .sort_values(["Prioridade", "Data/Hora Último Alerta"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return out


st.title("VFleets - Janela Crítica")
up = st.file_uploader(
    "Envie um CSV do análise de alertas do Vfleets", type=["csv"]
)

with st.sidebar:
    st.subheader("Parâmetros")
    prefer_motorista = st.checkbox(
        "Preferir Motorista (fallback para Placa quando 'Sem Identificação')",
        value=True,
    )
    janela_min = st.number_input("Tamanho da janela (min)", 10, 180, 30, 5)
    threshold = st.number_input("Limiar de pontos para crítica", 5, 1000, 100, 1)

    st.markdown("**Horário de trabalho do analista**")
    work_start = st.time_input(
        "Início do expediente", value=time(8, 0), key="work_start"
    )
    work_end = st.time_input(
        "Fim do expediente", value=time(17, 0), key="work_end"
    )

    only_last = st.checkbox(
        "Mostrar apenas a última situação crítica por motorista",
        value=False,
        help="Desmarque para ver todas as situações críticas de cada motorista.",
    )

    pesos_df = pd.DataFrame(
        {
            "Tipo": [FRIENDLY_TIPO.get(k, k) for k in DEFAULT_PESOS.keys()],
            "Peso": list(DEFAULT_PESOS.values()),
        }
    )
    st.markdown("**Pesos por Tipo (atual)**")
    st.dataframe(
        pesos_df.sort_values("Tipo"), use_container_width=True, height=360
    )

if up is not None:
    content = up.read()
    try:
        raw = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
    except Exception:
        raw = pd.read_csv(
            io.BytesIO(content),
            sep=None,
            engine="python",
            encoding="latin1",
        )

    cols = [c.strip() for c in raw.columns]
    lower = {c.lower(): c for c in cols}

    col_tipo = lower.get("tipo") or lower.get("rótulo") or lower.get("rotulo")
    col_nivel = lower.get("nivel") or lower.get("nível") or lower.get("nivel_evento")
    col_rotulo = lower.get("rotulo") or lower.get("rótulo") or lower.get("label")
    col_placa = lower.get("placa")
    col_motorista = lower.get("motorista")
    col_datahora = lower.get("data/hora") or lower.get("datahora") or None
    col_data = lower.get("data")
    col_hora = lower.get("hora")

    motoristas_unicos = (
        raw[col_motorista].dropna().astype(str).str.strip().unique().tolist()
        if col_motorista in raw.columns
        else []
    )
    motoristas_unicos = [
        m for m in motoristas_unicos if m and m != "Sem Identificação"
    ]
    motoristas_unicos.sort()
    sel_mot = st.selectbox(
        "Filtrar por Motorista (opcional)",
        options=["(Todos)"] + motoristas_unicos,
        index=0,
    )

    raw_filtered = raw.copy()
    if sel_mot != "(Todos)":
        raw_filtered = raw_filtered[
            raw_filtered[col_motorista].astype(str).str.strip() == sel_mot
        ].copy()

    if not col_tipo or not col_placa or not col_motorista:
        st.error("Mapeie pelo menos: TIPO, PLACA e MOTORISTA.")
    else:
        if col_datahora is None and (not col_data or not col_hora):
            st.error("Informe 'DATA' e 'HORA' ou selecione 'DATA/HORA'.")
        else:
            with st.spinner("Processando..."):
                resultados, resumo = classificar_eventos(
                    raw_filtered,
                    col_tipo=col_tipo,
                    col_data=col_data,
                    col_hora=col_hora,
                    col_datahora=col_datahora,
                    col_placa=col_placa,
                    col_motorista=col_motorista,
                    col_nivel=col_nivel,
                    col_rotulo=col_rotulo,
                    janela_min=int(janela_min),
                    threshold=int(threshold),
                    prefer_motorista=bool(prefer_motorista),
                )

            st.success("Classificação concluída!")

            st.subheader("Resultados")
            legiveis = format_resultados_for_display(resultados)
            st.dataframe(legiveis, use_container_width=True, height=500)

            st.subheader(
                "Tratativas por motorista"
                + (
                    " (apenas última por motorista)"
                    if only_last
                    else " (todas as situações críticas)"
                )
            )
            trat = tabela_tratativas(
                resultados,
                raw_filtered,
                col_motorista,
                col_datahora,
                col_data,
                col_hora,
                work_start=work_start,
                work_end=work_end,
                only_last=only_last,
            )
            if trat.empty:
                st.info(
                    "Sem situações críticas de motoristas ou coluna STATUS ausente no CSV."
                )
            else:
                pendentes_exp = trat[
                    (trat["Status"] == "Pendente Tratativa")
                    & (trat["Dentro do Expediente?"] == True)
                ]
                colm1, colm2, colm3 = st.columns(3)
                with colm1:
                    st.metric(
                        "Pendentes dentro do expediente",
                        int(pendentes_exp.shape[0]),
                    )
                with colm2:
                    st.metric(
                        "Pendentes (total)",
                        int(
                            trat[trat["Status"] == "Pendente Tratativa"].shape[0]
                        ),
                    )
                with colm3:
                    st.metric(
                        "Já tratados",
                        int(trat[trat["Status"] == "Já tratado"].shape[0]),
                    )

                st.dataframe(
                    trat.drop(columns=["Prioridade"]),
                    use_container_width=True,
                    height=420,
                )

            st.subheader("Ranking – último nível alcançado por entidade (desc)")
            rank = ranking_criticos(resultados)
            colA, colB = st.columns(2)
            with colA:
                st.caption("Motoristas")
                st.dataframe(
                    rank[rank["Tipo de Entidade"] == "Motorista"],
                    use_container_width=True,
                    height=340,
                )
            with colB:
                st.caption("Placas")
                st.dataframe(
                    rank[rank["Tipo de Entidade"] == "Placa"],
                    use_container_width=True,
                    height=340,
                )

            st.subheader("Downloads")
            buf = io.BytesIO()
            writer = pd.ExcelWriter(buf, engine="xlsxwriter")
            with writer as w:
                legiveis.to_excel(w, sheet_name="Resultados", index=False)
                resultados.to_excel(w, sheet_name="Resultados (raw)", index=False)
                rank.to_excel(w, sheet_name="Ranking", index=False)
                if "trat" in locals() and not trat.empty:
                    trat.to_excel(w, sheet_name="Tratativas", index=False)
            st.download_button(
                "Baixar Excel",
                data=buf.getvalue(),
                file_name="classificacao_janela_critica_final14.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
else:
    st.info("Envie um CSV para iniciar.")
