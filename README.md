# VFleets - Janela Crítica (Streamlit)

App Streamlit para classificar alertas por janela móvel (30 min) com limiar configurável, reset pós-crítica e fallback por **Placa** quando **Motorista = "Sem Identificação"**.

## Features
- Ordenação **sempre crescente** por Data/Hora.
- Janela deslizante por entidade (prefere **Motorista**; se ausente/“Sem Identificação”, cai para **Placa**).
- **Reset imediato** dos pontos após gerar uma situação crítica (N1 → zera → reconta para N2/N3).
- Coluna **Detalhe** listando os eventos que contribuíram para o disparo.
- Filtro/visualização de resultados legíveis e resumo.
- Export para Excel.

## Rodando localmente
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy no Streamlit Community Cloud
1. Crie um repositório **GitHub** (público ou privado com acesso ao Streamlit).
2. Suba estes arquivos: `app.py`, `requirements.txt`, `.streamlit/config.toml`, `.gitignore`, `README.md`.
3. No Streamlit Cloud, escolha **"New app"** e aponte para o repositório/branch.
4. Defina `app.py` como o **main file**.
5. Deploy!

## CSV esperado
- Deve conter colunas de **Data** e **Hora** (ou Data/Hora unificada), **Tipo**, **Placa**, **Motorista** e (opcional) **Nível**.
- Quando `Motorista` for **"Sem Identificação"**, o app **força** a entidade por **Placa** (garante que placas distintas não sejam somadas).
- Pesos por tipo podem ser ajustados no app.

## Observações
- Se algum arquivo CSV vier com codificação diferente (ex.: Latin-1), o app tentará detectar automaticamente.
- A coluna `Detalhe` mostrará cada evento da janela no formato: `HH:MM:SS TIPO [PLACA] (+PESO)`.