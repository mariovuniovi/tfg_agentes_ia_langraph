# Sistema MLOps Multiagente — TFG

Sistema MLOps multiagente que lleva un dataset desde la validación hasta un modelo
registrado en producción, con aprobación humana (HITL) en dos puntos del proceso.
Construido sobre un **StateGraph de LangGraph con enrutado determinista**: los agentes
LLM se usan únicamente donde aportan razonamiento (validación de datos, planificación
de modelos y auditoría del resultado); el entrenamiento, la evaluación y el despliegue
son código Python determinista y reproducible.

> Trabajo de Fin de Grado — Universidad de Oviedo, 2026.
> La memoria (LaTeX) vive en [`docs/thesis/`](docs/thesis/).

---

## 1. Arquitectura

### El grafo

Un router determinista (`workflow_controller`, **sin LLM**) lee el estado tras cada
nodo y decide el siguiente paso. Ningún LLM decide el flujo; cada decisión de enrutado
queda registrada en el log.

```
                                ┌───────────────────────┐
           START ──────────────►│  workflow_controller  │◄──── (cada nodo vuelve
                                │  (router determinista)│       al controller)
                                └───────────┬───────────┘
                                            │
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  ┌─────────────┐
   │ data_validator 🤖│─►│ dataset_approval👤│─►│  planner 🤖  │─►│ executor ⚙️ │
   │ react agent + 11 │  │ gate HITL 1       │  │ react agent, │  │ entrena los │
   │ tools de datos   │  │ (interrupt)       │  │ plan top-5   │  │ candidatos  │
   └──────────────────┘  └──────────────────┘  └──────────────┘  └──────┬──────┘
                                                                        │
   ┌──────────────┐  ┌──────────────────────┐  ┌─────────────────┐  ┌───▼─────────┐
   │ deployer ⚙️  │◄─│ deployment_approval👤│◄─│ report_writer 🤖│◄─│evaluation ⚙️│
   │ registro en  │  │ gate HITL 2          │  │ auditoría LLM   │  │ decisión de │
   │ MLflow → END │  │ (interrupt)          │  │ estructurada    │  │ promoción   │
   └──────────────┘  └──────────────────────┘  └─────────────────┘  └─────────────┘

        🤖 etapa con LLM        ⚙️ determinista        👤 aprobación humana
```

### Principios de diseño

1. **Determinista primero.** Carga de datos, entrenamiento, métricas y decisión de
   promoción son Python puro. Los LLM solo intervienen para interpretar datos crudos,
   razonar la estrategia de modelado y redactar la auditoría.
2. **Cada etapa es un paquete de dominio profundo.** El grafo
   ([`mlops_graph.py`](src/mlops_agents/graphs/mlops_graph.py), ~150 líneas) es pura
   topología: consume cada etapa a través de una interfaz de un símbolo
   (`data_validator_node`, `planner_node`, `run_training_plan`, …).
3. **Contratos tipados.** Cada nodo devuelve su porción de estado mediante un contrato
   Pydantic ([`contracts/outputs.py`](src/mlops_agents/contracts/outputs.py)); el
   paquete `contracts/` no importa ningún módulo de dominio (regla de capas).
4. **HITL con `interrupt()`.** Las dos aprobaciones humanas viven en nodos-gate del
   grafo; todo el código previo a un `interrupt()` es idempotente (el nodo se
   re-ejecuta al reanudar).
5. **Experiencia acumulada + evidencia verificable.** Cada ejecución escribe un
   registro de experiencia (SQLite); el planner recupera experiencias similares y
   reglas ML estáticas como *evidencia citada*, y una capa de validación híbrida
   comprueba que el plan cite evidencia real (con detección de conflictos duros y
   blandos) antes de aceptarlo — si no, reintento acotado.

### Tres puertas de entrada al mismo dominio

| Entrada | Camino | Uso |
|---|---|---|
| **UI web** | `frontend/` (Next.js) → `api/` (FastAPI, REST + WebSocket) → grafo | demo interactiva con HITL por navegador |
| **CLI** | `scripts/run_pipeline.py` → `graphs/cli.py` → grafo | ejecución por consola con HITL interactivo |
| **Medición** | `scripts/measure_agentic_cost.py` → grafo (headless, auto-aprueba) | experimentos de coste/tiempo del capítulo 6 de la memoria |

---

## 2. Árbol de módulos

```
src/mlops_agents/            ← dominio (78 ficheros, mypy estricto en verde)
├── graphs/                  topología y control de flujo
│   ├── mlops_graph.py         StateGraph: nodos + wrappers finos (~150 líneas)
│   ├── workflow_controller.py router determinista (las reglas de enrutado viven aquí)
│   ├── approval_nodes.py      los 2 gates HITL (interrupt())
│   ├── cli.py                 entrada CLI (estado inicial + streaming + prompt HITL)
│   └── taxonomy.py            categorías de nodo (agente / determinista / HITL)
│
├── data_validation/          ETAPA LLM 1 — validación de datos
│   ├── node.py                nodo del grafo: contexto → agente → extracción tipada
│   ├── agent.py               react agent (11 tools: carga, joins, schema, calidad,
│   │                          imputación, gaps temporales)
│   ├── context.py             construcción del contexto + extracción de tool-results
│   └── schema_contract.py     validación determinista del schema subido
│
├── planning/                 ETAPA LLM 2 — planificación de modelos
│   ├── node.py                nodo: contexto → agente (máx. 2 intentos) → plan validado
│   ├── agent.py               constructor del react agent (salida estructurada)
│   ├── tools.py               4 tools con traza (modelos, experiencias, reglas)
│   ├── validation.py          validación híbrida: integridad del plan, referencias de
│   │                          evidencia, conflictos duros/blandos
│   ├── context.py             ground-truth determinista para validar al agente
│   └── trace.py / prompts.py  traza de tools / mensajes de reintento
│
├── training/                 ETAPA DETERMINISTA — entrenamiento multi-candidato
│   ├── executor.py            orquestación: run_training_plan (única interfaz),
│   │                          selección de campeón, MLflow padre/hijos (~360 líneas)
│   ├── tabular_runner.py      clasificación + regresión (validación, retrain)
│   ├── forecasting_runner.py  todo lo específico de series temporales: folds
│   │                          temporales, exógenas sin leakage, evaluación en test
│   ├── profiler.py            perfil del dataset (buckets para el experience pool)
│   ├── splitter.py            split train-pool / test
│   ├── validation_policy.py   estrategia de validación para forecasting
│   ├── exog_policy.py · exog_extender.py · validation_folds.py
│   ├── trial_budget.py        nº de trials determinista + sampler Optuna (Grid/TPE)
│   ├── override_validation.py estrechado seguro de search spaces del planner
│   └── experience_record.py   escritura del registro de experiencia por ejecución
│
├── evaluation/               promotion.py (decisión determinista: umbrales + campeón
│   │                         actual en MLflow) · report_writer.py (auditoría LLM
│   │                         estructurada) · champion.py (resolución de nombre)
├── deployment/               deployer.py — registro en MLflow Model Registry
│
├── experience/               pool.py (SQLite) + retrieval.py (similitud por buckets)
├── knowledge/                reader.py — base de reglas ML estática (ml_rules.yaml)
├── models/                   loader.py (registro de 20 modelos vía YAML) +
│                             factories.py + search_spaces.py
├── forecasting/              seasonality.py — política de season length por frecuencia
│
├── contracts/                contratos Pydantic (salidas de nodo → estado, planes,
│                             perfiles, schema, evidencia); SIN imports de dominio
├── state/                    agent_state.py — AgentState (TypedDict del grafo)
├── tools/                    data_tools.py · join_discovery_tools.py · mlflow_tools.py
├── prompts/                  YAML por agente (modelo + prompt) + loader.py
├── config/                   settings.py (pydantic-settings, lee .env) + constants.py
├── observability/            pricing.py — coste por tokens (model_pricing.yaml)
└── utils/                    llm.py (factoría ChatOpenAI) + logging.py

api/                          backend FastAPI (la frontera HTTP/WebSocket)
├── main.py                   app + CORS + /health
├── routers/                  runs.py (POST /runs, WS /ws/{id}, aprobación HITL) ·
│                             experiments.py (proxy MLflow) · uploads.py (CSV+schema)
├── services/                 pipeline.py (tarea async: stream del grafo → eventos
│                             WebSocket) · run_store.py · mlflow_client.py
└── tests/                    suite del backend (incluida en pytest por defecto)

frontend/                     UI Next.js — cliente tipado del WebSocket (~15 tipos de
                              evento): stepper del pipeline, gates HITL, experimentos

scripts/                      run_pipeline.py · run_benchmark.py (benchmark offline del
                              experience pool) · seed_mlflow.py · measure_agentic_cost.py
                              + agentic_cost_aggregate.py (medición coste/tiempo, cap. 6)
                              · generadores de datasets sintéticos

tests/                        espeja src/ 1:1 (test_data_validation/, test_planning/,
                              test_training/, …) + test_api/ + test_integration/
docs/thesis/                  memoria en LaTeX y PDF(capítulos + anexos)
data/ · storage/ · mlruns/    datasets de muestra · experience pool (SQLite) · MLflow
```

---

## 3. Calidad verificable

Cada afirmación se comprueba con un comando:

| Afirmación | Comando | Resultado esperado |
|---|---|---|
| Suite de tests | `uv run pytest -m "not integration"` | **638 passed** (los unitarios no llaman a ningún LLM real) |
| Tipado estricto | `uv run mypy src/` | **0 errores** (modo strict + plugin de Pydantic) |
| Lint | `uv run ruff check .` | **0 errores** (excepciones por-fichero documentadas en `pyproject.toml`) |
| Integración (LLM real) | `uv run pytest -m integration` | requiere `OPENAI_API_KEY` |

Convenciones: `tests/` espeja la estructura de `src/`; los tests unitarios mockean el
LLM; `contracts/` no importa dominio; los nodos devuelven estado parcial tipado y
nunca mutan `AgentState` in situ.

---

## 4. Stack tecnológico

- **Python 3.12 + UV** (src-layout) · **LangGraph + LangChain** (orquestación)
- **OpenAI API** — `gpt-5.4-mini` (validador de datos y planner) + `gpt-5.4-nano`
  (report writer); modelo y reasoning effort se configuran por agente en `prompts/*.yaml`
- **scikit-learn, LightGBM, XGBoost, CatBoost** (tabular) · **statsforecast +
  skforecast** (forecasting: AutoETS, AutoARIMA, ML recursivo)
- **Optuna** — búsqueda de hiperparámetros determinista (GridSampler exhaustivo para
  espacios enumerables, TPE con semilla para el resto)
- **MLflow** (tracking + Model Registry) · **SQLite** (experience pool)
- **FastAPI + Next.js** (API y UI)

---

## 5. Cómo ejecutarlo

### Con Docker (recomendado — no requiere Python ni Node locales)

```bash
git clone https://github.com/mariovuniovi/tfg_agentes_ia_langraph.git
cd tfg_agentes_ia_langraph
cp .env.example .env          # rellenar OPENAI_API_KEY
docker compose up --build
```

| Contenedor | Puerto | Rol |
|---|---|---|
| `mlops-frontend` | **3000** | UI Next.js — abrir <http://localhost:3000> |
| `mlops-api` | **8000** | FastAPI + grafo LangGraph (docs OpenAPI en `/docs`) |
| `mlops-mlflow` | **5000** | Servidor MLflow (runs, métricas, registry; volumen persistente) |

Día a día: `docker compose up` (sin rebuild) · `docker compose up --build` (tras
cambios de código) · `docker compose down -v` (reset completo, borra el volumen de
MLflow) · `docker compose logs -f api`.

### En local (desarrollo)

```bash
uv sync                                          # dependencias Python
docker compose up mlflow                         # solo el contenedor de MLflow
uv run uvicorn api.main:app --reload --port 8000 # backend (otra terminal)
cd frontend && npm install && npm run dev        # frontend (otra terminal)
```

O sin interfaz web, por consola:

```bash
uv run python scripts/run_pipeline.py data/samples/iris_measurements.csv data/samples/iris_labels.csv
```

### Usar GitHub Models (gratis, sin coste de OpenAI)

El sistema habla con la API de OpenAI, pero también con **cualquier endpoint
compatible**. [GitHub Models](https://github.com/marketplace/models) ofrece modelos
gratuitos (con límite de peticiones/día) usando solo un token de GitHub — ideal para
probar la app sin gastar dinero. Basta con tres variables en el `.env`, sin tocar código:

```bash
OPENAI_API_KEY=ghp_tu_github_token          # un Personal Access Token de GitHub
OPENAI_BASE_URL=https://models.github.ai/inference
OPENAI_MODEL_OVERRIDE=openai/gpt-4.1-mini   # fuerza este modelo en todos los agentes
```

`OPENAI_BASE_URL` redirige las llamadas al endpoint de GitHub Models y
`OPENAI_MODEL_OVERRIDE` impone un modelo compatible en todos los agentes (los nombres
de GitHub Models difieren de los de OpenAI). Definir un `base_url` desactiva
automáticamente la Responses API de OpenAI, que GitHub Models no soporta.

> **Nota:** GitHub Models es un servicio distinto con límite de peticiones diario en el
> nivel gratuito; la fiabilidad de los agentes que requieren seguir instrucciones con
> precisión (validador y planner) puede ser algo menor que con la API de OpenAI. Úsalo
> para explorar el sistema; usa OpenAI para resultados consistentes.

### Sembrar el experience pool (opcional)

El planner recupera experiencias de entrenamientos pasados. Para poblarlas con el
benchmark offline (21 datasets públicos: sklearn / OpenML / yfinance / CSV locales):

```bash
uv run python scripts/run_benchmark.py
uv run python scripts/seed_mlflow.py             # runs de ejemplo en MLflow
```

---

## 6. Forecasting sin leakage (resumen)

El caso central del executor de forecasting: **una serie objetivo + varias exógenas**,
donde algunos valores exógenos no se conocerán en el momento de predecir. Cada columna
exógena se etiqueta con `future_availability` (`known_future` / `unknown_future`):

1. La estrategia de validación se elige de forma determinista según longitud de
   historia y drift esperado (single split / rolling / expanding).
2. En cada fold, las columnas `unknown_future` se extienden **solo con historia de
   entrenamiento** (naive carry / ETS / AutoARIMA).
3. **Nunca** se usan valores futuros realizados de una columna `unknown_future` —
   este es el cortafuegos anti-leakage.
4. Estrategias aplicadas y métricas por fold quedan en el registro de experiencia.

Implementación en [`training/forecasting_runner.py`](src/mlops_agents/training/forecasting_runner.py)
y [`training/exog_extender.py`](src/mlops_agents/training/exog_extender.py).

---

## Licencia

TFG — Universidad de Oviedo, 2026.
