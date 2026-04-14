# Estructura del proyecto — descripción de carpetas y ficheros

> Referencia rápida de para qué sirve cada fichero del proyecto.

---

## Raíz del proyecto

| Fichero | Propósito |
|---------|-----------|
| `pyproject.toml` | Configuración central de UV: dependencias, versión de Python, y configuración de ruff, mypy y pytest. Todo en un solo fichero. |
| `uv.lock` | Lockfile generado por UV (se crea con `uv sync`). Fija las versiones exactas de todos los paquetes. Se commitea a git. |
| `.python-version` | Fija la versión de Python a `3.12` para UV y pyenv. |
| `.env` | Variables de entorno reales (ignorado por git). Copia de `.env.example` con valores reales. |
| `.env.example` | Plantilla documentada de las variables de entorno necesarias. Se commitea a git. |
| `.gitignore` | Excluye `.venv/`, `.env`, artefactos de MLflow, cachés, etc. |
| `CLAUDE.md` | Instrucciones del proyecto para Claude Code — se carga automáticamente en cada sesión. Incluye comandos, arquitectura y convenciones. |
| `PLAN.md` | Plan de trabajo con historias de usuario. Marca el progreso del TFG. |
| `STRUCTURE.md` | Este fichero — descripción de la estructura del proyecto. |
| `langgraph.json` | Config para el servidor de LangGraph Cloud/local. Apunta al grafo compilado. |
| `Dockerfile` | Build multi-stage con UV: instala dependencias en una capa separada del código fuente para cache eficiente. |
| `docker-compose.yml` | Levanta dos servicios: MLflow Tracking Server (puerto 5000) y la app Streamlit (puerto 8501). |

---

## `src/mlops_agents/` — paquete principal

### `state/` — Estado compartido del grafo

| Fichero | Propósito |
|---------|-----------|
| `agent_state.py` | Define `AgentState` (TypedDict): el estado compartido que se pasa entre todos los nodos del grafo. **Leer antes de tocar cualquier agente.** |
| `schemas.py` | Esquemas Pydantic para outputs estructurados: `RouterOutput` (decisión del supervisor), `ValidationResult`, `TrainingResult`, `EvaluationResult`. |

### `config/` — Configuración de la aplicación

| Fichero | Propósito |
|---------|-----------|
| `settings.py` | `Settings` (Pydantic BaseSettings): lee todas las variables de entorno desde `.env`. Nunca hardcodear tokens — usar siempre `settings.github_token`, etc. |
| `constants.py` | Constantes globales: umbrales de calidad (`MIN_ACCURACY_TO_DEPLOY`), nombres de agentes, aliases de MLflow. |

### `utils/` — Utilidades compartidas

| Fichero | Propósito |
|---------|-----------|
| `llm.py` | Factoría de LLMs: `get_llm()` devuelve el modelo principal (worker agents), `get_router_llm()` devuelve el modelo más barato para el supervisor (nano en lugar de mini). |
| `logging.py` | Setup de loguru: devuelve un logger con el nombre del módulo. Usar `get_logger(__name__)` — nunca `print()`. |
| `runners.py` | Entry point del script `mlops-dashboard` (registrado en `pyproject.toml`). |

### `tools/` — Herramientas deterministas `@tool`

Las herramientas son **funciones Python puras** decoradas con `@tool`. No llaman a LLMs. Los agentes las llaman y luego interpretan los resultados.

| Fichero | Propósito |
|---------|-----------|
| `data_tools.py` | `load_dataset` (resumen CSV), `validate_schema` (columnas esperadas), `check_missing_values` (% de nulos por columna). |
| `evidently_tools.py` | `check_data_quality` (reporte de calidad con Evidently AI), `check_data_drift` (detección de drift PSI entre dataset actual y referencia). |
| `training_tools.py` | `tune_hyperparameters` (búsqueda con Optuna, N trials), `train_model` (entrena sklearn, guarda `.pkl`, devuelve métricas). |
| `mlflow_tools.py` | `log_experiment` (loguea modelo+métricas a MLflow), `get_best_run` (consulta los mejores runs), `register_model` (registra en Model Registry), `set_model_alias` (asigna champion/challenger). |

### `prompts/` — Plantillas de prompts en YAML

Separar prompts del código permite editarlos sin tocar Python y ver diffs limpios en git.

| Fichero | Propósito |
|---------|-----------|
| `loader.py` | `get_prompt(name)` — carga el YAML y devuelve un `PromptTemplate` de LangChain. |
| `supervisor.yaml` | System prompt del supervisor: roles de los 4 agentes + reglas de enrutamiento del pipeline. |
| `data_agent.yaml` | System prompt del agente de validación: proceso de validación en 5 pasos. |
| `training_agent.yaml` | System prompt del agente de entrenamiento: selección de modelo, tuning, entrenamiento, logging. |
| `evaluation_agent.yaml` | System prompt del agente de evaluación: criterios de promoción (accuracy ≥ 0.80, F1 ≥ 0.75). |
| `deployment_agent.yaml` | System prompt del agente de despliegue: registro, alias challenger, espera aprobación humana. |

### `agents/` — Definición de los agentes especialistas

Cada agente es un `create_react_agent` de LangGraph: bucle ReAct (razona → llama herramienta → observa → repite).

| Fichero | Propósito |
|---------|-----------|
| `data_agent.py` | `build_data_agent()` — agente con herramientas de data_tools + evidently_tools. |
| `training_agent.py` | `build_training_agent()` — agente con herramientas de training_tools + mlflow_tools. |
| `evaluation_agent.py` | `build_evaluation_agent()` — agente con herramientas de mlflow_tools (consulta y compara runs). |
| `deployment_agent.py` | `build_deployment_agent()` — agente con herramientas de registro y alias de MLflow. |
| `supervisor.py` | `supervisor_node(state)` — nodo del supervisor: usa LLM con structured output (`RouterOutput`) para decidir el siguiente agente. Cada decisión se loguea con su razonamiento. |
| `registry.py` | `get_agent(name)` — factoría con `@lru_cache`: construye y cachea los agentes la primera vez. Evita reconstruirlos en cada llamada. |

### `graphs/` — Topología del grafo LangGraph

| Fichero | Propósito |
|---------|-----------|
| `mlops_graph.py` | **Fichero principal** — construye el `StateGraph` con los 5 nodos, los nodos wrapper de cada agente (que devuelven `Command(goto="supervisor")`), el `deployer_node` con `interrupt()` para HITL, y el grafo compilado `graph`. También contiene `main()` para ejecución por CLI. |
| `subgraphs/training_flow.py` | Reservado para un sub-workflow de reentrenamiento iterativo (si el modelo no pasa evaluación en el primer intento). No conectado al grafo principal por ahora. |

### `mcp_servers/` — Servidores MCP (Model Context Protocol)

Permiten a Claude Code acceder a MLflow y a los datasets directamente desde el IDE vía `/mcp`.

| Fichero | Propósito |
|---------|-----------|
| `mlflow_server.py` | Servidor FastMCP que expone: `list_experiments`, `get_experiment_runs`, `list_registered_models`. |
| `data_server.py` | Servidor FastMCP que expone: `list_datasets`, `preview_dataset`. |

---

## `dashboard/` — Interfaz Streamlit

| Fichero | Propósito |
|---------|-----------|
| `app.py` | Entry point de Streamlit. Configura la página y muestra el menú de navegación. |
| `pages/01_pipeline.py` | Lanza el pipeline desde la UI: selecciona dataset, ejecuta el grafo y muestra el log en tiempo real. |
| `pages/02_experiments.py` | Browser de experimentos MLflow: tabla de runs con métricas y parámetros. |
| `pages/03_monitoring.py` | Detección de drift: sube dos CSVs y muestra el reporte de Evidently AI. |
| `pages/04_chat.py` | Interfaz de chat para interactuar con los agentes en lenguaje natural. |
| `components/metrics_display.py` | Componente reutilizable: muestra un dict de métricas como tarjetas Streamlit. |
| `components/chat_interface.py` | Componente reutilizable: renderiza el historial de mensajes LangChain como hilo de chat. |

---

## `tests/` — Suite de pruebas

Espeja la estructura de `src/`. Cada módulo tiene su fichero de tests correspondiente.

| Fichero | Propósito |
|---------|-----------|
| `conftest.py` | Fixtures compartidas: `sample_csv` (CSV temporal con columna `target`), `mock_llm` (LLM mockeado). Comprobar aquí antes de crear fixtures nuevas. |
| `test_agents/test_supervisor.py` | Tests unitarios del supervisor: verifica el enrutamiento correcto sin llamadas reales al LLM. |
| `test_agents/test_data_agent.py` | Tests del builder del agente de validación. |
| `test_tools/test_data_tools.py` | Tests de las herramientas deterministas (no necesitan mock de LLM). |
| `test_tools/test_mlflow_tools.py` | Tests de las herramientas de MLflow con MLflow mockeado. |
| `test_graphs/test_mlops_graph.py` | Tests de estructura del grafo: verifica que compila y tiene los nodos esperados. |
| `test_integration/test_end_to_end.py` | Test end-to-end completo. Requiere `GITHUB_TOKEN` real y MLflow activo. Marcado `@pytest.mark.integration`. |

---

## `data/` — Datasets y esquemas

| Fichero | Propósito |
|---------|-----------|
| `samples/iris.csv` | Dataset Iris con 30 filas y columna `target` — listo para usar en demos y tests. |
| `schemas/dataset_schema.json` | Esquema esperado del dataset de entrada (columnas requeridas, tipos). |

---

## `scripts/` — Scripts de utilidad

| Fichero | Propósito |
|---------|-----------|
| `run_pipeline.py` | Ejecuta el pipeline completo desde CLI: `uv run python scripts/run_pipeline.py [path/dataset.csv]` |
| `seed_mlflow.py` | Crea 3 runs de ejemplo en MLflow (RandomForest baseline, RandomForest tuned, GradientBoosting) para tener datos de demo en la UI. |

---

## `.claude/` — Configuración de Claude Code

| Fichero | Propósito |
|---------|-----------|
| `settings.json` | Permisos: qué comandos Bash puede ejecutar Claude automáticamente (UV, git) y qué está denegado (rm -rf, pip, leer .env). |
| `.mcp.json` | Configura los servidores MCP para la sesión de Claude Code: mlflow-server, data-server y github. |
| `rules/langgraph-agents.md` | Reglas que se aplican automáticamente cuando Claude edita ficheros de `agents/` o `graphs/`. |
| `rules/testing.md` | Reglas que se aplican cuando Claude edita ficheros de `tests/`. |
| `commands/test-agent.md` | Slash command `/test-agent <nombre>` — ejecuta pytest filtrado por nombre de módulo. |
| `commands/run-pipeline.md` | Slash command `/run-pipeline [dataset]` — ejecuta el pipeline por CLI. |
| `skills/create-agent/SKILL.md` | Skill auto-invocable cuando se pide crear un nuevo agente — lista los 10 pasos a seguir. |

---

## Flujo de ejecución resumido

```
.env (GITHUB_TOKEN, GITHUB_MODEL)
    │
    ▼
config/settings.py          ← lee variables de entorno
    │
    ▼
utils/llm.py                ← crea ChatOpenAI apuntando a GitHub Models
    │
    ▼
agents/registry.py          ← construye los 4 agentes con sus tools y prompts
    │
    ▼
graphs/mlops_graph.py       ← StateGraph: supervisor → agentes → supervisor → END
    │
    ├── dashboard/app.py    ← UI Streamlit (invoca el grafo)
    └── scripts/run_pipeline.py  ← CLI (invoca el grafo)
```
