# Plan de Trabajo — Multi-Agent MLOps System (TFG)

> Cada historia de usuario representa una unidad funcional entregable.
> Se marca con `[x]` cuando está completamente implementada y probada.

---

## Epic 1 — Scaffolding y configuración del proyecto

| # | Historia | Estado |
|---|----------|--------|
| E1-1 | Como desarrollador, quiero una estructura de directorios organizada con UV src-layout para poder trabajar con imports limpios y builds reproducibles. | ✅ Done |
| E1-2 | Como desarrollador, quiero un `pyproject.toml` completo con todas las dependencias, ruff, mypy y pytest configurados para poder ejecutar `uv sync` y tener el entorno listo. | ✅ Done |
| E1-3 | Como desarrollador, quiero un `.env.example` con todas las variables necesarias documentadas para saber exactamente qué configurar antes de ejecutar el proyecto. | ✅ Done |
| E1-4 | Como desarrollador, quiero un `CLAUDE.md` con las convenciones del proyecto para que Claude Code tenga contexto en cada sesión. | ✅ Done |
| E1-5 | Como desarrollador, quiero `Dockerfile` y `docker-compose.yml` para poder levantar MLflow + Streamlit con un solo `docker compose up`. | ✅ Done |

---

## Epic 2 — Estado compartido y esquemas

| # | Historia | Estado |
|---|----------|--------|
| E2-1 | Como desarrollador, quiero un `AgentState` TypedDict con todos los campos del pipeline para que los nodos puedan leer y escribir estado de forma tipada. | ✅ Done |
| E2-2 | Como desarrollador, quiero esquemas Pydantic (`RouterOutput`, `ValidationResult`, `TrainingResult`, `EvaluationResult`) para que los outputs de LLM y herramientas sean estructurados y validados. | ✅ Done |

---

## Epic 3 — Herramientas deterministas (tools)

| # | Historia | Estado |
|---|----------|--------|
| E3-1 | Como agente de validación, quiero herramientas `load_dataset`, `validate_schema` y `check_missing_values` para poder inspeccionar un CSV antes de entrenamiento. | ✅ Done |
| E3-2 | Como agente de validación, quiero herramientas Evidently AI (`check_data_quality`, `check_data_drift`) para poder detectar problemas de calidad y drift estadístico. | ✅ Done |
| E3-3 | Como agente de entrenamiento, quiero una herramienta `tune_hyperparameters` con Optuna para poder encontrar los mejores hiperparámetros automáticamente. | ✅ Done |
| E3-4 | Como agente de entrenamiento, quiero una herramienta `train_model` que soporte `random_forest`, `gradient_boosting` y `logistic_regression` con sklearn para poder entrenar y guardar modelos. | ✅ Done |
| E3-5 | Como agente de evaluación, quiero herramientas MLflow (`log_experiment`, `get_best_run`) para poder registrar métricas y comparar runs. | ✅ Done |
| E3-6 | Como agente de despliegue, quiero herramientas `register_model` y `set_model_alias` para poder promover modelos al MLflow Model Registry. | ✅ Done |
| **E3-7** | **Como desarrollador, quiero tests unitarios para todas las herramientas deterministas para verificar que funcionan correctamente sin llamadas a LLM.** | ✅ Hecho |

---

## Epic 4 — Agentes especialistas

| # | Historia | Estado |
|---|----------|--------|
| E4-1 | Como pipeline, quiero un `data_agent` que valide un CSV usando sus herramientas y devuelva un informe de validación claro en linguaje natural. | ✅ Done (scaffold) |
| E4-2 | Como pipeline, quiero un `training_agent` que seleccione modelo, afine hiperparámetros, entrene y loguee en MLflow, devolviendo el `run_id`. | ✅ Done (scaffold) |
| E4-3 | Como pipeline, quiero un `evaluation_agent` que compare el modelo candidato con el baseline y emita una recomendación `promote/reject/retrain`. | ✅ Done (scaffold) |
| E4-4 | Como pipeline, quiero un `deployment_agent` que registre el modelo en MLflow Registry y solicite aprobación humana antes de promover a champion. | ✅ Done (scaffold) |
| **E4-5** | **Como desarrollador, quiero tests unitarios para cada agente mockeando el LLM para verificar que los builders funcionan y las herramientas están bien registradas.** | ⬜ Pendiente |
| **E4-6** | **Como desarrollador, quiero probar manualmente cada agente de forma aislada (sin supervisor) con un dataset real para validar los prompts.** | ⬜ Pendiente |

---

## Epic 5 — Supervisor y grafo principal

| # | Historia | Estado |
|---|----------|--------|
| E5-1 | Como pipeline, quiero un `supervisor_node` con structured output (`RouterOutput`) que enrute a los agentes correctos siguiendo las reglas del pipeline. | ✅ Done (scaffold) |
| E5-2 | Como pipeline, quiero un `StateGraph` compilado con los 5 nodos (supervisor + 4 agentes) que fluya `START → supervisor → agentes → supervisor → END`. | ✅ Done (scaffold) |
| E5-3 | Como pipeline, quiero un `deployer_node` con `interrupt()` que pause antes de la promoción a champion y espere aprobación humana. | ✅ Done (scaffold) |
| **E5-4** | **Como desarrollador, quiero un test que verifique que el grafo tiene los nodos esperados y compila sin errores.** | ⬜ Pendiente |
| **E5-5** | **Como desarrollador, quiero ejecutar el pipeline end-to-end con el dataset `iris.csv` y verificar que el supervisor enruta correctamente los 4 stages.** | ⬜ Pendiente |
| **E5-6** | **Como desarrollador, quiero verificar el flujo HITL: el pipeline pausa en `deployer_node`, recibe `Command(resume={"approved": True})` y completa la promoción.** | ⬜ Pendiente |

---

## Epic 6 — Integración MLflow

| # | Historia | Estado |
|---|----------|--------|
| **E6-1** | **Como usuario, quiero ejecutar `uv run python scripts/seed_mlflow.py` y ver 3 runs en la UI de MLflow para tener datos de demo.** | ⬜ Pendiente |
| **E6-2** | **Como agente de evaluación, quiero que `get_best_run` recupere el run champion actual y lo compare con el candidato correctamente.** | ⬜ Pendiente |
| **E6-3** | **Como agente de despliegue, quiero que `register_model` y `set_model_alias` funcionen contra un MLflow real (local o Docker).** | ⬜ Pendiente |

---

## Epic 7 — Dashboard Streamlit

| # | Historia | Estado |
|---|----------|--------|
| E7-1 | Como usuario, quiero una página "Pipeline" en Streamlit que me permita seleccionar un dataset y lanzar el pipeline desde la UI. | ✅ Done (scaffold) |
| E7-2 | Como usuario, quiero una página "Experiments" que muestre los runs de MLflow en una tabla con métricas y parámetros. | ✅ Done (scaffold) |
| E7-3 | Como usuario, quiero una página "Monitoring" donde pueda subir dos CSVs y ver el informe de drift de Evidently. | ✅ Done (scaffold) |
| E7-4 | Como usuario, quiero una página "Chat" donde pueda hablar con los agentes en lenguaje natural. | ✅ Done (scaffold) |
| **E7-5** | **Como usuario, quiero que la página Pipeline muestre el log en tiempo real (streaming) mientras los agentes trabajan.** | ✅ Done |
| **E7-6** | **Como usuario, quiero que el dashboard detecte cuando hay un `interrupt()` pendiente y muestre un botón de "Aprobar / Rechazar" despliegue.** | ✅ Done |
| **E7-7** | **Como usuario, quiero lanzar `uv run streamlit run dashboard/app.py` y que las 4 páginas carguen sin errores de import.** | ⬜ Pendiente |

---

## Epic 8 — Logging y observabilidad

> **Contexto:** El logging actual está valorado en 3/10. Solo escribe a stderr, tiene un bug de re-registro de handler, no persiste a fichero y el dashboard solo muestra "nodo completado". Estas historias lo llevan a un 8/10.

| # | Historia | Estado |
|---|----------|--------|
| **E8-1** | **Como desarrollador, quiero corregir el bug en `get_logger()` que llama `logger.remove()` en cada import para que el handler de loguru no se registre múltiples veces.** | ✅ Done |
| **E8-2** | **Como desarrollador, quiero que los logs se escriban a `logs/pipeline.log` con rotación diaria (máx. 7 días) además de a stderr, para poder revisar ejecuciones pasadas.** | ⬜ Pendiente |
| **E8-3** | **Como desarrollador, quiero un sink en memoria (`queue.Queue`) en `utils/logging.py` que acumule los log entries del run actual para que Streamlit los pueda leer en tiempo real.** | ⬜ Pendiente |
| **E8-4** | **Como usuario del dashboard, quiero que la página "Pipeline" muestre el contenido real de los mensajes de cada agente (no solo "nodo completado") mientras el grafo hace streaming.** | ⬜ Pendiente |
| **E8-5** | **Como usuario del dashboard, quiero una nueva página "Logs" (`05_logs.py`) que muestre los logs del run actual filtrables por nivel (DEBUG/INFO/WARNING/ERROR) y por agente.** | ⬜ Pendiente |
| **E8-6** | **Como desarrollador, quiero que el supervisor loguee cada decisión de enrutamiento (agente elegido + razonamiento) en INFO para poder auditar el comportamiento del pipeline.** | ⬜ Pendiente |
| **E8-7** | **Como desarrollador, quiero que los logs de herramientas deterministas incluyan duración de ejecución (ms) para poder identificar cuellos de botella.** | ⬜ Pendiente |

---

## Epic 9 — Servidores MCP

| # | Historia | Estado |
|---|----------|--------|
| E9-1 | Como desarrollador, quiero servidores MCP para MLflow y datos implementados con FastMCP. | ✅ Done (scaffold) |
| **E9-2** | **Como desarrollador, quiero levantar los servidores MCP y verificar que las herramientas aparecen en Claude Code (`/mcp`).** | ⬜ Pendiente |

---

## Epic 10 — Calidad y pruebas

| # | Historia | Estado |
|---|----------|--------|
| **E10-1** | **Como desarrollador, quiero ejecutar `uv run pytest -m "not integration"` y que todos los tests unitarios pasen (0 fallos).** | ⬜ Pendiente |
| **E10-2** | **Como desarrollador, quiero ejecutar `uv run ruff check .` sin errores de linting.** | ⬜ Pendiente |
| **E10-3** | **Como desarrollador, quiero ejecutar `uv run mypy src/` con 0 errores de tipo.** | ⬜ Pendiente |
| **E10-4** | **Como desarrollador, quiero ejecutar el test de integración end-to-end con `GITHUB_TOKEN` real y verificar que el pipeline completo funciona.** | ⬜ Pendiente |

---

## Epic 11 — Demo y entrega TFG

| # | Historia | Estado |
|---|----------|--------|
| **E11-1** | **Como evaluador del TFG, quiero ver una demo del pipeline completo: dataset → validación → entrenamiento → evaluación → aprobación humana → registro en MLflow.** | ⬜ Pendiente |
| **E11-2** | **Como evaluador, quiero ver el dashboard Streamlit con datos reales de MLflow mostrando experimentos, métricas y comparación de modelos.** | ⬜ Pendiente |
| **E11-3** | **Como evaluador, quiero ver el flujo HITL en acción: el pipeline pausado esperando aprobación y reanudándose tras la decisión humana.** | ⬜ Pendiente |
| **E11-4** | **Como desarrollador, quiero que `docker compose up` arranque todo el stack (MLflow + app) y la demo funcione sin configuración manual.** | ⬜ Pendiente |

---

## Leyenda

| Símbolo | Significado |
|---------|-------------|
| ✅ Done | Implementado y verificado |
| ⬜ Pendiente | Por implementar |
| 🔄 En curso | Actualmente en desarrollo |
| ⚠️ Bloqueado | Bloqueado por dependencia |

---

*Última actualización: 2026-04-19*
