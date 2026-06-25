# Thesis Index

## Chapter Outline
| # | Title (ES) | Subsections | File | Status |
|---|-----------|-------------|------|--------|
| 1 | Introducción | Motivación, Objetivos, Alcance del trabajo, Estructura del Trabajo | — | pending |
| 2 | Conceptos Básicos | MLOps, Sistemas multi-agente basados en LLMs, API de OpenAI como capa de acceso al modelo, LangGraph, Patrón ReAct y tools, Human-in-the-Loop, Principio de restraint agéntico: cuándo no usar agentes, Herramientas auxiliares | — | pending |
| 3 | Estudio de Alternativas | Alternativas a LangGraph, Alternativas al proveedor del modelo, Alternativas al almacén de experiencias, Alternativas al sistema de seguimiento de experimentos, Alternativas al stack de interfaz de usuario | cap03_alternativas.tex | done |
| 4 | Arquitectura | Visión general del sistema, Patrón controlador determinista y agentes especializados, Diseño del estado compartido, Flujo de ejecución end-to-end, Puertas de aprobación humana, Pool de experiencias y base de conocimiento estático, Infraestructura de despliegue, Tipos de nodos y contratos de ejecución | cap04_arquitectura.tex + cap04_nodos.tex | done |
| 5 | Desarrollo e Implementación | Agente de validación de datos, Agente de entrenamiento, Agente de evaluación, Agente de despliegue, Pool de experiencia, API REST y frontend, Contenerización | cap05_join_discovery.tex (parcial) | pending |
| 6 | Resultados | Introducción, Evaluación del pool de experiencias (benchmarks), Comportamiento agéntico: casos de estudio, Coste agéntico y supervisión humana | cap06_resultados.tex | done |
| 7 | Conclusiones y Trabajo Futuro | Conclusiones, Limitaciones, Escalabilidad del proyecto, Trabajo futuro | — | pending |

## Terminology (ES)
| English term | Spanish term used |
|---|---|
| workflow controller | controlador de flujo |
| experience pool | pool de experiencias |
| HITL gate | puerta de aprobación humana |
| structured output | salida estructurada |
| dataset profile | perfil del dataset |
| training plan | plan de entrenamiento |
| training executor | ejecutor de entrenamiento |
| evaluation report | informe de evaluación / informe de auditoría |
| champion model | modelo campeón |
| model registry | registro de modelos |
| experiment tracking | seguimiento de experimentos |
| planner tools | herramientas del planificador |
| tool trace | traza de herramientas |
| validation context | contexto de validación |
| experience record | registro de experiencia |
| join discovery | descubrimiento de join agéntico |
| row explosion | explosión de filas |
| unique ratio | ratio de unicidad |
| base dataset row count | cardinalidad del dataset canónico |
| nRMSE | error normalizado (RMSE / media objetivo × 100) |
| benchmark run | ejecución de benchmark |
| regime | régimen (estadístico / supervisado / paseo aleatorio) |
| join plan | plan de join |
| join candidate | candidato de join |
| base dataset | dataset base |
| SMAPE | SMAPE (error porcentual absoluto medio simétrico) |
| reasoning tokens | tokens de razonamiento |
| capacity check | chequeo de capacidad |
| retry (planner) | reintento (planner\_status=retry\_ok) |
| LLM share | cuota LLM |
| left coverage | cobertura izquierda (fracción de filas de hechos con correspondencia) |
| containment (FK⊊PK) | contención (FK subconjunto de PK) |
| orphan foreign key | clave foránea huérfana |
| referential integrity | integridad referencial |
| ordinal encoding | codificación ordinal |
| id column drop | descarte de columnas identificadoras |

## Cross-references
| Label | Introduced in | Description |
|---|---|---|
| sec:alternativas | cap03 | Sección raíz del Estudio de Alternativas |
| sec:alt-langgraph | cap03 | LangGraph vs AutoGen, CrewAI, Semantic Kernel, Airflow |
| sec:alt-proveedor | cap03 | GitHub Models vs OpenAI, Anthropic, Ollama, HuggingFace |
| sec:alt-experiencia | cap03 | SQLite vs PostgreSQL, BD vectorial, MLflow, JSON |
| sec:alt-mlflow | cap03 | MLflow vs W&B, DVC, ClearML, Neptune |
| sec:alt-frontend | cap03 | Next.js+FastAPI vs Streamlit, Gradio, Dash, Flask |
| tab:alt-langgraph | cap03 | Tabla comparativa frameworks de orquestación |
| tab:alt-proveedor | cap03 | Tabla comparativa proveedores de modelos |
| tab:alt-experiencia | cap03 | Tabla comparativa almacenes de experiencias |
| tab:alt-mlflow | cap03 | Tabla comparativa herramientas de seguimiento |
| tab:alt-frontend | cap03 | Tabla comparativa stacks de interfaz de usuario |
| sec:arquitectura | cap04 | Sección raíz del capítulo de Arquitectura |
| sec:vision-general | cap04 | Visión general del sistema y sus cuatro capas |
| sec:patron-supervisor | cap04 | Patrón controlador determinista + agentes especializados |
| sec:estado-compartido | cap04 | AgentState TypedDict y contrato del dataset |
| sec:flujo-ejecucion | cap04 | Flujo end-to-end del grafo (8 fases) + descubrimiento de join agentico en Fase 1 |
| fig:join-discovery-flow | cap04 | Subproceso de descubrimiento de join dentro de data\_validator |
| sec:hitl-gates | cap04 | Dos puertas HITL: dataset\_approval y deployment\_approval |
| sec:experience-pool | cap04 | SQLite pool, DatasetProfile, recuperación por similitud, ml\_rules.yaml |
| sec:infraestructura | cap04 | Docker compose: mlflow, api, frontend; separación de almacenamiento |
| fig:vision-general | cap04 | Diagrama ASCII de capas del sistema |
| fig:flujo-ejecucion | cap04 | Grafo de estados con transiciones principales |
| fig:docker-services | cap04 | Tabla de servicios Docker |
| sec:nodos-contratos | cap04_nodos | Subsección: tipos de nodos y contratos de ejecución |
| tab:nodos-resumen | cap04_nodos | Tabla resumen de los 9 nodos del grafo |
| sec:nodo-controller | cap04_nodos | workflow\_controller: máquina de estados determinista |
| sec:nodo-data-validator | cap04_nodos | data\_validator: agente ReAct de validación |
| sec:nodo-gate1 | cap04_nodos | dataset\_approval: puerta HITL 1 |
| sec:nodo-planner | cap04_nodos | planner: agente ReAct + salida estructurada |
| sec:nodo-executor | cap04_nodos | executor: entrenamiento determinista con Optuna |
| sec:nodo-evaluation | cap04_nodos | evaluation: módulo de promoción determinista |
| sec:nodo-report-writer | cap04_nodos | report\_writer: LLM de informe de auditoría |
| sec:nodo-gate2 | cap04_nodos | deployment\_approval: puerta HITL 2 |
| sec:nodo-deployer | cap04_nodos | deployer: registro en MLflow Model Registry |
| sec:experience-pool-detalle | cap04_experience_pool | Subsección principal: pool de experiencias |
| sec:experience-schema | cap04_experience_pool | ExperienceRecord: campos y semántica |
| sec:experience-storage | cap04_experience_pool | Tres tablas SQLite: experiences, candidate_results, model_artifacts |
| sec:dataset-profile | cap04_experience_pool | DatasetProfile: discretización por cubos |
| sec:similarity-function | cap04_experience_pool | Función de similitud ponderada; ratio y tier |
| sec:experience-value | cap04_experience_pool | Por qué es útil: evidencia empírica vs reglas estáticas |
| sec:experience-planner-integration | cap04_experience_pool | Integración con el planificador y validación anti-alucinación |
| sec:experience-current-state | cap04_experience_pool | 19 experiencias de referencia (benchmark) |
| tab:similarity-weights | cap04_experience_pool | Tabla de pesos de campos en la función de similitud |
| lst:similarity | cap04_experience_pool | Fragmento del cálculo de similitud ponderada |
| lst:workflow-controller | cap04 | Extracto del workflow\_controller |
| lst:hitl-gate | cap04 | Patrón del nodo dataset\_approval\_node |
| sec:resultados | cap06 | Sección raíz del capítulo de Resultados |
| sec:pool-evaluacion | cap06 | Evaluación del pool de experiencias (benchmark) |
| sec:pool-forecasting | cap06 | Resultados benchmark pronóstico (11 datasets) |
| sec:pool-clasif-regr | cap06 | Resultados benchmark clasificación y regresión (12 datasets) |
| tab:pool-forecasting | cap06 | Tabla de 11 datasets de pronóstico con RMSE y nRMSE |
| tab:pool-classification | cap06 | Tabla de 7 datasets de clasificación con macro-F1 |
| tab:pool-regression | cap06 | Tabla de 5 datasets de regresión con RMSE |
| sec:casos-estudio | cap06 | Comportamiento agéntico: seis casos de estudio interactivos |
| sec:caso-retrieval | cap06 | Recuperación de experiencias (sim 1,00 y 0,83) y razonamiento del planificador |
| sec:caso-joins | cap06 | Descubrimiento de 3 joins multi-tabla en Grid Demand |
| sec:caso-exog | cap06 | Adaptación de familia de modelos con exógenas (bakery) |
| sec:caso-robustez | cap06 | Robustez del controlador ante 3 entradas malformadas |
| sec:caso-generalidad | cap06 | Generalidad del sistema (remite a clasificación/regresión 6.1) |
| sec:caso-joins-imperfectos | cap06 | Joins imperfectos + categóricas en regresión (retail_sales: cobertura parcial, FK huérfanas, dedup, imputación, id-drop) |
| tab:joins-imperfectos | cap06 | Tabla de 3 joins de retail_sales (cobertura, contención, acción) |
| fig:cap06-join-imperfecto | cap06 | Plan de join imperfecto de retail_sales en la puerta de aprobación (captura UI) |
| sec:coste-agente | cap06 | Coste agéntico y supervisión humana |
| sec:coste-desglose | cap06 | Desglose de tiempo/coste por nodo (4 ejecuciones) |
| sec:coste-tamano | cap06 | Coste agéntico independiente del tamaño del dataset (mecanismo + caveat columnas) |
| lst:load-dataset | cap06 | Resumen de 451 caracteres que recibe el LLM (muestra de 3 filas fija) |
| sec:coste-variabilidad | cap06 | Coste estocástico: decisión estable, coste variable |
| sec:coste-hitl | cap06 | Tiempo de supervisión humana en las puertas HITL |
| sec:join-discovery | cap05_join_discovery | Sección dedicada: descubrimiento de joins en el data\_validator (mecanismo completo) |
| sec:join-principio | cap05_join_discovery | Principio de diseño: decisión agéntica vs medición determinista |
| sec:join-perfilado | cap05_join_discovery | Perfilado determinista previo (profile\_raw\_datasets) inyectado al agente |
| sec:join-candidatos | cap05_join_discovery | Selección de dataset base + propuesta de ≤20 candidatos por el agente |
| sec:join-evaluacion | cap05_join_discovery | evaluate\_join\_candidates: métricas deterministas y advertencias |
| sec:join-ejecucion | cap05_join_discovery | JoinPlan + execute\_join\_plan: garantías del merge determinista |
| sec:join-auditoria | cap05_join_discovery | JoinPlan auditado en la puerta dataset\_approval |
| tab:join-metricas | cap05_join_discovery | Tabla de métricas deterministas del evaluador de joins con umbrales |
| tab:casos-resumen | cap06 | Resumen de los 4 casos de pronóstico (sim, campeón, SMAPE) |
| tab:robustez | cap06 | Escenarios de error forzado y mensajes del controlador |
| tab:coste-desglose | cap06 | Tiempo de cómputo por nodo y coste USD por ejecución |
| tab:variabilidad | cap06 | 4 repeticiones de la panadería: coste variable |
| fig:cap06-join-plan | cap06 | Plan de join inferido para Grid Demand (captura UI) |
