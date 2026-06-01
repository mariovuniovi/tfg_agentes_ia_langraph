# Thesis Index

## Chapter Outline
| # | Title (ES) | Subsections | File | Status |
|---|-----------|-------------|------|--------|
| 1 | Introducción | Motivación, Objetivos, Alcance del trabajo, Estructura del Trabajo | — | pending |
| 2 | Conceptos Básicos | MLOps, Sistemas multi-agente basados en LLMs, API de OpenAI como capa de acceso al modelo, LangGraph, Patrón ReAct y tools, Human-in-the-Loop, Principio de restraint agéntico: cuándo no usar agentes, Herramientas auxiliares | — | pending |
| 3 | Estudio de Alternativas | Alternativas a LangGraph, Alternativas al proveedor del modelo, Alternativas al almacén de experiencias, Alternativas al sistema de seguimiento de experimentos, Alternativas al stack de interfaz de usuario | cap03_alternativas.tex | done |
| 4 | Arquitectura | Visión general del sistema, Patrón controlador determinista y agentes especializados, Diseño del estado compartido, Flujo de ejecución end-to-end, Puertas de aprobación humana, Pool de experiencias y base de conocimiento estático, Infraestructura de despliegue | cap04_arquitectura.tex | done |
| 5 | Desarrollo e Implementación | Agente de validación de datos, Agente de entrenamiento, Agente de evaluación, Agente de despliegue, Pool de experiencia, API REST y frontend, Contenerización | — | pending |
| 6 | Resultados | Métricas de rendimiento del pipeline, Evaluación del pool de experiencia (benchmarks), Análisis del comportamiento agéntico | — | pending |
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
| sec:flujo-ejecucion | cap04 | Flujo end-to-end del grafo (8 fases) |
| sec:hitl-gates | cap04 | Dos puertas HITL: dataset\_approval y deployment\_approval |
| sec:experience-pool | cap04 | SQLite pool, DatasetProfile, recuperación por similitud, ml\_rules.yaml |
| sec:infraestructura | cap04 | Docker compose: mlflow, api, frontend; separación de almacenamiento |
| fig:vision-general | cap04 | Diagrama ASCII de capas del sistema |
| fig:flujo-ejecucion | cap04 | Grafo de estados con transiciones principales |
| fig:docker-services | cap04 | Tabla de servicios Docker |
| lst:workflow-controller | cap04 | Extracto del workflow\_controller |
| lst:hitl-gate | cap04 | Patrón del nodo dataset\_approval\_node |
