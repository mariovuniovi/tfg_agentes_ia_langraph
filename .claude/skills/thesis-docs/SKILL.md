---
name: thesis-docs
description: Generate LaTeX thesis chapters and sections in academic Spanish for a Universidad de Oviedo TFG. Use when user wants to write, generate, or update a thesis chapter, section, or subsection.
---

# thesis-docs

## Output conventions

- Language: academic Spanish
- Format: standalone LaTeX — `\section{}` is top-level, `\subsection{}` below (no `\documentclass`)
- Bibliography: BibLaTeX + Biber; cite with `\cite{}` referencing `Biblio.bib`
- Figures: `\begin{figure}[H]` with `\caption{}` and `\label{fig:XX}`
- Code listings: `\begin{lstlisting}[language=Python]`
- Self-contained: compiles when `\input{}`-ed into a parent document

## Workflow

- [ ] **1. Ask which chapter/section to generate**

- [ ] **2. Read THESIS_INDEX.md**
  - Path: `docs/thesis/THESIS_INDEX.md`
  - If it does not exist, create it from the template below and ask user to confirm before proceeding

- [ ] **3. Brainstorm (personal voice)**
  - For personal-voice sections (Motivación, Conclusiones, Limitaciones): ask 2–4 targeted questions before generating. Examples: "¿Cuál fue la principal dificultad técnica?", "¿Qué te motivó a usar LangGraph?", "¿Qué limitaciones identificas?"
  - For purely technical sections: ask the user if they have notes or if the spec alone is sufficient. If they say the spec is sufficient, skip questions.
  - Wait for answers before writing

- [ ] **4. Read relevant specs**
  - Announce which spec files you will read before reading them (user can redirect)
  - Read from `docs/superpowers/specs/` first — these are dense summaries, prefer them over source code
  - Only read source code for specific implementation details not in specs

- [ ] **5. Announce output file**
  - Say: _"Writing to `docs/thesis/capXX_<topic>.tex`"_ and whether it already exists (overwrite)

- [ ] **6. Write the chapter file**
  - Overwrite `docs/thesis/capXX_<topic>.tex` in place — git history is the safety net

- [ ] **7. Update THESIS_INDEX.md**
  - Set chapter status to `done`
  - Add new terms to Terminology table
  - Add new labels to Cross-references table

## THESIS_INDEX.md template

Create this file if it does not exist, then ask user to confirm:

```markdown
# Thesis Index

## Chapter Outline
| # | Title (ES) | Subsections | File | Status |
|---|-----------|-------------|------|--------|
| 1 | Introducción | Motivación, Objetivos, Alcance del trabajo, Estructura del Trabajo | — | pending |
| 2 | Conceptos Básicos | MLOps, Sistemas multi-agente basados en LLMs, API de OpenAI como capa de acceso al modelo, LangGraph, Patrón ReAct y tools, Human-in-the-Loop, Principio de restraint agéntico: cuándo no usar agentes, Herramientas auxiliares | — | pending |
| 3 | Estudio de Alternativas | Alternativas a LangGraph, Alternativas al proveedor de modelos, Alternativas al almacén de experiencia | — | pending |
| 4 | Arquitectura | Visión general del sistema, Patrón supervisor-trabajador, Diseño del estado compartido, Flujo de ejecución end-to-end, Human-in-the-Loop en el nodo de despliegue | — | pending |
| 5 | Desarrollo e Implementación | Agente de validación de datos, Agente de entrenamiento, Agente de evaluación, Agente de despliegue, Pool de experiencia, API REST y frontend, Contenerización | — | pending |
| 6 | Resultados | Métricas de rendimiento del pipeline, Evaluación del pool de experiencia (benchmarks), Análisis del comportamiento agéntico | — | pending |
| 7 | Conclusiones y Trabajo Futuro | Conclusiones, Limitaciones, Escalabilidad del proyecto, Trabajo futuro | — | pending |

## Terminology (ES)
| English term | Spanish term used |
|---|---|

## Cross-references
| Label | Introduced in | Description |
|---|---|---|
```
