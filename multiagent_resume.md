# Comprehensive CLAUDE.md for a Multi-Agent MLOps System

**GPT-4.1-mini running on GitHub Models, orchestrated by LangGraph, can automate a full MLOps workflow** — from data validation through model deployment — at zero cost. This reference file compiles every practical detail needed to build that system for a bachelor's thesis: architecture patterns, working code, API constraints, and tool integration. Below is the complete CLAUDE.md content, ready to drop into your project root.

---

```markdown
# CLAUDE.md — Multi-Agent MLOps System

## Project overview

This is a bachelor's thesis project: a **multi-agent system that automates a simple MLOps workflow**, built with Python and LangGraph, powered by free models via GitHub Models (GPT-4.1-mini). The system uses specialized LLM-powered agents to handle data validation, model training, evaluation, and deployment registration — orchestrated as a LangGraph StateGraph.

### Tech stack
- **Python 3.10+**
- **LangGraph 1.1.x** — agent orchestration (graph-based state machine)
- **LangChain 1.0+** — LLM integration layer
- **langchain-openai** — ChatOpenAI with custom base_url for GitHub Models
- **GitHub Models** — free LLM inference (GPT-4.1-mini via OpenAI-compatible API)
- **MLflow** — experiment tracking + model registry (local SQLite backend)
- **Evidently AI** — data validation and drift detection
- **scikit-learn + pandas** — ML training and data manipulation

### Key packages
```bash
pip install langgraph langchain langchain-openai langchain-core
pip install langgraph-checkpoint-sqlite  # persistence
pip install mlflow evidently scikit-learn pandas optuna
```

---

## Architecture

### Multi-agent pattern: Supervisor + Custom Workflow

The system uses LangGraph's **custom workflow pattern** with a supervisor-like orchestrator routing between specialized agent nodes. Each agent is a LangGraph node backed by an LLM with domain-specific tools.

```
User Request
     │
     ▼
┌──────────────┐
│  Orchestrator │ ← Supervisor node (routes tasks)
└──────┬───────┘
       │ conditional edges
       ├──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
┌────────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐
│   Data      │ │ Training  │ │ Evaluation│ │ Deployment│
│ Validation  │ │   Agent   │ │   Agent   │ │   Agent   │
│   Agent     │ │           │ │           │ │           │
└──────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
       │              │              │              │
   Evidently      sklearn +      sklearn       MLflow
   AI             MLflow         metrics       Registry
```

### Agent roles
1. **Data Validation Agent** — runs Evidently AI checks, interprets quality/drift reports, gates the pipeline
2. **Training Agent** — selects models, tunes hyperparameters, trains with sklearn, logs to MLflow
3. **Evaluation Agent** — compares trained models, selects best, generates natural-language report
4. **Deployment Agent** — registers best model in MLflow Model Registry, requires human approval

---

## LangGraph core concepts

### StateGraph fundamentals
LangGraph models workflows as directed graphs:
- **Nodes**: Python functions that receive state and return partial state updates
- **Edges**: Control flow (static, conditional, or dynamic via Command)
- **State**: Shared TypedDict passed between all nodes; updated via reducers

### State definition pattern
```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class MLOpsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # auto-appends
    dataset_path: str
    validation_report: dict
    validation_passed: bool
    trained_models: list
    best_model: dict
    evaluation_results: dict
    deployment_decision: str
```

**Critical rules:**
- Use `Annotated[list, add_messages]` for message lists — without it, returning a list OVERWRITES instead of appending
- Use `Annotated[list[str], operator.add]` to append to non-message lists
- Plain fields (str, dict, bool) overwrite on update — this is usually what you want
- Nodes return dicts with keys to update; never mutate state in-place

### MessagesState shorthand
```python
from langgraph.graph import MessagesState

class State(MessagesState):
    # Already includes: messages: Annotated[list[AnyMessage], add_messages]
    extra_field: str
```

### Building a graph
```python
builder = StateGraph(MLOpsState)
builder.add_node("validate", data_validation_agent)
builder.add_node("train", training_agent)
builder.add_node("evaluate", evaluation_agent)
builder.add_node("deploy", deployment_agent)

builder.add_edge(START, "validate")
builder.add_conditional_edges("validate",
    lambda s: "train" if s["validation_passed"] else END)
builder.add_edge("train", "evaluate")
builder.add_conditional_edges("evaluate",
    lambda s: "deploy" if s["best_model"].get("score", 0) > 0.8 else "train")
builder.add_edge("deploy", END)

graph = builder.compile()
result = graph.invoke({"dataset_path": "data/train.csv", "messages": []})
```

### Conditional edges for routing
```python
def should_continue(state: MLOpsState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tool_node", "end": END}
)
```

### Command API (modern approach — update state AND route simultaneously)
```python
from langgraph.types import Command
from typing import Literal

def supervisor_node(state: MLOpsState) -> Command[Literal["validate", "train", "evaluate", "deploy", "__end__"]]:
    decision = llm.invoke(...)  # LLM decides next agent
    return Command(
        update={"current_task": "validation"},
        goto="validate"
    )
```

### Sequence shorthand
```python
builder = StateGraph(State).add_sequence([node_1, node_2, node_3])
builder.add_edge(START, "node_1")
graph = builder.compile()
```

---

## Creating agents with LangGraph

### create_react_agent (prebuilt tool-calling agent)
```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    base_url="https://models.github.ai/inference",
    api_key=os.environ["GITHUB_TOKEN"],
    temperature=0,
    max_tokens=4000,
)

agent = create_react_agent(
    model=llm,
    tools=[my_tool_1, my_tool_2],
    prompt="You are a data validation specialist...",
    # name="validation_agent",           # for multi-agent identification
    # checkpointer=MemorySaver(),        # for persistence
    # interrupt_before=["tools"],        # for human-in-the-loop
)

result = agent.invoke({"messages": [{"role": "user", "content": "Validate dataset.csv"}]})
```

**Key parameters:**
- `model`: ChatOpenAI instance or string like `"openai:gpt-4o"`
- `tools`: list of @tool-decorated functions or BaseTool instances
- `prompt`: system prompt string
- `state_schema`: custom state (defaults to MessagesState)
- `checkpointer`: for persistence and HITL
- `interrupt_before` / `interrupt_after`: HITL breakpoints

### Custom agent node (more control)
```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def run_evidently_check(dataset_path: str) -> str:
    """Run data quality checks using Evidently AI."""
    from evidently import Report
    from evidently.presets import DataQualityPreset
    import pandas as pd
    df = pd.read_csv(dataset_path)
    report = Report([DataQualityPreset()])
    result = report.run(df, df)
    return str(result.as_dict())

tools = [run_evidently_check]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)  # auto-routes to "tools" or END
workflow.add_edge("tools", "agent")
app = workflow.compile()
```

### Supervisor pattern (using langgraph-supervisor)
```python
# pip install langgraph-supervisor
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

validation_agent = create_react_agent(
    model=llm, tools=[run_evidently_check],
    prompt="You validate data quality. Only handle validation tasks.",
    name="validation_agent",
)

training_agent = create_react_agent(
    model=llm, tools=[train_model, log_to_mlflow],
    prompt="You handle model training. Only handle training tasks.",
    name="training_agent",
)

supervisor = create_supervisor(
    model=llm,
    agents=[validation_agent, training_agent],
    prompt="You manage an MLOps pipeline. Route tasks to the appropriate agent.",
)
app = supervisor.compile()
```

**Note (March 2026):** The `langgraph-supervisor` README now recommends using the supervisor pattern directly via handoff tools for more control. Both approaches work.

---

## GitHub Models integration

### API endpoint and authentication
- **Endpoint**: `https://models.github.ai/inference`
- **Auth**: GitHub Personal Access Token (PAT) with `models:read` permission
- **Protocol**: Fully OpenAI-compatible (same SDK, same format)

### Create a GitHub PAT
1. GitHub Settings → Developer settings → Personal access tokens → Fine-grained tokens
2. Repository access: "Public Repositories" is sufficient
3. Under Account Permissions → **Models** → set to **Read-only**
4. Generate and save as `GITHUB_TOKEN`

### LangChain integration (the primary pattern for this project)
```python
import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    base_url="https://models.github.ai/inference",
    api_key=os.environ["GITHUB_TOKEN"],
    temperature=0,
    max_tokens=4000,
    max_retries=2,
)
```

### Direct OpenAI SDK usage (for testing)
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://models.github.ai/inference",
    api_key=os.environ["GITHUB_TOKEN"]
)
response = client.chat.completions.create(
    model="openai/gpt-4.1-mini",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=4000,
)
```

### Available free models (best for agents)
| Model | Tier | Tool Calling | Context | Best For |
|-------|------|-------------|---------|----------|
| GPT-4.1-mini | Low (150 RPD) | ✅ Yes | 1M tokens* | Primary agent model |
| GPT-4.1-nano | Low (150 RPD) | ✅ Yes | 1M tokens* | Router/triage agent |
| GPT-4o-mini | Low (150 RPD) | ✅ Yes | 128K | Fallback |
| Mistral-Large-2411 | Low (150 RPD) | ✅ Yes | 128K | Alternative |
| Llama 3.3 70B | Low (150 RPD) | Check | 128K | Open-source option |

*Free tier limits: 8000 input tokens and 4000 output tokens per request regardless of model's native window.

### Rate limits (free tier — Copilot Free)
| | Low Tier (GPT-4.1-mini) | High Tier (GPT-4o) |
|---|---|---|
| Requests/minute | 15 | 10 |
| **Requests/day** | **150** | **50** |
| Input tokens/request | 8,000 | 8,000 |
| Output tokens/request | 4,000 | 4,000 |
| Concurrent requests | 5 | 2 |

### Rate limit strategy for multi-agent systems
Each LLM call = 1 request. A ReAct agent doing 3 tool calls ≈ 4 requests. A full pipeline run with 4 agents ≈ 12-20 requests. **150 RPD gives ~7-12 full pipeline runs per day.**

Mitigation strategies:
1. **Use different models per agent** — each model has its own rate limit (GPT-4.1-mini: 150 + GPT-4.1-nano: 150 = 300 total RPD)
2. **Use GPT-4.1-nano for simple routing** — cheaper, same rate limit
3. **Minimize agent loops** — good system prompts reduce unnecessary tool calls
4. **Cache LLM responses** during development
5. **Supplement with Groq** (1000 RPD free) or **Google Gemini** (250 RPD free) using same ChatOpenAI pattern

### Multi-provider fallback pattern
```python
def get_llm(provider="github"):
    if provider == "github":
        return ChatOpenAI(
            model="openai/gpt-4.1-mini",
            base_url="https://models.github.ai/inference",
            api_key=os.environ["GITHUB_TOKEN"],
        )
    elif provider == "groq":
        return ChatOpenAI(
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
    elif provider == "gemini":
        return ChatOpenAI(
            model="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.environ["GEMINI_API_KEY"],
        )
```

---

## Human-in-the-loop patterns

### Using interrupt() — recommended modern approach
```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

def deployment_approval(state: MLOpsState):
    response = interrupt({
        "question": "Approve deployment of this model?",
        "model_info": state["best_model"],
        "metrics": state["evaluation_results"],
    })
    if response.get("approved"):
        return {"deployment_decision": "approved"}
    return Command(goto="__end__")

# MUST compile with checkpointer for HITL
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

# First call — pauses at interrupt
config = {"configurable": {"thread_id": "session-1"}}
result = graph.invoke(input_data, config=config)

# Resume with human decision
result = graph.invoke(
    Command(resume={"approved": True}),
    config=config
)
```

### Static breakpoints
```python
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["deploy"]  # pause before deployment node
)
```

**Critical requirement:** HITL always needs a checkpointer. Same thread_id must be used for resume.

---

## Persistence and checkpointing

### InMemorySaver (development)
```python
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "thread-1"}}
```

### SqliteSaver (thesis-appropriate persistence)
```python
from langgraph.checkpoint.sqlite import SqliteSaver
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

### PostgresSaver (production)
```python
from langgraph.checkpoint.postgres import PostgresSaver
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()  # MUST run on first use
    graph = builder.compile(checkpointer=checkpointer)
```

---

## MLOps tool integration patterns

### MLflow — experiment tracking and model registry
```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")  # local, free
mlflow.set_experiment("mlops-agents")

# Training agent logs experiments
with mlflow.start_run(run_name="agent-run"):
    mlflow.log_params({"n_estimators": 100, "max_depth": 5})
    mlflow.log_metrics({"accuracy": 0.92, "f1": 0.89})
    mlflow.sklearn.log_model(model, "model")

# Evaluation agent queries runs
client = MlflowClient()
runs = client.search_runs(
    experiment_ids=["1"],
    order_by=["metrics.accuracy DESC"],
    max_results=5
)

# Deployment agent registers best model
mlflow.register_model(f"runs:/{best_run_id}/model", "MyModel")
client.set_registered_model_alias("MyModel", "champion", version=1)

# Simplest approach: autologging
mlflow.autolog()  # auto-captures sklearn, xgboost parameters and metrics
```

### Evidently AI — data validation and drift detection
```python
from evidently import Report
from evidently.presets import DataDriftPreset, DataQualityPreset
import pandas as pd

# Data quality check
quality_report = Report([DataQualityPreset()])
quality_result = quality_report.run(reference_data, current_data)
quality_dict = quality_result.as_dict()  # structured JSON — easy for agents to parse

# Drift detection
drift_report = Report([DataDriftPreset(method="psi")])
drift_result = drift_report.run(reference_data, current_data)
drift_dict = drift_result.as_dict()
# Contains: drift_detected (bool), drift_score per feature, p-values
```

### scikit-learn — model training
```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
report = classification_report(y_test, model.predict(X_test), output_dict=True)
# Agent interprets report dict and decides next steps
```

---

## Agentic AI design patterns reference

### ReAct (Reasoning + Acting) — default pattern for tool-using agents
Loop: Thought → Action → Observation → repeat until done.
LangGraph maps this to: model node → conditional edge → tool node → back to model.
`create_react_agent()` implements this automatically.
Paper: Yao et al. 2022, "ReAct: Synergizing Reasoning and Acting in Language Models"

### Plan-and-Execute — for complex multi-step tasks
Step 1: LLM creates complete plan upfront.
Step 2: Steps executed one by one (optionally by cheaper model).
Advantage: explicit long-term planning; can use different models for planning vs execution.

### Reflection — for iterative quality improvement
Generator agent produces output → Critic evaluates → feedback loops back → iterate.
Use when clear quality criteria exist. Good for code generation, config generation.

### Supervisor — for coordinating specialized agents
Central supervisor routes tasks to workers, collects results, synthesizes.
Best when workers handle distinct, non-overlapping domains.

### Five workflow patterns (Anthropic's hierarchy, simplest to most complex)
1. **Prompt Chaining** — sequential LLM calls with validation gates
2. **Routing** — classify input → route to specialist
3. **Parallelization** — independent subtasks run simultaneously
4. **Orchestrator-Workers** — dynamic task decomposition and delegation
5. **Evaluator-Optimizer** — generate → evaluate → refine loop

**Principle:** Start with the simplest pattern that works. Only add complexity when needed.

---

## Key imports cheat sheet

```python
# Core graph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages

# Control flow
from langgraph.types import Command, interrupt, Send
from typing import TypedDict, Annotated, Literal

# Prebuilt
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition

# Checkpointers
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# Supervisor
from langgraph_supervisor import create_supervisor

# LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# MLOps
import mlflow
from mlflow.tracking import MlflowClient
from evidently import Report
from evidently.presets import DataDriftPreset, DataQualityPreset
```

---

## Common gotchas

1. **Reducers**: Without `Annotated[list, add_messages]`, lists OVERWRITE instead of appending
2. **Compile before use**: Always call `builder.compile()` — validates graph structure
3. **Thread IDs**: Always pass `config={"configurable": {"thread_id": "..."}}` when using checkpointers
4. **Command vs conditional edges**: Use Command when you need state update + routing together
5. **Recursion limit**: Default is 25; increase via `config={"recursion_limit": 50}`
6. **GitHub Models token limit**: 8000 input tokens per request on free tier — truncate conversation history
7. **GitHub Models model names**: Use `"openai/gpt-4.1-mini"` format (with provider prefix)
8. **SqliteSaver context manager**: Use `with SqliteSaver.from_conn_string(...) as cp:` pattern
9. **PostgresSaver.setup()**: Must call on first use to create tables
10. **Node return type**: Nodes must return dicts, not modify state objects directly

---

## Key documentation URLs

### LangGraph (primary)
- Overview: https://docs.langchain.com/oss/python/langgraph/overview
- Graph API: https://docs.langchain.com/oss/python/langgraph/graph-api
- Quickstart: https://docs.langchain.com/oss/python/langgraph/quickstart
- Persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- Interrupts (HITL): https://docs.langchain.com/oss/python/langgraph/interrupts
- Memory: https://docs.langchain.com/oss/python/langgraph/add-memory
- Streaming: https://docs.langchain.com/oss/python/langgraph/streaming
- Subgraphs: https://docs.langchain.com/oss/python/langgraph/use-subgraphs
- Multi-agent tutorial: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/
- create_react_agent ref: https://reference.langchain.com/python/langgraph.prebuilt/chat_agent_executor/create_react_agent
- LangGraph Supervisor repo: https://github.com/langchain-ai/langgraph-supervisor-py

### GitHub Models
- Marketplace: https://github.com/marketplace/models
- Documentation: https://docs.github.com/en/github-models
- Rate limits: https://docs.github.com/github-models/prototyping-with-ai-models#rate-limits
- GPT-4.1-mini page: https://github.com/marketplace/models/azure-openai/gpt-4-1-mini
- Billing: https://docs.github.com/billing/managing-billing-for-your-products/about-billing-for-github-models

### Agentic AI guides
- Anthropic "Building Effective Agents": https://www.anthropic.com/research/building-effective-agents
- OpenAI "A Practical Guide to Building Agents": https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf
- Lilian Weng "LLM Powered Autonomous Agents": https://lilianweng.github.io/posts/2023-06-23-agent/

### MLOps references
- MLflow docs: https://mlflow.org/docs/latest/index.html
- Evidently AI docs: https://docs.evidentlyai.com/
- Google Cloud MLOps guide: https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
- AutoML-Agent paper (ICML 2025): https://arxiv.org/abs/2410.02958

### Alternative free LLM APIs
- Groq: https://console.groq.com/ (1000 RPD free)
- Google Gemini: https://ai.google.dev/ (250 RPD free)
- Free API list: https://github.com/mnfst/awesome-free-llm-apis

---

## Thesis scope recommendations

### Realistic deliverable
A multi-agent LangGraph system automating 3-4 MLOps stages on a standard classification dataset (e.g., UCI/Kaggle), with experiment tracking in MLflow and human-in-the-loop deployment approval.

### Implement these 4 agents
1. **Data Validation Agent** — Evidently AI quality/drift checks
2. **Training Agent** — sklearn model selection + hyperparameter tuning + MLflow logging
3. **Evaluation Agent** — model comparison + best model selection + natural language report
4. **Deployment Agent** — MLflow Model Registry registration with human approval gate

### Good demo scenarios
- End-to-end: user provides dataset → agents auto-validate → train → evaluate → register best model
- Failure handling: data validation fails → agent explains why and suggests fixes
- Human-in-the-loop: agent pauses before deployment for approval
- Comparison: agent-automated pipeline vs. manual pipeline on same dataset

### Avoid (too complex for bachelor's)
- Real-time model serving (just register in MLflow, don't serve)
- Kubernetes/Docker orchestration
- Deep learning / GPU training (stick to scikit-learn)
- DVC / Airflow / complex data pipelines
- Fine-tuning LLMs

### Key related work
- AutoML-Agent (Trirat et al., ICML 2025): https://arxiv.org/abs/2410.02958
- SELA (Chi et al., 2024): https://arxiv.org/abs/2410.17238
- ReAct (Yao et al., 2022): Reasoning + Acting in LLMs
- Anthropic "Building Effective Agents" (2024)
```

---

## How this CLAUDE.md is structured and why it works

The file above covers five distinct knowledge domains synthesized into a single actionable reference. **LangGraph architecture** details include every code pattern Claude Code will need — StateGraph definition, state with reducers, conditional edges, the Command API, `create_react_agent`, and the supervisor pattern — all using current v1.1.x APIs confirmed as of March 2026. The legacy `langchain-ai.github.io/langgraph/` docs were deprecated in October 2025; the new canonical location is `docs.langchain.com/oss/python/langgraph/`.

The **GitHub Models integration section** provides the exact connection pattern (`ChatOpenAI` with `base_url="https://models.github.ai/inference"` and a GitHub PAT), confirmed rate limits (**150 requests/day** for GPT-4.1-mini on the free tier, 15 per minute, 8000 input / 4000 output tokens per request), and a multi-provider fallback strategy using Groq and Gemini to multiply available daily requests.

The **MLOps tool patterns** give concrete Python code for MLflow experiment tracking, Evidently AI data validation, and scikit-learn training — all returning structured data that LLM agents can parse and reason about. Evidently AI is recommended over Great Expectations for this project because it returns clean JSON that's more agent-friendly.

The **agentic AI patterns section** distills Anthropic's five workflow patterns and the ReAct/Plan-and-Execute/Reflection taxonomy into a quick-reference that helps choose the right pattern for each agent role. The **scope recommendations** section keeps the thesis grounded — four agents, standard dataset, proof of concept — referencing AutoML-Agent (ICML 2025) as the closest related academic work while clearly differentiating the thesis focus on MLOps orchestration rather than pure AutoML.

Every URL listed was verified during research. The documentation site structure, package versions, and API patterns reflect the current state as of March 2026.