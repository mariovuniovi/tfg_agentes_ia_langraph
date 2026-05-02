# LangGraph supervisor pattern for multi-agent MLOps systems

**The LangGraph supervisor pattern uses a central LLM agent that dynamically routes tasks to specialized worker agents via tool-calling, and it is the most natural fit for your MLOps pipeline — but build it yourself with `StateGraph` rather than using the prebuilt `create_supervisor()` library.** LangChain themselves now recommend the manual approach for most use cases, and for a bachelor's thesis, a custom implementation demonstrates far deeper understanding of graph-based agent orchestration. Below is everything you need: the full API, architectural patterns, working code for your four-agent MLOps system, and the advanced features that will make your thesis stand out.

---

## How the supervisor actually works under the hood

The supervisor pattern in LangGraph centers on one critical insight: **the supervisor itself is a ReAct agent whose "tools" are handoff functions — one per worker agent**. When invoked, the supervisor LLM reads the full conversation history and decides which worker to delegate to by making a tool call (e.g., `transfer_to_data_validator`). That tool call returns a `Command(goto=agent_name)`, which directs the graph engine to execute that agent's subgraph. When the worker finishes, control returns to the supervisor, which reads the result and decides what to do next — call another agent, call the same one again, or respond directly to end the workflow.

The `langgraph-supervisor` package (v0.0.31, MIT license, Python ≥ 3.10) wraps this pattern into a single function call. Its full signature reveals the design:

```python
from langgraph_supervisor import create_supervisor

workflow = create_supervisor(
    agents=[agent1, agent2],          # Worker agents (any Pregel object)
    model=model,                       # LLM for routing decisions
    prompt="You are a supervisor...",  # System prompt guiding delegation
    output_mode="full_history",        # "full_history" or "last_message"
    parallel_tool_calls=False,         # Allow simultaneous agent dispatch
    supervisor_name="supervisor",      # Node name in the graph
    add_handoff_messages=True,         # Include handoff markers in history
    pre_model_hook=None,               # Hook before LLM (message trimming)
    post_model_hook=None,              # Hook after LLM (HITL, guardrails)
    state_schema=None,                 # Custom state beyond MessagesState
    response_format=None,              # Structured output schema
)
app = workflow.compile()
```

Internally, `create_supervisor()` builds a `StateGraph` with the supervisor as a central `create_react_agent` node, auto-generates `transfer_to_{agent_name}` handoff tools for each worker, and wires the graph so every worker routes back to the supervisor after completion. **The LLM makes every routing decision** — there are no hardcoded conditional edges. The supervisor loop terminates when the LLM responds without a tool call, producing a direct answer.

**Context passing** works through a shared `messages` list in the state. With `output_mode="full_history"`, all worker messages (including intermediate tool calls and reasoning) are appended to the shared history, so later agents can see what earlier agents produced. With `output_mode="last_message"` (the default), only each worker's final response is kept — more token-efficient but less context-rich.

---

## Three implementation approaches and which to choose

LangGraph currently offers three ways to build a supervisor system, each with distinct trade-offs:

**Approach 1: Prebuilt `create_supervisor()`** — The `langgraph-supervisor` library gets you a working multi-agent system in ~10 lines. It handles handoff tool creation, message routing, and history management automatically. However, LangChain's own README now states: *"We now recommend using the supervisor pattern directly via tools rather than this library for most use cases."* The library is maintained for backwards compatibility but is no longer the recommended path.

**Approach 2: Tool-wrapping (now officially recommended)** — Worker agents are wrapped as regular Python tools and given to a supervisor agent created with `create_react_agent`. Each tool function invokes a sub-agent internally and returns its final response as a string. This gives full control over what context each sub-agent receives.

**Approach 3: Custom `StateGraph` with `Command` routing** — You build the entire graph manually: define state, create worker nodes that return `Command(goto="supervisor")`, and write a supervisor node that uses structured output to pick the next worker. This is the most transparent and flexible approach.

**For your thesis, use Approach 3 with `create_react_agent()` for individual workers.** This gives you full control over the orchestration graph (which you need to explain and diagram), while still leveraging the well-tested ReAct loop for each specialist agent. The custom `StateGraph` approach is also aligned with LangChain's current recommendation and will not face deprecation risk.

---

## Supervisor versus every other LangGraph pattern

LangGraph now mirrors Anthropic's pattern taxonomy from their influential "Building Effective Agents" blog post. Understanding where the supervisor fits among these patterns is essential for your thesis framing.

| Pattern | Control flow | LLM calls for routing | Best for |
|---|---|---|---|
| **Prompt chaining** (`add_sequence`) | Static, linear | None | Known sequential steps like validate → train → evaluate |
| **Routing** (conditional edges) | Static branching | One classifier call | Categorizing inputs to specialized handlers |
| **Parallelization** (fan-out/fan-in) | Static parallel | None | Independent subtasks known in advance |
| **Orchestrator-worker** (`Send` API) | Dynamic parallel | One planning call | Dynamic task decomposition into parallel workers |
| **Supervisor** (handoff tools) | Dynamic sequential | One per routing step | Multi-step delegation with centralized control |
| **Evaluator-optimizer** (loop) | Conditional loop | One per iteration | Iterative quality refinement |
| **Swarm** (peer handoffs) | Decentralized | One per handoff | Conversational domain transfers |

A critical distinction for your thesis: **LangGraph's supervisor pattern is not the same as Anthropic's orchestrator-workers pattern.** The supervisor is a sequential delegator — it picks one agent at a time, waits for it to finish, then picks the next. Anthropic's orchestrator-workers pattern uses dynamic parallel decomposition (implemented in LangGraph via the `Send` API to spawn workers simultaneously). The supervisor is closer to a combination of Anthropic's routing and orchestrator patterns operating in a sequential loop.

For your MLOps pipeline, the supervisor pattern is the right choice because your stages have natural dependencies (you must validate data before training, train before evaluating, evaluate before deploying). The supervisor LLM can enforce this ordering while retaining the flexibility to skip steps, retry failed stages, or request additional validation — something a static `add_sequence()` pipeline cannot do. LangChain's own benchmarks show that **supervisor overhead accounts for ~30% of response time** due to the routing LLM calls, but this cost buys you dynamic decision-making that a fixed pipeline lacks.

---

## Complete MLOps supervisor implementation for your thesis

Here is a full custom implementation with four specialized agents, designed for GitHub Models' `gpt-4.1-mini`. This code is production-structured and thesis-ready:

```python
import operator
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel

# ── LLM Configuration (GitHub Models) ──────────────────────────
model = ChatOpenAI(
    model="gpt-4.1-mini",
    base_url="https://models.inference.ai.azure.com",
    api_key="YOUR_GITHUB_TOKEN",
)

# ── Shared State Definition ────────────────────────────────────
class MLOpsState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next: str
    pipeline_status: dict  # Track stage results

# ── Structured Routing Schema ──────────────────────────────────
class RouterOutput(BaseModel):
    next: Literal[
        "data_validator", "trainer", "evaluator", "deployer", "FINISH"
    ]
    reasoning: str

# ── Specialized Agent Tools ────────────────────────────────────
def validate_schema(data_path: str) -> str:
    """Validate dataset schema, check for missing values and types."""
    # Replace with real validation logic
    return "Schema valid: 10,000 rows, 15 features, 0 missing values, all types correct."

def check_data_drift(data_path: str) -> str:
    """Detect data drift using PSI between current and reference data."""
    return "No significant drift. PSI=0.04 (threshold: 0.1). Data is stable."

def train_model(model_type: str, hyperparameters: str) -> str:
    """Train an ML model with specified configuration."""
    return "Training complete. XGBoost model saved. Train accuracy: 0.94, val accuracy: 0.91."

def tune_hyperparameters(model_type: str, n_trials: int) -> str:
    """Run Optuna hyperparameter optimization."""
    return f"Optimization done ({n_trials} trials). Best: max_depth=6, lr=0.01, n_estimators=200."

def evaluate_model(model_path: str, test_data: str) -> str:
    """Evaluate model on held-out test set with multiple metrics."""
    return "Test metrics — Accuracy: 0.92, F1: 0.89, AUC-ROC: 0.95, Precision: 0.90, Recall: 0.88."

def compare_with_baseline(model_path: str, baseline_path: str) -> str:
    """Compare candidate model against production baseline."""
    return "Candidate outperforms baseline: +3% F1, +2% AUC-ROC. Recommend promotion."

def deploy_to_staging(model_path: str) -> str:
    """Deploy model to staging environment."""
    return "Model deployed to staging. Health check passed. Endpoint: /v2/predict."

def promote_to_production(model_path: str) -> str:
    """Promote staging model to production with canary rollout."""
    return "Canary deployment started: 10% traffic routed to new model. Monitoring active."

# ── Create Worker Agents ───────────────────────────────────────
data_validator = create_react_agent(
    model=model,
    tools=[validate_schema, check_data_drift],
    name="data_validator",
    prompt="You are a data validation specialist. Thoroughly validate datasets "
           "before they enter the ML pipeline. Always run both schema validation "
           "and drift detection. Report results clearly."
)

trainer = create_react_agent(
    model=model,
    tools=[train_model, tune_hyperparameters],
    name="trainer",
    prompt="You are an ML training specialist. First tune hyperparameters, "
           "then train the final model with optimal settings. Report training "
           "metrics and configuration used."
)

evaluator = create_react_agent(
    model=model,
    tools=[evaluate_model, compare_with_baseline],
    name="evaluator",
    prompt="You are a model evaluation specialist. Evaluate the trained model "
           "on test data AND compare against the production baseline. "
           "Recommend whether the model should be promoted."
)

deployer = create_react_agent(
    model=model,
    tools=[deploy_to_staging, promote_to_production],
    name="deployer",
    prompt="You are a deployment specialist. First deploy to staging and verify "
           "health, then promote to production with canary rollout. "
           "Report deployment status at each stage."
)

# ── Worker Wrapper Nodes ───────────────────────────────────────
def data_validator_node(state: MLOpsState) -> Command[Literal["supervisor"]]:
    result = data_validator.invoke(state)
    return Command(
        update={"messages": [
            HumanMessage(content=result["messages"][-1].content, name="data_validator")
        ]},
        goto="supervisor",
    )

def trainer_node(state: MLOpsState) -> Command[Literal["supervisor"]]:
    result = trainer.invoke(state)
    return Command(
        update={"messages": [
            HumanMessage(content=result["messages"][-1].content, name="trainer")
        ]},
        goto="supervisor",
    )

def evaluator_node(state: MLOpsState) -> Command[Literal["supervisor"]]:
    result = evaluator.invoke(state)
    return Command(
        update={"messages": [
            HumanMessage(content=result["messages"][-1].content, name="evaluator")
        ]},
        goto="supervisor",
    )

def deployer_node(state: MLOpsState) -> Command[Literal["supervisor"]]:
    result = deployer.invoke(state)
    return Command(
        update={"messages": [
            HumanMessage(content=result["messages"][-1].content, name="deployer")
        ]},
        goto="supervisor",
    )

# ── Supervisor Node ────────────────────────────────────────────
SUPERVISOR_PROMPT = """You are an MLOps pipeline supervisor managing 4 specialists:
- data_validator: validates data schema and checks for drift
- trainer: tunes hyperparameters and trains models
- evaluator: evaluates models and compares against baseline
- deployer: deploys to staging and promotes to production

RULES:
1. Always start with data_validator
2. Only proceed to trainer if validation passes
3. Only proceed to evaluator after training completes
4. Only proceed to deployer if evaluation recommends promotion
5. If any stage fails, report the failure and select FINISH
6. When the full pipeline completes successfully, select FINISH"""

members = ["data_validator", "trainer", "evaluator", "deployer"]

def supervisor_node(state: MLOpsState) -> Command[
    Literal["data_validator", "trainer", "evaluator", "deployer", "__end__"]
]:
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    response = model.with_structured_output(RouterOutput).invoke(messages)

    goto = response.next
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": response.next})

# ── Build the Graph ────────────────────────────────────────────
builder = StateGraph(MLOpsState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("data_validator", data_validator_node)
builder.add_node("trainer", trainer_node)
builder.add_node("evaluator", evaluator_node)
builder.add_node("deployer", deployer_node)
builder.add_edge(START, "supervisor")

graph = builder.compile()

# ── Execute ────────────────────────────────────────────────────
result = graph.invoke(
    {
        "messages": [HumanMessage(
            content="Run the full MLOps pipeline: validate data at /data/train.csv, "
                    "train an XGBoost model with 20 Optuna trials, evaluate against "
                    "the production baseline, and deploy if quality meets our bar."
        )],
        "next": "",
        "pipeline_status": {},
    },
    {"recursion_limit": 30},
)
```

**How results flow between agents**: Each worker node invokes its `create_react_agent`, extracts the final message, and wraps it as a `HumanMessage` with the agent's name. This message is appended to the shared `messages` list via the `operator.add` reducer. When the supervisor runs next, it sees the full history — including the worker's results — and uses `with_structured_output(RouterOutput)` to pick the next agent. The `RouterOutput` schema forces the LLM to choose exactly one of the four agents or `FINISH`, with a `reasoning` field that makes the decision auditable (excellent for your thesis).

---

## Advanced features that strengthen a thesis

**Preventing infinite loops** is critical for a robust system. LangGraph provides three layers of defense. The `recursion_limit` config parameter (default **25**) is a hard stop that raises `GraphRecursionError` when exceeded. For a softer approach, `RemainingSteps` is a managed state value that lets your supervisor check how many steps remain and gracefully terminate:

```python
from langgraph.managed.is_last_step import RemainingSteps

class MLOpsState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    next: str
    remaining_steps: RemainingSteps

def supervisor_node(state):
    if state["remaining_steps"] <= 2:
        return Command(goto=END)  # Graceful exit before hard limit
    # ... normal routing logic
```

**Failure handling** uses LangGraph's `RetryPolicy` at the node level for transient errors (API timeouts, rate limits), and state-based retry counters for semantic failures (model training didn't converge):

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "trainer", trainer_node,
    retry_policy=RetryPolicy(max_attempts=3)  # Retries on exceptions
)
```

For semantic retries, add an `error_count` to your state and check it in the supervisor — if evaluation fails twice, route to a fallback strategy rather than retrying indefinitely.

**Human-in-the-loop** works through the `interrupt()` function from `langgraph.types`. For your MLOps system, the most natural place is before deployment:

```python
from langgraph.types import interrupt

def deployer_node(state):
    approval = interrupt({
        "question": "Model passed evaluation. Approve deployment to production?",
        "metrics": state["messages"][-1].content
    })
    if approval != "approve":
        return Command(goto="supervisor", update={"messages": [
            HumanMessage(content="Deployment rejected by human reviewer.", name="deployer")
        ]})
    result = deployer.invoke(state)
    # ... continue with deployment
```

This requires a `checkpointer` (e.g., `InMemorySaver()`) attached via `graph.compile(checkpointer=checkpointer)` to persist state across the pause.

**Hierarchical teams** let you nest supervisors for complex organizations. A compiled supervisor graph is itself a `Pregel` object, so it can be passed as a worker to a parent supervisor — either via `create_supervisor()` or as a node in a custom `StateGraph`. This is powerful for thesis extensions: imagine a top-level orchestrator managing an "ML Team" supervisor (data validation + training) and a "Deployment Team" supervisor (evaluation + deployment + monitoring).

---

## Why custom beats prebuilt for your thesis

The decision between `create_supervisor()` and a custom `StateGraph` comes down to a clean trade-off matrix:

| Factor | Prebuilt `create_supervisor()` | Custom `StateGraph` |
|---|---|---|
| Lines of code | ~10-15 | ~50-100 |
| Routing transparency | Hidden (black-box LLM decisions) | Explicit (visible `RouterOutput` schema) |
| State customization | Limited to `MessagesState` + `state_schema` | Any `TypedDict` with custom keys and reducers |
| Error handling | Basic (tool-level only) | Full (retry policies, fallback nodes, error counters) |
| HITL integration | Via `post_model_hook` (some known issues) | Native `interrupt()` at any node |
| Academic value | Low (library call) | High (demonstrates understanding) |
| LangChain recommendation | "No longer recommended for most use cases" | Aligned with current best practices |

**The prebuilt library still has legitimate uses**: rapid prototyping, demos, and hierarchical nesting where the convenience outweighs the opacity. For your thesis, reference it in your related work section to show ecosystem awareness, but build your core system with `StateGraph`. Use `create_react_agent()` for the individual worker agents — this is uncontroversial and doesn't hide orchestration logic. The combination gives you the best of both worlds: a transparent, explainable orchestration layer with battle-tested agent internals.

One specific advantage for your thesis: the custom approach lets you define a `RouterOutput` Pydantic model with a `reasoning` field, making every routing decision auditable and loggable. This produces excellent material for your results chapter — you can show exactly why the supervisor chose to run evaluation after training, or why it decided to abort after a validation failure.

---

## Conclusion

The LangGraph supervisor pattern is a well-designed abstraction for multi-agent coordination, and it maps cleanly onto an MLOps pipeline where stages have natural dependencies. **Build your system using a custom `StateGraph` with `create_react_agent` workers** — this gives you the transparency your thesis demands, the flexibility to implement advanced features like HITL and structured retry logic, and alignment with LangChain's current recommendations. The `langgraph-supervisor` library (v0.0.31) is worth studying to understand the handoff tool mechanism and `output_mode` design, but its black-box nature limits both academic and production value. Your four-agent system (data validation → training → evaluation → deployment) orchestrated by a supervisor with structured output routing is a strong thesis architecture that demonstrates genuine understanding of agentic systems while solving a real MLOps problem.