# AI Picker & Packer Backend — Implementation Plan

> Reference document capturing the design agreed during the planning session.
> Use this as the starting point for implementation in future sessions.

## Context

This is a greenfield project. `CLAUDE.md` describes the back-end "brain" for a picker-and-packer robot: it accepts orders from an app front-end, plans the fastest in-store route, coordinates with (external) movement and arm-control systems, and commits inventory changes only when picking succeeds. The repo currently contains only `CLAUDE.md`, `LICENSE`, `.gitignore`, this plan file, and an empty Python 3.13 venv — there is nothing to refactor or reuse, so this plan is a from-scratch scaffold.

The intended stack (per `CLAUDE.md`): **LangChain** (LLM inference, via **OpenRouter**), **LangGraph** (agent orchestration), **n8n** (information flow between DB / app / brain), **SQLite** (local `.db` for tech demo).

## Decisions Already Made

- **Position model:** graph of nodes + edges in SQLite. Demo runs on a one-row, one-item-per-shelf store, but the schema scales to multi-row stores, one-way aisles, junctions, and special locations (start, pack station, charger) without migration.
- **What is LLM-driven vs deterministic:**
  - **LLM-driven (two call sites):**
    1. **Triage agent** — drives the orchestration loop by emitting structured tool calls (`plan_route`, `move_to`, `pick`, `mark_item_failed`, `complete_order`, `abort_order`). Reasons about error recovery, partial failures, and order completion.
    2. **Route-planning strategy selector** — picks one of several deterministic algorithms based on a structured summary of order shape.
  - **Deterministic:** the route algorithms themselves, the movement-comm HTTP exchange, the arm-comm HTTP exchange, and the database layer. Routing correctness and hardware communication never depend on an LLM.
- **LLM provider:** **OpenRouter** (OpenAI-compatible API). LangChain integration via `langchain-openai`'s `ChatOpenAI` configured with `base_url="https://openrouter.ai/api/v1"` and an `OPENROUTER_API_KEY`. Lets us swap underlying models (Claude, GPT, Llama, etc.) by changing one config string. Default model in `.env.example`: `anthropic/claude-sonnet-4` (strongest tool-use behavior on OpenRouter).
- **External mocks:** standalone mock HTTP servers (one for movement, one for arm). The comm agents make real HTTP calls so the network boundary is exercised end-to-end.
- **Front door:** both. FastAPI is the canonical backend; n8n is a thin wrapper that handles the reserve→commit DB choreography and forwards to FastAPI. Either path works; FastAPI direct gives a fast dev loop, n8n gives the production-shaped path.
- **Algorithm pool:** three algorithms — `nearest_neighbor`, `held_karp`, `two_opt` — covering the speed×quality plane. The graph-based distance matrix doesn't change which algorithms are appropriate; Christofides was considered (now viable thanks to guaranteed triangle inequality on shortest-path distances) but rejected as redundant against Held–Karp on small inputs and 2-opt on medium ones, and too noisy for the LLM's choice space.
- **Demo seed dataset:** the **"interesting"** preset only — ~12 SKUs along a single row with non-uniform spacing. Big enough that nearest-neighbor and Held–Karp can produce different routes, so the strategy selector earns its keep on camera.

## High-Level Architecture

```
[App Frontend]  <--HTTP-->  [n8n workflows]  <--HTTP-->  [FastAPI: brain]
       \                          |                            |
        \________ direct _________|                            |
                                  v                            v
                             [SQLite .db]  <-------------> [LangGraph multi-agent]
                                                            - Triage  (LLM via OpenRouter)
                                                                tools: plan_route, move_to,
                                                                       pick, mark_item_failed,
                                                                       complete_order, abort_order
                                                            - Route Planner
                                                                - Strategy selector (LLM via OpenRouter)
                                                                - Algorithm pool (deterministic)
                                                            - Movement Comm  (deterministic HTTP)
                                                            - Arm-Control Comm (deterministic HTTP)

                  [Mock Movement Server] <--HTTP-- Movement Comm Agent
                  [Mock Arm Server]      <--HTTP-- Arm-Control Comm Agent
```

Two-phase inventory pattern: n8n stages a deduction at order intake, the brain runs the pick workflow, and only on success does n8n commit the deduction (otherwise rollback and notify the app).

## Components & Files

### 1. Project bootstrap
- `requirements.txt` — `langchain`, `langgraph`, `langchain-openai`, `fastapi`, `uvicorn`, `pydantic`, `httpx`, `python-dotenv`, `pytest`. (Stdlib `sqlite3` is enough; no SQLAlchemy needed for the demo.)
- `.env.example` — `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` (e.g. `anthropic/claude-sonnet-4`), `MOVEMENT_SYSTEM_URL`, `ARM_SYSTEM_URL`, `DB_PATH`.
- `pyproject.toml` with a simple `src/` layout.

### 2. Database layer (`src/db/`)

**Graph-based schema** (`schema.sql`):

```sql
-- Every place the robot can stop or traverse.
CREATE TABLE nodes (
  node_id    INTEGER PRIMARY KEY,
  label      TEXT NOT NULL,        -- "A1-S03", "junction-12", "start", "pack"
  node_type  TEXT NOT NULL,        -- 'shelf' | 'junction' | 'start' | 'pack_station' | 'charger'
  x          REAL,                 -- optional cartesian (visualization + A* heuristic)
  y          REAL
);

-- Directed edges. Symmetric connections = two rows. Lets one-way aisles
-- and asymmetric costs (uphill / congestion) drop in without schema changes.
CREATE TABLE edges (
  edge_id    INTEGER PRIMARY KEY,
  from_node  INTEGER NOT NULL REFERENCES nodes(node_id),
  to_node    INTEGER NOT NULL REFERENCES nodes(node_id),
  cost       REAL    NOT NULL,
  UNIQUE (from_node, to_node)
);
CREATE INDEX idx_edges_from ON edges(from_node);

-- Products live on a node. A node may hold many SKUs in production;
-- the demo just happens to be 1:1.
CREATE TABLE products (
  sku_id           TEXT PRIMARY KEY,
  name             TEXT NOT NULL,
  node_id          INTEGER NOT NULL REFERENCES nodes(node_id),
  quantity_on_hand INTEGER NOT NULL
);
CREATE INDEX idx_products_node ON products(node_id);

-- Order lifecycle.
CREATE TABLE orders (
  order_id   INTEGER PRIMARY KEY,
  status     TEXT NOT NULL,        -- 'staged' | 'in_progress' | 'completed' | 'failed'
  created_at TEXT NOT NULL
);

CREATE TABLE order_items (
  order_id      INTEGER NOT NULL REFERENCES orders(order_id),
  sku_id        TEXT    NOT NULL REFERENCES products(sku_id),
  qty_requested INTEGER NOT NULL,
  qty_reserved  INTEGER NOT NULL,
  picked        INTEGER NOT NULL DEFAULT 0,  -- BOOL
  PRIMARY KEY (order_id, sku_id)
);

-- Staged-but-not-yet-committed deductions, applied to quantity_on_hand
-- on success and discarded on failure.
CREATE TABLE inventory_reservations (
  order_id INTEGER NOT NULL REFERENCES orders(order_id),
  sku_id   TEXT    NOT NULL REFERENCES products(sku_id),
  qty      INTEGER NOT NULL,
  PRIMARY KEY (order_id, sku_id)
);
```

`database.py` — connection helper, context-managed transactions, and functions:
- `reserve_stock(order)` / `commit_reservation(order_id)` / `rollback_reservation(order_id)`
- `get_nodes_for_skus(sku_ids) -> dict[sku_id, node_id]`
- `load_graph() -> (nodes, edges)` — returns the static graph for the route planner to load once at startup
- `mark_item_picked(order_id, sku_id)`

`seed.py` — populates the **"interesting"** demo store: a single row of ~12 shelf nodes with **non-uniform spacing** (varied edge costs), plus a `start` node at one end and a `pack_station` node at the other. One SKU per shelf node. The non-uniform spacing is what lets nearest-neighbor and Held–Karp produce different routes for the same order, making the LLM strategy selector visibly relevant.

### 3. LangGraph "brain" (`src/agents/`)
- `state.py` — typed `OrderState` (TypedDict): `order_id`, `items`, `node_route` (full step-by-step path once planned), `sku_visit_order`, `chosen_algorithm`, `algorithm_reasoning`, `current_index`, `picked_items`, `failed_items`, `last_action`, `last_action_result`, `errors`, `tick_count`, `status`.
- `llm.py` — thin OpenRouter client wrapper: `get_llm(model: str, temperature: float = 0)` returning a `ChatOpenAI` configured with `base_url="https://openrouter.ai/api/v1"`. Used by both LLM call sites. A `LLMClient` protocol lets tests inject scripted responses.
- `triage/` — **LLM-driven** orchestration agent:
  - `tools.py` — Pydantic models for the structured tool surface the LLM can emit:
    - `PlanRoute()` — kick off the route-planner subgraph
    - `MoveTo(node_id: int)` — invoke the movement comm node
    - `Pick(sku_id: str)` — invoke the arm-control comm node
    - `MarkItemFailed(sku_id: str, reason: str)` — record a failure, continue
    - `CompleteOrder()` — terminal success
    - `AbortOrder(reason: str)` — terminal failure (triggers rollback)
  - `prompts.py` — system prompt describing the role, the available tools, the safety rules ("never invent SKU IDs", "always plan route before moving", "abort if more than 30% of items fail").
  - `agent.py` — the LangGraph node. On each tick: serializes a redacted view of `OrderState` (no raw graph data — only what the LLM needs to decide), calls the LLM with structured-output schema = `Union[PlanRoute, MoveTo, Pick, MarkItemFailed, CompleteOrder, AbortOrder]`, validates the response, and writes the chosen action into state. Conditional edges in the parent graph dispatch to the corresponding deterministic node and loop back to triage.
  - **Safety nets:** max 50 ticks per order (then forced abort), invalid tool output → forced abort with logged reason, temperature=0 for reproducibility.
- `route_planner/` — directory:
  - `graph.py` — loads the static node/edge graph at startup; provides `shortest_paths(source, targets)` (Dijkstra) and `build_distance_matrix(stops)` which returns both the `k×k` cost matrix *and* the cached pairwise paths so the final route can be stitched back into a step-by-step movement plan.
  - `algorithms.py` — pure functions, each takes a distance matrix and returns a stop ordering:
    - `nearest_neighbor(matrix)` — O(n²), fast, ~25% from optimal on average. Default for small/simple orders.
    - `held_karp(matrix)` — exact optimum, exponential. Viable only for ≤ ~12 stops.
    - `two_opt(matrix)` — polynomial local search, near-optimal. Middle ground for medium orders.
  - `selector.py` — the **strategy-selector LLM call**. Receives a structured summary of the order (item count, bounding box, max pairwise distance, mean distance) — *not* raw positions, so the model can't be tempted to plan the route itself. Returns a Pydantic-constrained `{algorithm: "nearest_neighbor" | "held_karp" | "two_opt", reasoning: str}`. Falls back to `nearest_neighbor` on any LLM error or unknown algorithm name.
  - `node.py` — the LangGraph node that wires `build_distance_matrix → selector → chosen_algorithm → stitch_full_path`, writing both the chosen algorithm name and its reasoning into state so the demo can surface the LLM's decision.
- `movement_comm.py` — **deterministic** node that POSTs the target node to the movement system endpoint, awaits "arrived" or an error, and writes the result into `last_action_result` for triage to consume on the next tick.
- `arm_comm.py` — **deterministic** node that POSTs a "pick this SKU" command to the arm-control endpoint, awaits "picked" or an error, and writes the result into `last_action_result`.
- `graph.py` — wires the LangGraph `StateGraph`. Triage is the central hub: every other node returns control to triage, which calls the LLM to decide the next action. Conditional edges from triage dispatch to `plan_route`, `move`, `pick`, or terminal states based on the LLM's tool choice. The deterministic action nodes never branch on their own — they always loop back to triage.

### 4. FastAPI surface (`src/api/`)
- `main.py` — endpoints:
  - `POST /orders/process` — n8n calls this with the order; runs the LangGraph workflow and returns the final state (including the chosen algorithm + reasoning).
  - `GET /orders/{id}` — status polling for n8n / app.
  - `POST /orders/{id}/cancel` — abort + rollback.
- `models.py` — Pydantic schemas for orders / items / responses.

### 5. Mock external services (`src/mocks/`)
- `movement_server.py` — tiny FastAPI app: accepts a target node, sleeps briefly, returns "arrived".
- `arm_server.py` — tiny FastAPI app: accepts a pick command, sleeps briefly, returns "picked".
- These let the full LangGraph loop run end-to-end on one machine and exercise the real HTTP boundary the comm agents will use in production.

### 6. n8n workflows (`n8n/workflows/`)
- `order_intake.json` — exported workflow:
  1. Webhook from app → validate items → reserve stock in SQLite.
  2. HTTP call to `POST /orders/process` on the FastAPI brain.
  3. On success: commit reservation, notify app. On failure: rollback, notify app.
- Short `n8n/README.md` documenting how to import the workflow and where to point it (n8n is configured outside the repo, so a doc file is justified here).

### 7. Tests (`tests/`)
- `test_algorithms.py` — golden ordered routes for `nearest_neighbor`, `held_karp`, and `two_opt` on known distance matrices.
- `test_graph_routing.py` — Dijkstra correctness on the demo graph; verifies `build_distance_matrix` matches hand-computed values.
- `test_selector.py` — injects a stub `LLMClient` to confirm each enum value dispatches to the right algorithm and that bad LLM output falls back to `nearest_neighbor`.
- `test_triage.py` — injects a stub `LLMClient` that returns scripted tool sequences (happy path, single pick failure → recover, all picks failed → abort, exceeded tick budget → forced abort). Verifies dispatch and state transitions without hitting OpenRouter.
- `test_db.py` — reserve / commit / rollback semantics.
- `test_graph_e2e.py` — full LangGraph run against the mock movement + arm servers, with the stubbed `LLMClient` driving triage, asserting final state, picked items, and DB changes.

## Verification (once implemented)

- `pytest` — unit + e2e tests pass (all using stubbed `LLMClient`, no live OpenRouter calls).
- Live smoke test: start mock movement + mock arm + FastAPI brain + seeded DB; export `OPENROUTER_API_KEY`; POST a sample order to FastAPI directly. Watch the LangGraph traversal logs surface (a) the triage LLM's tool calls tick by tick, and (b) the strategy selector's chosen algorithm and reasoning.
- Inject a forced pick failure on one SKU (configurable in the mock arm server) to confirm the LLM-driven triage agent reasons about it and either recovers via `mark_item_failed` or aborts via `abort_order` — depending on how many items have failed so far.
- Inspect SQLite before/after to confirm the reserve → commit (or reserve → rollback) lifecycle for both happy and failed paths.
- Import the n8n workflow JSON into a local n8n instance and exercise the end-to-end path with `curl` against the n8n webhook.
