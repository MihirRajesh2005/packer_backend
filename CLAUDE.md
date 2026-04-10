## Project Outline
This is the AI-assisted back-end, or "brain" for a picker and packer robot. It will receive orders from an app front-end, plan out the fastest route the robot needs to take in the store to pick up said items, and once it's done, send confirmation to the app front-end and update a database.

## Planned Workflow
1. A customer selects a few items from the app.
2. The requested quantities, if available, are deducted from the database, but the changes are not yet committed.
3. A signal is sent to the main back-end, where a triage agent receives the order.
4. The order info contains SKU_ID and Position. The latter information is passed on to a route-planning agent that devises the fastest route to pick up all items.
5. Once an order of positions has been devised, the triage agent forwards the list to another agent that communicates with the application back-end for the robot's movement system (NOT BUILT HERE).
6. Upon reaching each position, the movement system stops and sends a signal back to the agent.
7. The agent then forwards the instruction back to the Triage Agent which then passes it on to an agent that talks to arm-control system back-end (NOT BUILT HERE).
8. Once the arm has picked up the item, it sends confirmation back to the agent, which passes the message on to the triage agent.
9. The triage agent then signals through agent for the movement system to proceed to the next position.
10. Once the list of requested items has been exhausted with all items picked, a confirmation is posted to the app front-end, and the changes made earlier to the database are committed.

## Intended tools to be used
1. Langchain - ai inference
2. Langgraph - agent orchestration
3. n8n - locally hosted, to orchestrate movement of information between the database, app, and back-end "brain"
4. sqlite - local .db file, for tech demo
5. FastAPI - HTTP surface for the brain (called by n8n, also usable directly for dev)
6. OpenRouter - LLM provider, accessed via `langchain-openai`'s `ChatOpenAI` with `base_url="https://openrouter.ai/api/v1"`. Lets the underlying model (Claude / GPT / Llama / etc.) be swapped via env var.

## Architectural Decisions

See `PLAN.md` for the full implementation plan. Key decisions:

- **Position model:** graph of nodes + edges in SQLite (`nodes`, `edges`, `products.node_id`). Demo store is one row of shelves with one SKU per node, but the schema scales without migration to multi-row stores, one-way aisles, junctions, and special locations (start, pack station, charger).
- **Route planner:** built on a Dijkstra-derived distance matrix. Three deterministic TSP algorithms are implemented — `nearest_neighbor`, `held_karp`, `two_opt` — covering the speed × quality trade-space.
- **LLM-driven vs deterministic split:**
  - **LLM-driven (two call sites, both via OpenRouter):**
    1. **Triage agent** — drives the orchestration loop by emitting structured Pydantic tool calls (`PlanRoute`, `MoveTo`, `Pick`, `MarkItemFailed`, `CompleteOrder`, `AbortOrder`). Reasons about error recovery, partial failures, and order completion. Has safety nets: max 50 ticks per order, forced abort on invalid tool output, temperature=0.
    2. **Route-planning strategy selector** — picks one of the three algorithms based on a structured summary of order shape (item count, bounding box, max/mean pairwise distance). Falls back to `nearest_neighbor` on any LLM error.
  - **Deterministic:** route algorithms themselves, movement-comm HTTP exchange, arm-comm HTTP exchange, and the entire database layer. Routing correctness and hardware communication never depend on an LLM.
- **External hardware mocks:** standalone FastAPI mock servers for the movement and arm-control systems, called over real HTTP by the comm agents — exercises the network boundary and lets the movement / arm system URLs be swapped to real hardware via env var.
- **Front door:** both n8n and FastAPI. FastAPI is the canonical brain backend; n8n is a thin wrapper that handles the reserve→commit DB choreography and forwards orders to FastAPI. Direct FastAPI calls are used for fast dev-loop iteration; n8n is the production-shaped path.
- **Demo seed dataset:** a single "interesting" preset — ~12 SKUs along a single row of shelves with non-uniform edge costs — chosen so that the strategy selector has a non-trivial choice to make on camera (nearest-neighbor and Held–Karp can produce different routes).