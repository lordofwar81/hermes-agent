# AIAgent.run_conversation() Refactoring Blueprint

## Overview
- **Current**: 3,001-line monolith method (lines 7876-10876)
- **Goal**: Reduce to ~400 lines via modular extraction
- **Status**: Complete blueprint ready for implementation

## Method Structure Analysis

### Section Breakdown

**S1: Pre-turn Initialization** (Lines 7904-8025, ~120 lines)
- Guard stdio installation  
- Runtime restoration from fallback
- User message sanitization  
- Task ID generation
- Retry counter reset
- Connection health check
- Memory nudge logic
- Message initialization and todo store hydration

**S2: System Prompt Construction** (Lines 8027-8208, ~180 lines)  
- Temporary system prompt handling
- Prompt caching and reuse logic
- Memory context integration
- Plugin context assembly
- External memory prefetch

**S3: Context Compression** (Lines 8210-8285, ~75 lines)
- Compression preflight checks
- Plugin hooks and memory prefetch
- Extension cache injection

**S4: Main Loop Setup** (Lines 8287-8315, ~28 lines)
- API message preparation pipeline
- System prompt injection
- Context injection logic

**S5: API Call Loop** (Lines 8316-8915, ~600 lines - THE MONSTER)
- Interrupt handling and budget management
- Step callback execution  
- Skill nudge tracking
- Message building and preparation
- `_interruptible_api_call()` execution
- Usage tracking and token accounting
- API error handling and retry logic

**S6: Response Processing** (Lines 8916-9025, ~110 lines)
- Response normalization
- Token usage recording
- Cache stats logging

**S7: Error Handling** (Lines 9026-9618, ~592 lines - COMPLEX HELL)
- API error classification and recovery
- Auth refresh handling
- Context length error detection
- Rate limit backoff
- Max retry exhaustion
- Payload-too-large handling
- Surrogate recovery
- Connection cleanup

**S8: Tool Call Processing** (Lines 9619-9905, ~286 lines)
- Tool call validation and repair
- JSON argument validation  
- Invalid tool retry logic
- Tool execution and error handling
- Post-tool compression check

**S9: Final Response Classification** (Lines 9906-10185, ~280 lines)
- Empty response detection
- Think-block processing
- Content-with-tools fallback
- Final message assembly
- Response quality filtering

**S10: Post-loop Cleanup** (Lines 10186-10876, ~690 lines)
- Max iterations handling
- Task resource cleanup (VM, browser)
- Session persistence
- Plugin hooks
- Memory sync and review
- Result assembly and return

## Proposed Module Extraction Plan

### Phase 1: Extract Error Handling (BIGGEST WIN - ~592 lines)

**New file:** `agent/api_retry_handler.py`
**Extracted from:** S7 (lines 9026-9618)

```python
class APIRetryHandler:
    def __init__(self, agent: 'AIAgent')
    
    def handle_interrupted(self, ...) -> dict | None
    def handle_api_error(self, api_error, ...) -> tuple[bool, dict | None]
        # Returns (should_retry, optional_result_dict)
    def _handle_surrogate_recovery(self, api_error, messages) -> bool
    def _handle_auth_refresh(self, status_code, ...) -> bool
    def _handle_context_length_error(self, error_msg, approx_tokens, ...) -> tuple[bool, dict | None]
    def _handle_payload_too_large(self, ...) -> tuple[bool, dict | None]
    def _handle_rate_limit(self, status_code, error_msg, ...) -> tuple[bool, dict | None]
    def _handle_client_error(self, status_code, error_msg, ...) -> dict | None
    def _handle_max_retries(self, api_error, ...) -> dict | None
    def _sleep_with_interrupt_check(self, wait_time, ...) -> dict | None
```

**Rationale:** This is the deepest-nested, most complex section. Every error type follows the same pattern: classify → attempt recovery → retry or return error dict. A handler class with clear methods per error type dramatically reduces cognitive load.

### Phase 2: Extract Response Processing (S6 + S9 - ~400 lines)

**New file:** `agent/response_processor.py`
**Extracted from:** S6 (8916-9025) + S9 (9906-10185)

```python
class ResponseProcessor:
    def __init__(self, agent: 'AIAgent')
    
    def normalize_response(self, response, api_mode) -> tuple[assistant_message, finish_reason]
    def normalize_content(self, content) -> str
    def record_usage(self, response) -> None
    def classify_final_response(self, assistant_message, messages, ...) -> tuple[str, bool, dict | None]
        # Returns (final_response, should_break, optional_error_result)
    def handle_empty_content(self, assistant_message, ...) -> tuple[str, bool, dict | None]
    def detect_codex_ack(self, ...) -> bool
```

**Rationale:** Response normalization, token accounting, and final classification are self-contained operations that only read agent state and messages, returning control flow signals.

### Phase 3: Extract Message Preparation (S1 + S2 + S3 - ~375 lines)

**New file:** `agent/message_builder.py`
**Extracted from:** S1 (7904-8025) + S2 (8027-8208) + S3 (8210-8285)

```python
class MessageBuilder:
    def __init__(self, agent: 'AIAgent')
    
    def initialize_turn(self, user_message, conversation_history, ...) -> TurnContext
    def build_api_messages(self, messages, current_turn_user_idx, ...) -> list[dict]
    def prepare_system_prompt(self, ...) -> str
    def inject_context(self, messages, current_turn_user_idx, ...) -> None
    def apply_prefills(self, api_messages, prefill_messages) -> list[dict]
    def _copy_and_inject_reasoning(self, messages, current_turn_user_idx, ext_prefetch_cache) -> list[dict]
    def _apply_system_prompt(self, api_messages, effective_system, plugin_turn_context) -> list[dict]
    def _apply_prefills(self, api_messages, prefill_messages) -> list[dict]
    def _apply_caching(self, api_messages) -> list[dict]
    def _sanitize_for_api(self, api_messages) -> list[dict]
```

**Rationale:** All message preparation is a pure transformation pipeline. Extracting it makes the main loop read like: `api_messages = self._msg_builder.build(...)`.

### Phase 4: Extract Tool Call Processing (S8 - ~286 lines)

**New file:** `agent/tool_processor.py`
**Extracted from:** S8 (9619-9905)

```python
class ToolProcessor:
    def __init__(self, agent: 'AIAgent')
    
    def process_tool_calls(self, assistant_message, messages, ...) -> tuple[list, dict | None]
        # Returns (tool_results, optional_error_result)
    def validate_and_repair(self, tool_calls) -> tuple[list, dict | None]
    def _repair_tool_names(self, tool_calls) -> list
    def _validate_json_args(self, tool_calls) -> tuple[list, list]
    def _build_invalid_tool_error(self, invalid_names, available) -> list[dict]
    def _build_invalid_json_recovery(self, invalid_args, tool_calls) -> list[dict]
```

**Rationale:** Tool validation and execution is a self-contained unit with clear error/retry logic that can return control signals instead of mutating state directly.

### Phase 5: Extract Turn Lifecycle (S4 + S10 - ~718 lines)

**Keep these as methods on `AIAgent` but rename and consolidate:**

```python
# On AIAgent:
def _init_turn(self, user_message, ...) -> TurnContext
    # Combines S1 + S2 + S3 + S4 into a dataclass

def _execute_api_call(self, ctx: TurnContext) -> tuple[dict, bool]
    # Combines S5 (main loop API call)

def _finalize_turn(self, ctx: TurnContext, result: dict) -> dict
    # Combines S10

@dataclass
class TurnContext:
    messages: list
    effective_task_id: str
    original_user_message: str
    current_turn_user_idx: int
    api_call_count: int
    final_response: str | None
    interrupted: bool
    _should_review_memory: bool
    iteration_budget: IterationBudget
    step_callback: Optional[callable]
    
    # Context components
    _ext_prefetch_cache: str
    _plugin_user_context: str
```

**Rationale:** The pre-turn setup, API call execution, and post-turn teardown are naturally paired. A `TurnContext` dataclass makes the loop variables explicit instead of `nonlocal`.

## Remaining in `run_conversation` After Extraction (~200 lines)

The main loop would be reduced to:

```python
def run_conversation(self, user_message, ...):
    # ~100 lines: Turn initialization
    ctx = self._msg_builder.initialize_turn(user_message, ...)
    
    # ~50 lines: Dependency injection  
    retry_handler = APIRetryHandler(self)
    response_proc = ResponseProcessor(self)
    tool_proc = ToolProcessor(self)
    
    # ~30 lines: Main loop
    while ctx.api_call_count < self.max_iterations and ctx.iteration_budget.remaining > 0:
        # ~10 lines: Interrupt/budget/skill checks
        
        # ~20 lines: Execute single API call with full error handling
        api_response, should_continue = self._execute_api_call(ctx, retry_handler, response_proc, tool_proc)
        
        if not should_continue:
            break
    
    return self._finalize_turn(ctx, ...)
```

This is approximately **200 lines** with clear delegation — a massive improvement from 3,001.

## Dependency Graph

```
run_conversation
  ├── TurnContext (dataclass)
  ├── MessageBuilder         [Phase 3 — new module]
  ├── APIRetryHandler        [Phase 1 — new module]  
  ├── ResponseProcessor      [Phase 2 — new module]
  ├── ToolProcessor          [Phase 4 — new module]
  ├── _init_turn()           [Phase 5 — on AIAgent]
  ├── _execute_api_call()    [Phase 5 — on AIAgent]
  └── _finalize_turn()       [Phase 5 — on AIAgent]
```

No circular dependencies. All new modules receive `agent: AIAgent` reference (read-only access to instance attrs). `TurnContext` is a plain dataclass.

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| **Missing context refs** | All new modules use getter methods, not direct attribute access |
| **Circular imports** | Place all new files in `agent/` subdirectory with clear dependency order |
| **Performance impact** | Method calls are negligible overhead vs. existing 3K-line complexity |
| **State synchronization** | All state changes remain on AIAgent instance; handlers use return values |
| **Tool compatibility** | Tool handlers unchanged; only call sites extracted to modules |
| **Plugin hooks** | Plugin callbacks remain in place; handlers respect existing hooks |
| **Memory sync** | Memory operations moved to `_finalize_turn()` and `_init_turn()` |
| **Iteration budget** | Budget tracking stays on AIAgent; handlers return break signals |

## Implementation Order

1. **Phase 1**: Extract `api_retry_handler.py` (unlocks biggest complexity reduction)
2. **Phase 5**: Create `TurnContext` dataclass and refactor _init_turn/_finalize_turn  
3. **Phase 2**: Extract `response_processor.py` (isolates response logic)
4. **Phase 4**: Extract `tool_processor.py` (isolates tool logic)
5. **Phase 3**: Extract `message_builder.py` (isolates message construction)
6. **Final cleanup**: Refactor remaining main loop into _execute_api_call()

## Expected Benefits

- **Readability**: Main loop becomes readable at ~200 lines
- **Testability**: Each module can be unit tested independently  
- **Maintainability**: Bug fixes targeted to specific modules
- **Performance**: No significant impact (method calls cheap)
- **Onboarding**: New developers understand each responsibility clearly
- **Debugging**: Stack traces meaningful to specific operations

## File Locations

- **Blueprint**: `/home/lordofwarai/.hermes/hermes-agent/REFACTORING_BLUEPRINT.md`
- **Target**: `/home/lordofwarai/.hermes/hermes-agent/run_agent.py` (lines 7876-10876)  
- **New modules**: `/home/lordofwarai/.hermes/hermes-agent/agent/`
  - `api_retry_handler.py`
  - `response_processor.py` 
  - `message_builder.py`
  - `tool_processor.py`

**Blueprint complete. Ready for implementation.**