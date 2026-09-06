"""
AD-BoN Runtime Gateway — FastAPI REST Server
=============================================
Exposes the AD-BoN Phase 4-v3 gateway (multi-tier routing, semantic cache,
prompt-cache layer, self-fixing critic-corrector loop) as an HTTP service.

Endpoints:
  POST /route                   — Route a query through the AD-BoN gateway
  POST /v1/chat/completions     — OpenAI-compatible chat endpoint
  GET  /health                  — Service health check
  GET  /cache/stats             — Semantic + prompt cache statistics
  GET  /benchmark               — Run the full Before/After benchmark suite

Start with:
  python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import time
import base64
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from phase4_runtime_gateway_v4 import (
    ADBoNGateway,
    CONFIDENCE_THRESHOLD,
    CACHE_THRESHOLD,
    extract_file_metadata,
    create_sample_assets,
)
from ai_assistant import AIAssistant

# ---------------------------------------------------------------------------
# App + single shared gateway instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AD-BoN Runtime Gateway API",
    description=(
        "Adaptive Dynamic Best-of-N LLM routing gateway with semantic caching, "
        "prompt-cache layer, self-fixing critic-corrector loop, and quality measurement."
    ),
    version="4.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cache_candidates = [
    os.path.join("data", "cache", "semantic_cache.json"),
    "semantic_cache.json",
]
cache_file = next((p for p in cache_candidates if os.path.exists(p)), os.path.join("data", "cache", "semantic_cache.json"))

_gateway = ADBoNGateway(
    use_mock_router=True,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    cache_threshold=CACHE_THRESHOLD,
    cache_file=cache_file,
    use_live_llm=True,
)

_assistant = AIAssistant(gateway=_gateway)

# Ensure sample files exist for quick testing
create_sample_assets()

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class RouteRequest(BaseModel):
    query: str = Field(..., description="The user query to route and answer.")
    target_quality: float = Field(0.76, ge=0.01, le=1.0, description="Minimum acceptable quality score (0-1).")
    system_prompt: Optional[str] = Field(None, description="Optional system instruction block (enables prompt caching).")
    context_block: Optional[str] = Field(None, description="Optional context / document block (enables prompt caching).")
    bypass_cache: bool = Field(False, description="If true, skip semantic cache lookup for this request.")
    file_path: Optional[str] = Field(None, description="Optional path to local multimodal file.")
    file_name: Optional[str] = Field(None, description="Optional name of uploaded file.")
    file_base64: Optional[str] = Field(None, description="Optional base64 payload of uploaded file.")


class AssistantChatRequest(BaseModel):
    message: str = Field(..., description="The user message for the AI Assistant.")
    conversation: Optional[List[Dict[str, Any]]] = Field(None, description="Recent conversation history.")
    target_quality: Optional[float] = Field(0.76, description="Minimum acceptable quality threshold.")


class RouteResponse(BaseModel):
    query: str
    final_response: str

    # Routing decision
    selected_model: str
    selected_budget_n: int
    routing_tier: str       # "junior" | "medium" | "senior" | "cache"
    primary_cluster: str

    # Cost metrics
    gateway_cost_usd: float
    senior_baseline_cost_usd: float
    cost_savings_pct: float

    # Quality metrics
    final_quality_score: float
    expected_quality: float
    baseline_quality_score: float
    quality_delta: float
    quality_retained_pct: float
    confidence_threshold: float
    self_fix_triggered: bool
    self_fix_status: str
    fallback_to_senior: bool

    # Cache signals
    is_semantic_cache_hit: bool
    is_prompt_cache_hit: bool
    cache_similarity: float
    retrieval_latency_ms: float

    # Multimodal additions
    file_metadata: Optional[Dict[str, Any]] = None
    multimodal_bypass_triggered: bool = False
    bypass_reason: Optional[str] = None
    multimodal_surcharge: float = 0.0


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "adbon-gateway"
    messages: List[ChatMessage]
    temperature: float = 0.7
    target_quality: float = 0.76
    system_prompt: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list
    usage: dict
    adbon_metadata: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_SENIOR_BASELINE_QUALITY = 0.95  # Conservative fixed baseline for Senior-only approach


def _build_route_response(query: str, report: dict) -> RouteResponse:
    """Translate gateway report dict → RouteResponse."""
    gw_quality = report["final_score"]
    quality_delta = gw_quality - _SENIOR_BASELINE_QUALITY
    quality_retained = (gw_quality / _SENIOR_BASELINE_QUALITY) * 100.0

    model_name_map = {
        "junior": "Ollama (phi3 / gemma3)",
        "medium": "Groq (compound / qwen)",
        "senior": "Google Gemini 1.5 Pro",
        "cache": "Semantic Memory (Instant)",
    }
    if report.get("is_cache_hit"):
        raw_cached = report.get("cached_model", "cache")
        selected_model = model_name_map.get(raw_cached, raw_cached)
        selected_budget = 0
        routing_tier = "cache"
        cluster = "cached"
        exp_quality = gw_quality
    else:
        selected_route = report.get("selected_route", {})
        raw_m = selected_route.get("name", selected_route.get("model", "unknown"))
        selected_model = model_name_map.get(selected_route.get("model"), raw_m)
        selected_budget = selected_route.get("budget", 1)
        routing_tier = report.get("delivering_tier", selected_route.get("tier", selected_route.get("model", "unknown")))
        cluster = report.get("primary_cluster_name", "unknown")
        exp_quality = selected_route.get("expected_quality", gw_quality)

    return RouteResponse(
        query=query,
        final_response=report["final_response"],
        selected_model=selected_model,
        selected_budget_n=selected_budget,
        routing_tier=routing_tier,
        primary_cluster=cluster,
        gateway_cost_usd=report["total_query_cost"],
        senior_baseline_cost_usd=report["senior_cost_baseline"],
        cost_savings_pct=report["cost_saving_percentage"],
        final_quality_score=gw_quality,
        expected_quality=exp_quality,
        baseline_quality_score=_SENIOR_BASELINE_QUALITY,
        quality_delta=round(quality_delta, 4),
        quality_retained_pct=round(quality_retained, 2),
        confidence_threshold=report.get("confidence_threshold", CONFIDENCE_THRESHOLD),
        self_fix_triggered=report.get("self_fix_triggered", False),
        self_fix_status=report.get("self_fix_status", "N/A"),
        fallback_to_senior=report.get("fallback_to_senior", False),
        is_semantic_cache_hit=report.get("is_cache_hit", False),
        is_prompt_cache_hit=report.get("is_prompt_cache_hit", False),
        cache_similarity=report.get("cache_similarity", 0.0),
        retrieval_latency_ms=report.get("latency_ms", report.get("retrieval_latency_ms", 0.0)),
        file_metadata=report.get("file_metadata"),
        multimodal_bypass_triggered=report.get("multimodal_bypass_triggered", False),
        bypass_reason=report.get("bypass_reason"),
        multimodal_surcharge=report.get("multimodal_surcharge", 0.0),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health():
    """Service liveness probe."""
    return {
        "status": "ok",
        "service": "AD-BoN Runtime Gateway",
        "version": "4.4.0",
        "semantic_cache_entries": _gateway.semantic_cache.size(),
        "confidence_threshold": _gateway.confidence_threshold,
        "cache_threshold": _gateway.cache_threshold,
        "multimodal_support": True,
        "supported_modalities": ["image", "audio", "video", "document"],
    }


@app.get("/cache/stats", tags=["Cache"])
def cache_stats():
    """Return live statistics for both semantic cache and prompt-cache layers."""
    pc = _gateway.prompt_cache.get_stats()
    return {
        "semantic_cache": {
            "total_entries": _gateway.semantic_cache.size(),
            "similarity_threshold": _gateway.cache_threshold,
        },
        "prompt_cache": pc,
    }


# ---------------------------------------------------------------------------
# AI Assistant Chat Endpoint
# ---------------------------------------------------------------------------
@app.post("/assistant/chat", tags=["Assistant"])
@app.post("/api/assistant/chat", tags=["Assistant"])
def assistant_chat(
    req: AssistantChatRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Execute an AI Assistant chat turn driven by AD-BoN gating and Groq generation.
    Protected by Bearer token API key validation.
    """
    import hmac
    expected_key = os.getenv("ADBON_API_KEY", "adbon-sec-key-2026-demo")
    
    # Check Bearer auth if authorization header is provided or enforced
    if authorization:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Valid API credentials are required.")
        token = authorization.split("Bearer ", 1)[1].strip()
        if not hmac.compare_digest(token, expected_key):
            raise HTTPException(status_code=401, detail="Invalid API credentials.")
    
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Field 'message' is required and cannot be empty.")
        
    try:
        return _assistant.chat(
            message=req.message.strip(),
            conversation=req.conversation,
            target_quality=req.target_quality,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        err_text = str(e)
        if "api key" in err_text.lower() or "groq_api_key" in err_text.lower():
            safe_msg = "GROQ_API_KEY is not configured on the backend."
        else:
            safe_msg = err_text
        raise HTTPException(status_code=500, detail=safe_msg)


MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB Limit

@app.post("/route", response_model=RouteResponse, tags=["Gateway"])
def route_query(request: RouteRequest):
    """
    Route a query through the AD-BoN multimodal gateway.
    Handles raw text, attached images, audio, video, and documents (up to 10 MB).
    """
    resolved_file_path = None

    # Handle client-uploaded base64 file
    if request.file_name and request.file_base64:
        os.makedirs("uploads", exist_ok=True)
        resolved_file_path = os.path.join("uploads", request.file_name)
        try:
            # Strip data URI header if present
            b64_str = request.file_base64
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            raw_bytes = base64.b64decode(b64_str)
            if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
                size_mb = len(raw_bytes) / (1024 * 1024)
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file '{request.file_name}' ({size_mb:.1f} MB) exceeds the 10 MB size limit."
                )
            with open(resolved_file_path, "wb") as f:
                f.write(raw_bytes)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode uploaded file: {e}")
    elif request.file_path:
        resolved_file_path = request.file_path
        if not os.path.exists(resolved_file_path):
            for candidate in [
                os.path.join("assets", "samples", request.file_path),
                os.path.join("assets", "samples", os.path.basename(request.file_path)),
            ]:
                if os.path.exists(candidate):
                    resolved_file_path = candidate
                    break
        if os.path.exists(resolved_file_path) and os.path.getsize(resolved_file_path) > MAX_FILE_SIZE_BYTES:
            size_mb = os.path.getsize(resolved_file_path) / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"File '{request.file_path}' ({size_mb:.1f} MB) exceeds the 10 MB size limit."
            )

    try:
        report = _gateway.route_query(
            query=request.query,
            target_quality=request.target_quality,
            bypass_cache=request.bypass_cache,
            system_prompt=request.system_prompt,
            context_block=request.context_block,
            file_path=resolved_file_path,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return _build_route_response(request.query, report)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["OpenAI-Compatible"])
def openai_chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Extracts the last user message and routes it through the AD-BoN gateway.
    """
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in messages list.")

    query = user_messages[-1].content
    system_msgs = [m.content for m in request.messages if m.role == "system"]
    system_prompt = system_msgs[-1] if system_msgs else request.system_prompt

    try:
        report = _gateway.route_query(
            query=query,
            target_quality=request.target_quality,
            system_prompt=system_prompt,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    gw_q = report["final_score"]
    q_retained = round((gw_q / _SENIOR_BASELINE_QUALITY) * 100.0, 2)

    completion_id = f"adbon-{int(time.time())}"
    return ChatCompletionResponse(
        id=completion_id,
        model=request.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": report["final_response"]},
                "finish_reason": "stop",
            }
        ],
        usage={
            "prompt_tokens": 0,   # could wire up _estimate_input_tokens if needed
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        adbon_metadata={
            "routing_tier": report.get("selected_route", {}).get("model", "cache"),
            "quality_score": gw_q,
            "quality_retained_pct": q_retained,
            "quality_delta": round(gw_q - _SENIOR_BASELINE_QUALITY, 4),
            "cost_usd": report["total_query_cost"],
            "cost_savings_pct": report["cost_saving_percentage"],
            "is_semantic_cache_hit": report.get("is_cache_hit", False),
            "is_prompt_cache_hit": report.get("is_prompt_cache_hit", False),
            "self_fix_triggered": report.get("self_fix_triggered", False),
            "self_fix_status": report.get("self_fix_status", "N/A"),
            "fallback_to_senior": report.get("fallback_to_senior", False),
        },
    )


@app.get("/benchmark", tags=["Evaluation"])
def run_benchmark():
    """
    Run the full Before/After benchmark suite and return structured results
    including per-query quality retention metrics for judge evaluation.
    """
    from benchmark_dashboard import run_benchmark_suite
    data = run_benchmark_suite(benchmark_cache_file="benchmark_api_cache.json")

    # Enrich each result with quality metrics
    baseline_q = _SENIOR_BASELINE_QUALITY
    for r in data["results"]:
        gw_q = r["quality_score"]
        r["baseline_quality_score"] = baseline_q
        r["quality_delta"] = round(gw_q - baseline_q, 4)
        r["quality_retained_pct"] = round((gw_q / baseline_q) * 100.0, 2)

    # Aggregate quality stats
    scores = [r["quality_score"] for r in data["results"]]
    data["kpis"]["avg_gateway_quality"] = round(sum(scores) / len(scores), 4)
    data["kpis"]["avg_baseline_quality"] = baseline_q
    data["kpis"]["avg_quality_retained_pct"] = round(
        sum(r["quality_retained_pct"] for r in data["results"]) / len(data["results"]), 2
    )
    data["kpis"]["quality_degraded_queries"] = sum(
        1 for r in data["results"] if r["quality_delta"] < -0.05
    )

    return data


if __name__ == "__main__":
    import uvicorn
    print("Starting AD-BoN Runtime Gateway API on http://127.0.0.1:8000 ...")
    print("  Interactive docs  →  http://127.0.0.1:8000/docs")
    print("  Health check      →  http://127.0.0.1:8000/health")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
