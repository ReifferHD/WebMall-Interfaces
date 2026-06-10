import os
import sys

# Fix Windows console encoding: replace unencodable chars instead of crashing
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# Import calculation function from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import calculation_results, interface_results_dir, extract_reasoning_effort
from rag.elasticsearch_client import ElasticsearchRAGClient
import json
import time
import re
from typing import Dict, List, Any, Tuple, Optional
import asyncio
from datetime import datetime
from dotenv import load_dotenv
import csv


# LangChain imports for model-agnostic approach
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI  # MODIFIED: Gemini runs
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.callbacks import get_usage_metadata_callback
from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError

# Import cart tools
from rag.rag_cart_tools import get_cart_tools, reset_all_carts



load_dotenv()


def create_chat_model(model_name: str, *, reasoning_effort: Optional[str] = None,
                      temperature: float = 0.0) -> Any:
    """Create the matching LangChain chat model for benchmark runners."""
    model_key = (model_name or "").lower()
    if model_key.startswith("gemini") or model_key.startswith("models/gemini"):
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    if model_key.startswith("claude"):
        return ChatAnthropic(model=model_name, temperature=temperature)
    if model_key.startswith("mistral"):
        return ChatMistralAI(model=model_name, temperature=temperature)

    kwargs: Dict[str, Any] = {"model": model_name}
    if model_key.startswith("gpt-5") and reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    elif temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


# Configuration: webmall URLs
URLS = {
    "URL_1": "https://webmall-1.informatik.uni-mannheim.de",
    "URL_2": "https://webmall-2.informatik.uni-mannheim.de",
    "URL_3": "https://webmall-3.informatik.uni-mannheim.de",
    "URL_4": "https://webmall-4.informatik.uni-mannheim.de",
    "URL_5": "https://webmall-solution.informatik.uni-mannheim.de"
}

# Parameter to choose used method
OPTIMIZATION_METHOD = os.getenv("OPTIMIZATION_METHOD", "none")

# Initialize Elasticsearch client for RAG
es_client = ElasticsearchRAGClient()

# Initialize embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def normalize_url(url: str) -> str:
    """Normalize URL for comparison by removing trailing slashes and converting to lowercase."""
    return url.rstrip('/').lower()


def fill_urls(text: str, urls: Dict[str, str]) -> str:
    """Replace URL placeholders with actual URLs."""
    for key, val in urls.items():
        text = text.replace("{{" + key + "}}", val)
    return text


async def get_embedding(text: str) -> Tuple[List[float], int]:
    """Get embedding vector from OpenAI and return tokens used."""
    try:
        # Use LangChain embeddings
        embedding = await embeddings_model.aembed_query(text)
        # Estimate tokens (roughly 1 token per 4 characters)
        tokens_used = len(text) // 4
        return embedding, tokens_used
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536, 0  # Return zero vector and zero tokens on error


# Global variables for tracking
search_history = []
details_history = []
search_results_cache = []  # Store actual results for easy access
tool_call_sequence = 0
token_tracker = {"embedding_tokens": 0}

# MODIFIED: per-task error-analysis logs (reset in get_model_answer)
filter_decisions_log = []       # list of {query, pre_filter_urls, post_filter_urls, removed_urls}
filter_llm_calls_log = []       # per-candidate gpt-5-nano filtering decisions
masking_events_log = []         # list of structured masking events for error analysis
_masked_message_ids_logged = set()  # dedup: only log first time a msg gets masked
current_expected_urls_for_masking = set()  # normalized expected URLs for event-level masking attribution
cache_keyword_used = None       # keyword used for plan-cache lookup this task
cache_template_applied = None   # template string injected, if cache hit
cache_template_metadata = None  # selected cache entry metadata for error analysis
cache_llm_calls_log = []        # gpt-4o-mini cache classifier/template calls
current_agent_model_used = None # actual agent model for this task


def _norm_url_for_log(url: str) -> str:
    return (url or "").rstrip("/").lower()


def _tokenize_query_for_log(text: str) -> set:
    return {tok for tok in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(tok) > 1}


def _query_similarity_for_log(a: str, b: str) -> float:
    ta = _tokenize_query_for_log(a)
    tb = _tokenize_query_for_log(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _next_tool_call_sequence() -> int:
    global tool_call_sequence
    tool_call_sequence += 1
    return tool_call_sequence


def _urls_visible_after_mask_event(event: Dict[str, Any]) -> set:
    """URLs that became visible again after a specific masked observation."""
    event_sequence = int(event.get("tool_call_sequence") or 0)
    visible_after = set()

    if event_sequence:
        for search_record in search_history:
            if int(search_record.get("tool_call_sequence") or 0) <= event_sequence:
                continue
            visible_after |= {
                _norm_url_for_log(u)
                for u in search_record.get("result_urls", [])
                if u
            }
        for detail_record in details_history:
            if int(detail_record.get("tool_call_sequence") or 0) <= event_sequence:
                continue
            visible_after |= {
                _norm_url_for_log(u)
                for u in (
                    detail_record.get("requested_urls", [])
                    + detail_record.get("result_urls", [])
                )
                if u
            }
        return visible_after

    # Backward-compatible fallback for logs created before global sequencing.
    if event.get("tool_type") == "search":
        event_index = int(event.get("search_call_index") or 0)
        for search_record in search_history:
            later_index = int(search_record.get("call_index") or 0)
            if later_index and event_index and later_index <= event_index:
                continue
            visible_after |= {
                _norm_url_for_log(u)
                for u in search_record.get("result_urls", [])
                if u
            }
    elif event.get("tool_type") == "details":
        event_index = int(event.get("details_call_index") or 0)
        for detail_record in details_history:
            later_index = int(detail_record.get("call_index") or 0)
            if later_index and event_index and later_index <= event_index:
                continue
            visible_after |= {
                _norm_url_for_log(u)
                for u in (
                    detail_record.get("requested_urls", [])
                    + detail_record.get("result_urls", [])
                )
                if u
            }
    return visible_after


def _extract_template_steps_for_log(template: str) -> List[str]:
    steps: List[str] = []
    in_steps = False
    for line in (template or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("STEPS"):
            in_steps = True
            continue
        if in_steps and re.match(r"^[A-Z][A-Z _-]+:", stripped):
            break
        if in_steps:
            match = re.match(r"^\d+\.\s*(.+)$", stripped)
            if match:
                steps.append(match.group(1).strip())
    return steps


def _build_cache_summary(expected_flat: List[str],
                         parsed_urls: Optional[List[str]] = None,
                         error_type: Optional[str] = None) -> Dict[str, Any]:
    """Derived signals for cache-hit failure attribution."""
    if not last_cache_hit:
        call_types: Dict[str, int] = {}
        for call in cache_llm_calls_log:
            ct = call.get("call_type", "unknown")
            call_types[ct] = call_types.get(ct, 0) + 1
        return {
            "cache_hit": False,
            "cache_keyword_used": cache_keyword_used,
            "normal_agent_skipped": False,
            "cache_helper_model": CACHE_MODEL if OPTIMIZATION_METHOD == "caching" else None,
            "cache_hit_model": None,
            "agent_model_used": current_agent_model_used,
            "cache_llm_call_count": len(cache_llm_calls_log),
            "cache_llm_call_types": call_types,
            "cache_match_policy": CACHE_MATCH_POLICY if OPTIMIZATION_METHOD == "caching" else None,
            "error_class": None,
            "causal_flags": [],
        }

    expected_urls = {_norm_url_for_log(u) for u in (expected_flat or []) if u}
    parsed = {_norm_url_for_log(u) for u in (parsed_urls or []) if u}

    search_urls_seen = {
        _norm_url_for_log(u)
        for record in search_history
        for u in record.get("result_urls", [])
        if u
    }
    detail_urls_seen = {
        _norm_url_for_log(u)
        for record in details_history
        for u in (record.get("requested_urls", []) + record.get("result_urls", []))
        if u
    }
    tool_urls_seen = search_urls_seen | detail_urls_seen

    fns = expected_urls - parsed
    fps = parsed - expected_urls
    expected_visible = expected_urls & tool_urls_seen
    expected_missing_from_tools = expected_urls - tool_urls_seen
    expected_seen_not_returned = fns & tool_urls_seen
    fp_surfaced_by_tools = fps & tool_urls_seen
    fp_not_surfaced_by_tools = fps - tool_urls_seen

    template_steps = _extract_template_steps_for_log(cache_template_applied or "")
    error_class = None
    if last_cache_hit and (error_type or fns or fps):
        if fps and not fns and not error_type:
            error_class = "CacheTemplateOvergeneralizationError"
        else:
            error_class = "CacheTemplateMisapplicationError"

    causal_flags = []
    if last_cache_hit and (fns or fps or error_type):
        causal_flags.append("cache_hit_failure")
    if fps and not fns:
        causal_flags.append("cache_fp_only")
    if fns:
        causal_flags.append("cache_fn_present")
    if expected_missing_from_tools:
        causal_flags.append("cache_search_missed_expected_urls")
    if expected_seen_not_returned:
        causal_flags.append("cache_expected_url_seen_not_returned")
    if fp_surfaced_by_tools:
        causal_flags.append("cache_selected_extra_surfaced_urls")
    if fp_not_surfaced_by_tools:
        causal_flags.append("cache_selected_unsurfaced_urls")
    if error_type:
        causal_flags.append("cache_hit_execution_error")

    metadata = cache_template_metadata or {}
    call_types: Dict[str, int] = {}
    for call in cache_llm_calls_log:
        ct = call.get("call_type", "unknown")
        call_types[ct] = call_types.get(ct, 0) + 1
    return {
        "cache_hit": bool(last_cache_hit),
        "cache_keyword_used": cache_keyword_used,
        "normal_agent_skipped": bool(last_cache_hit),
        "cache_helper_model": CACHE_MODEL if OPTIMIZATION_METHOD == "caching" else None,
        "cache_hit_model": CACHE_HIT_MODEL if last_cache_hit else None,
        "agent_model_used": current_agent_model_used,
        "cache_llm_call_count": len(cache_llm_calls_log),
        "cache_llm_call_types": call_types,
        "cache_match_policy": metadata.get("cache_match_policy"),
        "cache_gate_key": metadata.get("cache_gate_key"),
        "cached_gate_key": metadata.get("cached_gate_key"),
        "cache_entry_quality_score": metadata.get("quality_score"),
        "cache_entry_support_count": metadata.get("support_count"),
        "cache_entry_usage_count_before": metadata.get("usage_count_before"),
        "cache_entry_source_task_excerpt": metadata.get("source_task_excerpt"),
        "current_signature": metadata.get("current_signature"),
        "cached_signature": metadata.get("cached_signature"),
        "template_steps": template_steps,
        "template_step_count": len(template_steps),
        "template_search_step_count": sum(1 for step in template_steps if "search_products" in step),
        "template_detail_step_count": sum(1 for step in template_steps if "get_product_details" in step),
        "actual_search_queries": [record.get("query", "") for record in search_history],
        "actual_search_count": len(search_history),
        "actual_detail_count": len(details_history),
        "expected_urls_count": len(expected_urls),
        "expected_visible_urls": sorted(expected_visible),
        "expected_visible_urls_count": len(expected_visible),
        "expected_missing_from_tools": sorted(expected_missing_from_tools),
        "expected_missing_from_tools_count": len(expected_missing_from_tools),
        "expected_seen_not_returned": sorted(expected_seen_not_returned),
        "expected_seen_not_returned_count": len(expected_seen_not_returned),
        "fp_urls": sorted(fps),
        "fp_count": len(fps),
        "fp_surfaced_by_tools": sorted(fp_surfaced_by_tools),
        "fp_surfaced_by_tools_count": len(fp_surfaced_by_tools),
        "fp_not_surfaced_by_tools": sorted(fp_not_surfaced_by_tools),
        "fp_not_surfaced_by_tools_count": len(fp_not_surfaced_by_tools),
        "fn_urls": sorted(fns),
        "fn_count": len(fns),
        "error_class": error_class,
        "causal_flags": causal_flags,
    }


def _masking_research_loop_signal(masking_events: List[Dict[str, Any]],
                                  search_records: List[Dict[str, Any]]) -> bool:
    for event in masking_events:
        if event.get("tool_type") != "search":
            continue
        event_query = event.get("search_query", "")
        event_urls = {_norm_url_for_log(u) for u in event.get("masked_urls", []) if u}
        event_index = int(event.get("search_call_index") or 0)
        for later in search_records:
            later_index = int(later.get("call_index") or 0)
            if later_index and event_index and later_index <= event_index:
                continue
            later_urls = {_norm_url_for_log(u) for u in later.get("result_urls", []) if u}
            same_query = _query_similarity_for_log(event_query, later.get("query", "")) >= 0.6
            repeated_results = bool(event_urls and later_urls and (event_urls & later_urls))
            if same_query or repeated_results:
                return True
    return False


def _masking_detail_refetch_signal(masking_events: List[Dict[str, Any]],
                                   detail_records: List[Dict[str, Any]]) -> bool:
    for event in masking_events:
        if event.get("tool_type") != "details":
            continue
        event_urls = {_norm_url_for_log(u) for u in event.get("masked_urls", []) if u}
        event_index = int(event.get("details_call_index") or 0)
        if not event_urls:
            continue
        for later in detail_records:
            later_index = int(later.get("call_index") or 0)
            if later_index and event_index and later_index <= event_index:
                continue
            later_urls = {
                _norm_url_for_log(u)
                for u in (later.get("requested_urls", []) + later.get("result_urls", []))
                if u
            }
            if event_urls & later_urls:
                return True
    return False


def _build_masking_summary(expected_flat: List[str],
                           parsed_urls: Optional[List[str]] = None,
                           error_type: Optional[str] = None) -> Dict[str, Any]:
    """Derived causal signals for masking attribution.

    These fields make masking errors auditable instead of relying only on the
    optimization mode or a generic iteration-limit exception.
    """
    events = list(masking_events_log)
    masked_urls = {
        _norm_url_for_log(u)
        for event in events
        for u in event.get("masked_urls", [])
        if u
    }
    expected_urls = {_norm_url_for_log(u) for u in (expected_flat or []) if u}
    parsed = {_norm_url_for_log(u) for u in (parsed_urls or []) if u}
    masked_expected_urls = sorted(masked_urls & expected_urls)
    masked_missing_expected_urls = sorted((masked_urls & expected_urls) - parsed)
    masked_unrecovered_expected_urls = set()
    masked_recovered_later_expected_urls = set()
    masked_expected_evidence_chain = []
    for event in events:
        event_masked_urls = {
            _norm_url_for_log(u)
            for u in event.get("masked_urls", [])
            if u
        }
        event_masked_expected = event_masked_urls & expected_urls
        later_visible_urls = _urls_visible_after_mask_event(event)
        event_missing_final = event_masked_expected - parsed
        event_recovered_later = event_missing_final & later_visible_urls
        event_unrecovered = event_missing_final - later_visible_urls
        for url in event_masked_expected:
            if url in parsed:
                continue
            if url in later_visible_urls:
                masked_recovered_later_expected_urls.add(url)
            else:
                masked_unrecovered_expected_urls.add(url)
        if event_masked_expected:
            masked_expected_evidence_chain.append({
                "message_index": event.get("message_index"),
                "tool_type": event.get("tool_type", "unknown"),
                "search_query": event.get("search_query", ""),
                "tool_call_sequence": event.get("tool_call_sequence"),
                "masked_expected_urls": sorted(event_masked_expected),
                "missing_from_final_expected_urls": sorted(event_missing_final),
                "recovered_later_expected_urls": sorted(event_recovered_later),
                "unrecovered_expected_urls": sorted(event_unrecovered),
            })
    masked_recovered_later_expected_urls -= masked_unrecovered_expected_urls

    re_search_after_mask = _masking_research_loop_signal(events, search_history)
    detail_refetch_after_mask = _masking_detail_refetch_signal(events, details_history)
    graph_recursion_after_masking = bool(events) and error_type == "GraphRecursionError"
    masked_expected_evidence_unavailable = bool(masked_unrecovered_expected_urls)
    masked_evidence_before_loop = bool(events) and (
        re_search_after_mask or detail_refetch_after_mask or graph_recursion_after_masking
    )

    causal_flags = []
    if masked_missing_expected_urls:
        causal_flags.append("masked_expected_url_missing_final")
    if masked_unrecovered_expected_urls:
        causal_flags.append("masked_expected_url_unrecovered")
    if masked_expected_evidence_unavailable:
        causal_flags.append("masked_expected_evidence_unavailable")
    if masked_evidence_before_loop:
        causal_flags.append("masked_evidence_before_loop")
    if re_search_after_mask:
        causal_flags.append("re_search_after_mask")
    if detail_refetch_after_mask:
        causal_flags.append("detail_refetch_after_mask")
    if graph_recursion_after_masking:
        causal_flags.append("graph_recursion_after_masking")

    error_class = None
    error_class_basis = None
    if masked_expected_evidence_unavailable:
        error_class = "MaskedObservation-LostUrlError"
        error_class_basis = "expected_url_masked_and_not_recovered"

    return {
        "has_masking_events": bool(events),
        "total_masking_events": len(events),
        "masked_search_outputs": sum(1 for e in events if e.get("tool_type") == "search"),
        "masked_detail_outputs": sum(1 for e in events if e.get("tool_type") == "details"),
        "masked_urls_count": len(masked_urls),
        "masked_expected_urls": masked_expected_urls,
        "masked_expected_urls_count": len(masked_expected_urls),
        "masked_missing_expected_urls": masked_missing_expected_urls,
        "masked_missing_expected_urls_count": len(masked_missing_expected_urls),
        "masked_unrecovered_expected_urls": sorted(masked_unrecovered_expected_urls),
        "masked_unrecovered_expected_urls_count": len(masked_unrecovered_expected_urls),
        "masked_recovered_later_expected_urls": sorted(masked_recovered_later_expected_urls),
        "masked_recovered_later_expected_urls_count": len(masked_recovered_later_expected_urls),
        "masked_expected_evidence_chain": masked_expected_evidence_chain,
        "masked_expected_evidence_unavailable": masked_expected_evidence_unavailable,
        "masked_evidence_before_loop": masked_evidence_before_loop,
        "re_search_after_mask": re_search_after_mask,
        "detail_refetch_after_mask": detail_refetch_after_mask,
        "graph_recursion_after_masking": graph_recursion_after_masking,
        "error_class": error_class,
        "error_class_basis": error_class_basis,
        "causal_flags": causal_flags,
    }


def _build_partial_tool_calls_log() -> List[Dict[str, Any]]:
    """Tool log available even when the agent aborts before final messages."""
    partial_log: List[Dict[str, Any]] = []
    for search_record in search_history:
        partial_log.append({
            "tool_name": "search_products",
            "tool_args": {
                "call_index": search_record.get("call_index"),
                "tool_call_sequence": search_record.get("tool_call_sequence"),
                "query": search_record.get("query"),
                "match_count": search_record.get("match_count"),
                "use_hybrid": search_record.get("use_hybrid"),
            },
            "tool_output": {
                "results_found": search_record.get("results_found"),
                "status": "success",
                "result_urls": search_record.get("result_urls", []),
            },
            "timestamp": search_record.get("timestamp"),
            "tool_type": "search",
        })
    for detail_record in details_history:
        partial_log.append({
            "tool_name": "get_product_details",
            "tool_type": "details",
            "tool_args": {
                "call_index": detail_record.get("call_index"),
                "tool_call_sequence": detail_record.get("tool_call_sequence"),
                "urls": detail_record.get("requested_urls", []),
            },
            "tool_output": {
                "status": "success",
                "result_urls": detail_record.get("result_urls", []),
            },
            "timestamp": detail_record.get("timestamp"),
        })
    return partial_log

# MODIFIED: Optimization Method 1 — Pre-filtering (RankRAG-inspired,
# https://arxiv.org/abs/2407.02485). A small LLM scores each search
# candidate as relevant True/False (parallel calls); the agent only sees
# the kept results. Unlike RankRAG we use a prompted off-the-shelf model,
# no fine-tuning and no downstream re-ranker (the main agent re-ranks).
_filter_llm_client = None
FILTER_MODEL = os.getenv("FILTER_MODEL", "gpt-5-nano")
FILTER_CONCURRENCY = int(os.getenv("FILTER_CONCURRENCY", "10"))
filter_token_tracker = {"prompt_tokens": 0, "completion_tokens": 0}


async def _score_passage(query: str, passage: str, candidate: Dict[str, Any],
                         candidate_rank: int, semaphore) -> Dict[str, Any]:
    """Per-passage relevance call following the RankRAG prompt template."""
    global _filter_llm_client, filter_token_tracker
    async with semaphore:
        started = time.time()
        prompt = (
            f"For the question \"{query}\", assess whether the "
            f"passage is relevant to the question. {passage}\n"
            "Answer with only True or False."
        )
        log_entry = {
            "model": FILTER_MODEL,
            "call_type": "filter_relevance_decision",
            "prompt_template": "rankrag_true_false_v1",
            "query": query,
            "candidate_rank": candidate_rank,
            "candidate_url": candidate.get("url", ""),
            "candidate_title": (candidate.get("title") or "")[:220],
            "candidate_summary_preview": (candidate.get("summary") or "")[:360],
            "passage_preview": passage[:520],
            "prompt_chars": len(prompt),
            "decision": True,
            "answer_raw": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": None,
            "error": None,
        }
        try:
            resp = await _filter_llm_client.chat.completions.create(
                model=FILTER_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt,
                }],
                max_completion_tokens=2000,
                reasoning_effort="minimal",
            )
            if resp.usage:
                filter_token_tracker["prompt_tokens"] += resp.usage.prompt_tokens
                filter_token_tracker["completion_tokens"] += resp.usage.completion_tokens
                log_entry["prompt_tokens"] = resp.usage.prompt_tokens
                log_entry["completion_tokens"] = resp.usage.completion_tokens
                log_entry["total_tokens"] = getattr(
                    resp.usage,
                    "total_tokens",
                    resp.usage.prompt_tokens + resp.usage.completion_tokens,
                )
            ans_raw = resp.choices[0].message.content or ""
            ans = ans_raw.lower()
            log_entry["answer_raw"] = ans_raw.strip()
            # Look for True before False in the answer; default to keeping on tie
            t_idx = ans.find("true")
            f_idx = ans.find("false")
            if t_idx == -1 and f_idx == -1:
                log_entry["decision_reason"] = "no_boolean_default_keep"
                decision = True
            elif f_idx == -1:
                log_entry["decision_reason"] = "true_only"
                decision = True
            elif t_idx == -1:
                log_entry["decision_reason"] = "false_only"
                decision = False
            else:
                log_entry["decision_reason"] = "first_boolean_wins"
                decision = t_idx < f_idx
            log_entry["decision"] = decision
            return log_entry
        except Exception as e:
            print(f"[WARN] rerank call failed: {e} - treating as relevant")
            log_entry["decision"] = True
            log_entry["decision_reason"] = "error_default_keep"
            log_entry["error"] = str(e)
            return log_entry
        finally:
            log_entry["latency_ms"] = round((time.time() - started) * 1000)


async def filter_with_small_llm(query: str, results: List[Dict]) -> List[Dict]:
    """RankRAG-style re-ranking: keep the top-k relevant passages."""
    global _filter_llm_client, filter_decisions_log, filter_llm_calls_log
    if not results:
        return results

    if _filter_llm_client is None:
        from openai import AsyncOpenAI
        _filter_llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    semaphore = asyncio.Semaphore(FILTER_CONCURRENCY)
    passages = [
        (((r.get("title") or "")[:160]) + " — " + ((r.get("summary") or "")[:300])).strip(" —")
        for r in results
    ]
    candidate_logs = await asyncio.gather(
        *[
            _score_passage(query, p, results[i], i + 1, semaphore)
            for i, p in enumerate(passages)
        ]
    )
    filter_llm_calls_log.extend(candidate_logs)
    relevances = [entry.get("decision", True) for entry in candidate_logs]

    # Keep all results judged relevant, but guarantee at least top-k survive
    # (safety net: prevents the filter from discarding correct results).
    FILTER_MIN_KEEP = int(os.getenv("FILTER_MIN_KEEP", "3"))
    relevant_idx = [i for i, ok in enumerate(relevances) if ok]
    if len(relevant_idx) >= FILTER_MIN_KEEP:
        kept = [results[i] for i in relevant_idx]
    else:
        # Merge relevant + top-k (preserving order, no duplicates)
        keep_set = set(relevant_idx) | set(range(min(FILTER_MIN_KEEP, len(results))))
        kept = [results[i] for i in sorted(keep_set)]
    print(f"LLM filter kept {len(kept)}/{len(results)} results (min-keep={FILTER_MIN_KEEP})")

    # MODIFIED: persist filter decision for error analysis (per-task log)
    pre_urls = [r.get("url", "") for r in results]
    post_urls = [r.get("url", "") for r in kept]
    post_set = set(post_urls)
    removed_urls = [u for u in pre_urls if u not in post_set]
    filter_decisions_log.append({
        "query": query,
        "pre_filter_urls": pre_urls,
        "post_filter_urls": post_urls,
        "removed_urls": removed_urls,
        "pre_count": len(pre_urls),
        "post_count": len(post_urls),
        "filter_model": FILTER_MODEL,
        "filter_min_keep": FILTER_MIN_KEEP,
        "candidate_decisions": candidate_logs,
    })
    return kept


# ============================================================================
# MODIFIED: Optimization Method 2 — Agentic Plan Caching (Zhang et al. 2025,
# https://arxiv.org/abs/2506.14852). Plan templates are distilled from
# successful runs and stored under an exact-match intent keyword. Cache hit:
# the cheap CACHE_HIT_MODEL (gpt-4o-mini) executes the cached template and
# the large planner is skipped. Cache miss: the normal agent runs and, on
# success, a new template is extracted and stored.
# ============================================================================
CACHE_MODEL = os.getenv("CACHE_MODEL", "gpt-4o-mini")
CACHE_HIT_MODEL = os.getenv("CACHE_HIT_MODEL", CACHE_MODEL)
PLAN_CACHE_FILE = os.getenv("PLAN_CACHE_FILE", "cache/plan_cache.json")
CACHE_MATCH_POLICY = os.getenv("CACHE_MATCH_POLICY", "deterministic_gate")
CACHE_MAX_TEMPLATES_PER_KEYWORD = int(os.getenv("CACHE_MAX_TEMPLATES_PER_KEYWORD", "4"))
CACHE_MIN_STORE_F1 = float(os.getenv("CACHE_MIN_STORE_F1", "1.0"))
# MODIFIED: when set, the cache is read-only -- no new templates are distilled
# or stored during the run. Used for the evaluation runs so that only the
# disjoint warming pool can populate the cache, keeping the warming-to-eval
# measurement clean (no intra-run learning from earlier evaluation tasks).
CACHE_FREEZE = os.getenv("CACHE_FREEZE", "0").lower() in ("1", "true", "yes")
cache_token_tracker = {"prompt_tokens": 0, "completion_tokens": 0}
last_cache_hit = False
_cache_llm_client = None


def _normalize_cache_entry(keyword: str, entry: Any) -> Optional[Dict[str, Any]]:
    """Normalize legacy/new cache entries into a common internal structure."""
    if isinstance(entry, str):
        return {
            "keyword": keyword,
            "template": entry,
            "signature": {},
            "gate_key": "",
            "quality_score": 1.0,
            "support_count": 1,
            "usage_count": 0,
            "source_task_excerpt": "",
            "pitfalls": [],
        }
    if not isinstance(entry, dict):
        return None
    template = entry.get("template")
    if not isinstance(template, str) or not template.strip():
        return None
    signature = entry.get("signature", {})
    if not isinstance(signature, dict):
        signature = {}
    # MODIFIED: pitfalls list of {task_excerpt, description} dicts derived
    # from failed cold-run tasks of the same keyword. Injected into the
    # cache-hit prompt so the small LLM sees concrete anti-examples.
    pitfalls_raw = entry.get("pitfalls", []) or []
    pitfalls: List[Dict[str, str]] = []
    if isinstance(pitfalls_raw, list):
        for p in pitfalls_raw:
            if isinstance(p, dict) and p.get("task_excerpt"):
                pitfalls.append({
                    "task_excerpt": str(p.get("task_excerpt", "")).strip(),
                    "description": str(
                        p.get("description") or p.get("error_class") or ""
                    ).strip(),
                    "note": str(p.get("note", "")).strip(),
                })
    return {
        "keyword": keyword,
        "template": template,
        "signature": signature,
        "gate_key": str(entry.get("gate_key", "") or ""),
        "quality_score": float(entry.get("quality_score", 1.0) or 0.0),
        "support_count": int(entry.get("support_count", 1) or 1),
        "usage_count": int(entry.get("usage_count", 0) or 0),
        "source_task_excerpt": str(entry.get("source_task_excerpt", "") or ""),
        "pitfalls": pitfalls,
    }


def _load_plan_cache() -> Dict[str, List[Dict[str, Any]]]:
    if os.path.exists(PLAN_CACHE_FILE):
        try:
            with open(PLAN_CACHE_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "entries" in raw and isinstance(raw["entries"], dict):
                raw_entries = raw["entries"]
            elif isinstance(raw, dict):
                raw_entries = raw
            else:
                return {}
            normalized: Dict[str, List[Dict[str, Any]]] = {}
            for keyword, value in raw_entries.items():
                entries = value if isinstance(value, list) else [value]
                norm_entries = []
                for item in entries:
                    norm = _normalize_cache_entry(keyword, item)
                    if norm is not None:
                        norm_entries.append(norm)
                if norm_entries:
                    normalized[keyword] = norm_entries
            return normalized
        except Exception:
            return {}
    return {}


def _save_plan_cache(cache: Dict[str, List[Dict[str, Any]]]) -> None:
    d = os.path.dirname(PLAN_CACHE_FILE)
    if d:
        os.makedirs(d, exist_ok=True)
    serializable_entries: Dict[str, List[Dict[str, Any]]] = {}
    for keyword, entries in cache.items():
        serializable_entries[keyword] = []
        for entry in entries:
            serializable_entries[keyword].append({
                "keyword": keyword,
                "template": entry.get("template", ""),
                "signature": entry.get("signature", {}),
                "gate_key": entry.get("gate_key", "") or _cache_gate_key(entry.get("signature", {})),
                "quality_score": float(entry.get("quality_score", 0.0) or 0.0),
                "support_count": int(entry.get("support_count", 1) or 1),
                "usage_count": int(entry.get("usage_count", 0) or 0),
                "source_task_excerpt": entry.get("source_task_excerpt", ""),
                "pitfalls": entry.get("pitfalls", []) or [],
            })
    with open(PLAN_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "_meta": {
                "schema_version": 2,
                "entry_count": sum(len(v) for v in cache.values()),
            },
            "entries": serializable_entries,
        }, f, indent=2)


# Load cache at module import time so it persists across the run
plan_cache: Dict[str, List[Dict[str, Any]]] = _load_plan_cache() if OPTIMIZATION_METHOD == "caching" else {}
print(f"[plan-cache] loaded {sum(len(v) for v in plan_cache.values())} template(s) "
      f"across {len(plan_cache)} keyword(s) from {PLAN_CACHE_FILE}"
      if OPTIMIZATION_METHOD == "caching" else "")


# ============================================================================
# MODIFIED: Optimization Method 3 — Observation Masking (Lindenbauer et al.
# 2025, https://arxiv.org/abs/2508.21433). Rolling window: only the newest
# MASKING_WINDOW tool outputs stay in full, older ToolMessage contents are
# replaced. No extra LLM calls.
# ============================================================================
MASKING_WINDOW = int(os.getenv("MASKING_WINDOW", "2"))
# MODIFIED: MASKING_MODE controls how older tool outputs are compressed.
#   "placeholder"     (default, paper-faithful): replace with placeholder string
#                     "[Output omitted -- N lines]" (Lindenbauer et al. 2025).
#   "empty"           replace with the empty string, no placeholder at all.
#                     Tests whether the placeholder itself adds useful signal
#                     beyond pure removal.
MASKING_MODE = os.getenv("MASKING_MODE", "placeholder").lower()


def _extract_urls_from_tool_content(content: str) -> List[str]:
    """Best-effort extraction of product URLs from a ToolMessage content string."""
    if not content or not isinstance(content, str):
        return []
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return []
    urls: List[str] = []
    if isinstance(parsed, dict):
        results = parsed.get("results")
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict) and r.get("url"):
                    urls.append(r["url"])
        # Also handle get_product_details shape (top-level products list)
        products = parsed.get("products")
        if isinstance(products, list):
            for p in products:
                if isinstance(p, dict) and p.get("url"):
                    urls.append(p["url"])
    return urls


def _summarize_tool_content(content: str) -> Dict[str, Any]:
    """Best-effort summary of a ToolMessage for masking/error-analysis logs."""
    summary: Dict[str, Any] = {
        "tool_type": "unknown",
        "query": "",
        "urls": [],
    }
    if not content or not isinstance(content, str):
        return summary
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return summary

    if isinstance(parsed, dict):
        results = parsed.get("results")
        if isinstance(results, list):
            summary["tool_type"] = "search"
            summary["query"] = str(parsed.get("query", "") or "")
            summary["urls"] = [
                r.get("url", "") for r in results
                if isinstance(r, dict) and r.get("url")
            ]
            return summary

        details = parsed.get("product_details")
        if isinstance(details, list):
            summary["tool_type"] = "details"
            summary["urls"] = [
                r.get("url", "") for r in details
                if isinstance(r, dict) and r.get("url")
            ]
            return summary

        products = parsed.get("products")
        if isinstance(products, list):
            summary["tool_type"] = "details"
            summary["urls"] = [
                r.get("url", "") for r in products
                if isinstance(r, dict) and r.get("url")
            ]
            return summary

    return summary


def _mask_old_observations(state: dict) -> dict:
    """pre_model_hook: replace old ToolMessage contents with a short placeholder.
    Only the last MASKING_WINDOW tool outputs are kept in full.
    Returns llm_input_messages so the original state is not modified."""
    import copy
    global masking_events_log, _masked_message_ids_logged, current_expected_urls_for_masking

    orig_messages = state["messages"]
    tool_indices = [i for i, m in enumerate(orig_messages) if isinstance(m, ToolMessage)]

    tool_meta_by_index: Dict[int, Dict[str, Any]] = {}
    search_call_counter = 0
    details_call_counter = 0
    for tool_call_counter, idx in enumerate(tool_indices, start=1):
        meta = _summarize_tool_content(orig_messages[idx].content or "")
        meta["tool_call_sequence"] = tool_call_counter
        if meta["tool_type"] == "search":
            search_call_counter += 1
            meta["search_call_index"] = search_call_counter
        elif meta["tool_type"] == "details":
            details_call_counter += 1
            meta["details_call_index"] = details_call_counter
        tool_meta_by_index[idx] = meta

    # MODIFIED: capture masking event (once per message) BEFORE deepcopy/mask.
    if len(tool_indices) > MASKING_WINDOW:
        to_mask_idx = tool_indices[:-MASKING_WINDOW]
        for i in to_mask_idx:
            msg = orig_messages[i]
            msg_key = id(msg)
            if msg_key in _masked_message_ids_logged:
                continue
            orig_content = msg.content or ""
            meta = tool_meta_by_index.get(i, {})
            urls_before = list(meta.get("urls", []) or _extract_urls_from_tool_content(orig_content))
            masked_expected_urls = sorted({
                _norm_url_for_log(u)
                for u in urls_before
                if _norm_url_for_log(u) in current_expected_urls_for_masking
            })
            masking_events_log.append({
                "message_index": i,
                "tool_type": meta.get("tool_type", "unknown"),
                "search_query": meta.get("query", ""),
                "tool_call_sequence": meta.get("tool_call_sequence"),
                "search_call_index": meta.get("search_call_index"),
                "details_call_index": meta.get("details_call_index"),
                "masked_urls": urls_before,
                "masked_expected_urls": masked_expected_urls,
                "masked_expected_urls_count": len(masked_expected_urls),
                "masking_mode": MASKING_MODE,
                "masking_window": MASKING_WINDOW,
                "content_lines": (orig_content.count("\n") + 1) if orig_content else 0,
            })
            _masked_message_ids_logged.add(msg_key)

    messages = copy.deepcopy(orig_messages)
    if len(tool_indices) <= MASKING_WINDOW:
        return {"llm_input_messages": messages}
    to_mask = tool_indices[:-MASKING_WINDOW]
    for i in to_mask:
        orig = messages[i].content or ""
        n_lines = orig.count('\n') + 1
        if MASKING_MODE == "empty":
            # No-placeholder variant: pure content removal so the agent
            # receives only the tool-call evidence in the message history,
            # not the substituted "[Output omitted ...]" string.
            messages[i].content = ""
        else:
            # Default: placeholder mask, paper-faithful (Lindenbauer et al.).
            messages[i].content = f"[Output omitted -- {n_lines} lines]"
    print(f"  Observation masking ({MASKING_MODE}): masked {len(to_mask)} "
          f"old tool outputs (window={MASKING_WINDOW})")
    return {"llm_input_messages": messages}


async def _cache_llm_call(prompt: str, max_tokens: int = 2000,
                          call_type: str = "cache_llm_call",
                          metadata: Optional[Dict[str, Any]] = None) -> str:
    """Single cheap-LLM call with per-task token tracking."""
    global _cache_llm_client, cache_token_tracker, cache_llm_calls_log
    if _cache_llm_client is None:
        from openai import AsyncOpenAI
        _cache_llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    started = time.time()
    log_entry = {
        "model": CACHE_MODEL,
        "call_type": call_type,
        "metadata": metadata or {},
        "prompt_preview": prompt[:900],
        "prompt_chars": len(prompt),
        "max_completion_tokens": max_tokens,
        "response_preview": "",
        "response_raw": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency_ms": None,
        "error": None,
    }
    try:
        kwargs = {"model": CACHE_MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_completion_tokens": max_tokens}
        if CACHE_MODEL.startswith("gpt-5"):
            kwargs["reasoning_effort"] = "minimal"
        resp = await _cache_llm_client.chat.completions.create(**kwargs)
        if resp.usage:
            cache_token_tracker["prompt_tokens"] += resp.usage.prompt_tokens
            cache_token_tracker["completion_tokens"] += resp.usage.completion_tokens
            log_entry["prompt_tokens"] = resp.usage.prompt_tokens
            log_entry["completion_tokens"] = resp.usage.completion_tokens
            log_entry["total_tokens"] = getattr(
                resp.usage,
                "total_tokens",
                resp.usage.prompt_tokens + resp.usage.completion_tokens,
            )
        response_text = (resp.choices[0].message.content or "").strip()
        log_entry["response_preview"] = response_text[:900]
        log_entry["response_raw"] = response_text
        return response_text
    except Exception as e:
        print(f"[WARN] cache LLM call failed: {e}")
        log_entry["error"] = str(e)
        return ""
    finally:
        log_entry["latency_ms"] = round((time.time() - started) * 1000)
        cache_llm_calls_log.append(log_entry)


_KEYWORD_VOCAB = [
    "specific_product_search",     # task names a concrete product (brand+model)
    "vague_product_search",        # task describes desired properties, no named product
    "compatible_product_search",   # accessories/parts compatible with a named product
    "substitute_product_search",   # alternatives to a named product
    "cheapest_product_search",     # cheapest of a NAMED product (plain price comparison)
    "cheapest_specific_search",    # cheapest matching CONCRETE specs (e.g. 16 GB RAM)
    "cheapest_vague_search",       # cheapest matching VAGUE use-case (e.g. for streaming)
    "add_to_cart_and_checkout",    # transactional cart/checkout task
]

# MODIFIED: few-shot examples per keyword for the cache classifier. Synthetic
# products only (none from the evaluation subset), so the classifier learns
# the pattern without data leakage.
_KEYWORD_EXAMPLES = {
    "specific_product_search": [
        "Find all offers for the Sony Alpha 7C II.",
        "All offers for the Logitech G Pro X Superlight 2.",
    ],
    "vague_product_search": [
        "Find an ergonomic vertical mouse for long office days.",
        "Recommend lightweight running shoes for marathon training.",
        "Find all offers for the largest available Crucial BX series SSD.",
    ],
    "compatible_product_search": [
        "Find DDR5 memory modules compatible with the ASUS ROG B650-A motherboard.",
        "Find a charger compatible with the Dell XPS 13 (2024).",
        "Find keyboards with the same color as this case: https://example.com/product/corsair-case",
    ],
    "substitute_product_search": [
        "Alternatives to the Bose QuietComfort 45 headphones.",
        "Substitute products for the Apple Magic Keyboard.",
        "Find the cheapest alternative for this monitor: https://example.com/product/asus-27",
    ],
    "cheapest_product_search": [
        "Find the cheapest offer for the Sony WH-1000XM5.",
        "Cheapest offer for the Apple AirPods Pro 2.",
        "Find the cheapest offer for a Crucial MX500 1TB SATA SSD.",
    ],
    "cheapest_specific_search": [
        "Find the cheapest 32 GB DDR5 6000 MHz memory kit.",
        "Cheapest 27-inch 4K monitor with HDR support.",
        "Find the cheapest motherboard with RGB lighting and DDR5 support.",
    ],
    "cheapest_vague_search": [
        "Find the cheapest gaming laptop for streaming.",
        "Cheapest entry-level mirrorless camera for vlogging.",
        "Find the cheapest SSD suitable for everyday office use.",
    ],
    "add_to_cart_and_checkout": [
        "Add 2 USB-C cables to the cart and complete checkout at WebMall-1.",
        "Buy a Logitech webcam from WebMall-3 by adding it to cart and checking out.",
    ],
}


# MODIFIED (Layer A): deterministic pre-classification rules.
# Catches the unambiguous cases (URL-referenced substitute/compatible,
# cart/checkout actions) BEFORE the LLM is called. The LLM then only
# handles cases that genuinely require semantic judgment.
def _pre_classify_with_rules(task: str) -> "Optional[str]":
    """Try to classify with deterministic regex/keyword rules.
    Returns one of _KEYWORD_VOCAB on a confident match, else None."""
    if not task:
        return None
    t = task.lower()
    # Cart / checkout actions
    if any(w in t for w in [
        "add to cart", "add the following", "place an order",
        "checkout", "complete the purchase", "purchase the following",
    ]):
        return "add_to_cart_and_checkout"
    # URL reference to a product page indicates a "this product" task
    has_url = ("/product/" in t) or ("webmall-" in t and "://" in t)
    if has_url:
        # Substitute: explicit alternative wording
        if any(w in t for w in [
            "alternative", "substitute", "similar to",
            "replacement for", "instead of",
        ]):
            return "substitute_product_search"
        # Compatible: covers "compatible with", same-color/size, fit-mention,
        # backup-mention, generic "this case/product" referencing variants.
        if any(w in t for w in [
            "compatible", " fits ", "fit and", "best fit",
            "same color", "same size",
            "this case", "this product",
            "back up", "back-up", "backup",
        ]):
            return "compatible_product_search"
    return None


# MODIFIED (Layer C): hard-snap to vocabulary. If the LLM emits an
# invalid keyword (e.g. "cheapest_substitute_search"), we use fuzzy
# match to snap to the closest valid one rather than letting an
# invalid keyword pollute the cache.
def _snap_keyword_to_vocab(kw: str) -> str:
    """Map an LLM-emitted keyword to a valid vocab entry.
    Order: exact -> substring -> token-overlap -> fuzzy -> safe default."""
    if not kw:
        return "vague_product_search"
    if kw in _KEYWORD_VOCAB:
        return kw
    # Substring in either direction (legacy snap behaviour)
    for v in _KEYWORD_VOCAB:
        if v in kw or kw in v:
            return v
    # Token-overlap with priority: substitute/compatible/cart win over cheapest/specific
    tokens = set(kw.split("_"))
    priority_map = [
        ({"substitute", "alternative", "alternatives"}, "substitute_product_search"),
        ({"compatible", "compatibility", "fits"}, "compatible_product_search"),
        ({"cart", "checkout", "purchase", "buy"}, "add_to_cart_and_checkout"),
    ]
    for token_set, target in priority_map:
        if tokens & token_set:
            return target
    if "cheapest" in tokens or "lowest" in tokens or "price" in tokens:
        if "specific" in tokens or "spec" in tokens or "specs" in tokens:
            return "cheapest_specific_search"
        if "vague" in tokens:
            return "cheapest_vague_search"
        return "cheapest_product_search"
    if "specific" in tokens:
        return "specific_product_search"
    if "vague" in tokens:
        return "vague_product_search"
    # Last resort: fuzzy match
    import difflib
    matches = difflib.get_close_matches(kw, _KEYWORD_VOCAB, n=1, cutoff=0.5)
    if matches:
        return matches[0]
    print(f"[WARN] classifier emitted unrecognized keyword '{kw}' -> defaulting to vague_product_search")
    return "vague_product_search"


def _strip_task_boilerplate(task: str) -> str:
    """Extract the actual task description from the prompt boilerplate.
    Tasks are wrapped in <instructions>...</instructions> followed by
    <task>...</task>. The classifier only needs the <task> content -- the
    instructions block is the same for every task and adds noise that
    pushes the classifier toward the wrong keyword."""
    if not task:
        return ""
    import re as _re
    m = _re.search(r"<task>\s*(.+?)\s*</task>", task, flags=_re.DOTALL)
    if m:
        return m.group(1).strip()
    return task.strip()


def _extract_task_signature(task: str, keyword: str = "") -> Dict[str, Any]:
    """Create a compact structural signature for cache matching.
    The signature is intentionally coarse-grained: it captures task shape and
    domain tags without overfitting to concrete entities."""
    import re as _re

    task_only = _strip_task_boilerplate(task).lower()
    tags = set()

    tag_rules = {
        "connector": ["hdmi", "displayport", "mini displayport", "usb-c", "adapter", "cable"],
        "storage": ["ssd", "storage", "nvme", "sata", "external ssd", "m.2"],
        "memory": ["ram", "ddr4", "ddr5", "memory kit", "udimm", "dimm"],
        "gpu": ["gpu", "graphics card", "rtx", "radeon", "geforce"],
        "cpu": ["cpu", "processor", "ryzen", "threadripper", "intel core"],
        "motherboard": ["motherboard", "socket", "am5", "am4", "wrx8", "b650", "z790"],
        "monitor": ["monitor", "display", "27-inch", "27 inch", "1080p", "144hz", "4k"],
        "keyboard": ["keyboard", "keypad"],
        "audio": ["headphones", "earbuds", "speaker", "audio"],
        "camera": ["camera", "mirrorless", "dslr", "canon", "sony alpha"],
        "cooling": ["cooler", "aio", "liquid freezer", "fan", "radiator"],
        "phone": ["smartphone", "galaxy", "iphone", "pixel"],
        "color_match": ["same color", "white", "black", "rgb"],
        "size_match": ["same size", "27 inch", "32 gb", "1tb", "2tb", "4tb"],
        "backup": ["backup", "dump the data", "fully dump"],
        "performance_tradeoff": ["slight loss in performance", "alternative", "substitute"],
    }
    for tag, patterns in tag_rules.items():
        if any(p in task_only for p in patterns):
            tags.add(tag)

    if keyword == "compatible_product_search" and "connector" in tags:
        tags.add("compatible_connector")
    if keyword == "compatible_product_search" and "motherboard" in tags:
        tags.add("compatible_hardware")
    if keyword == "substitute_product_search" and "gpu" in tags:
        tags.add("substitute_gpu")
    if keyword.startswith("cheapest_") and "storage" in tags:
        tags.add("price_storage")

    numbers = _re.findall(r"\b\d+(?:\.\d+)?(?:tb|gb|mhz|hz|mm|inch|in|w)?\b", task_only)
    has_url = ("/product/" in task_only) or ("webmall-" in task_only and "://" in task_only)

    return {
        "keyword": keyword,
        "has_url": has_url,
        "asks_cheapest": any(w in task_only for w in ["cheapest", "lowest price", "best price"]),
        "has_numbers": bool(numbers),
        "numbers_bucket": min(len(numbers), 4),
        "tags": sorted(tags),
    }


_CACHE_FAMILY_TAGS = {
    "connector", "storage", "memory", "gpu", "cpu", "motherboard", "monitor",
    "keyboard", "audio", "camera", "cooling", "phone",
}
_CACHE_BEHAVIOR_TAGS = {
    "color_match", "size_match", "backup", "performance_tradeoff",
    "compatible_connector", "compatible_hardware", "substitute_gpu",
    "price_storage",
}
_CACHE_NUMERIC_SHAPE_KEYWORDS = {
    "vague_product_search",
    "cheapest_specific_search",
    "cheapest_vague_search",
    "compatible_product_search",
    "substitute_product_search",
}


# MODIFIED: CACHE_GATE_LEVEL selects the gate.
#   "shape_only" (default): kw + url + cheap.
#     Deterministic exact match. The template uses <PRODUCT>/<SPEC>
#     placeholders so family/behavior/numeric tags are noise. Hit-rate is
#     high; over-general templates can hurt F1.
#   "strict" (legacy): kw + url + cheap + num + family + behavior.
#     v1/v2 behaviour. Often blocks all reuse across disjoint warming pools.
CACHE_GATE_LEVEL = os.getenv("CACHE_GATE_LEVEL", "shape_only").lower()


def _cache_gate_key(signature: Dict[str, Any]) -> str:
    """Deterministic cache key for safe template reuse.

    Cache hits do not use a similarity score. A template is reused iff the
    current task and the cached task produce the same exact structural key
    at the configured CACHE_GATE_LEVEL.
    """
    keyword = str(signature.get("keyword", "") or "")
    url_flag = int(bool(signature.get("has_url")))
    cheap_flag = int(bool(signature.get("asks_cheapest")))
    if CACHE_GATE_LEVEL == "shape_only":
        return "|".join([
            f"kw={keyword}",
            f"url={url_flag}",
            f"cheap={cheap_flag}",
        ])
    # Legacy strict gate.
    tags = set(signature.get("tags", []) or [])
    family = ",".join(sorted(tags & _CACHE_FAMILY_TAGS)) or "none"
    behavior = ",".join(sorted(tags & _CACHE_BEHAVIOR_TAGS)) or "none"
    number_shape = (
        str(int(signature.get("numbers_bucket", 0) or 0))
        if keyword in _CACHE_NUMERIC_SHAPE_KEYWORDS
        else "any"
    )
    return "|".join([
        f"kw={keyword}",
        f"url={url_flag}",
        f"cheap={cheap_flag}",
        f"num={number_shape}",
        f"family={family}",
        f"behavior={behavior}",
    ])


def _cache_signature_is_reusable(signature: Dict[str, Any]) -> bool:
    """Fail closed when a task is too underspecified for deterministic reuse.

    MODIFIED: when CACHE_GATE_LEVEL == "shape_only" the gate itself only
    consults kw/url/cheap, so the family/behavior-tag reusability check is
    redundant -- skip it and accept any signature except cart/checkout
    (which is excluded from the retrieval-only experiment anyway).
    """
    keyword = str(signature.get("keyword", "") or "")
    if keyword == "add_to_cart_and_checkout":
        return False
    if CACHE_GATE_LEVEL == "shape_only":
        return True

    tags = set(signature.get("tags", []) or [])
    family_tags = tags & _CACHE_FAMILY_TAGS
    behavior_tags = tags & _CACHE_BEHAVIOR_TAGS

    if keyword in {"specific_product_search", "cheapest_product_search"}:
        return bool(family_tags)
    if keyword in {"vague_product_search", "cheapest_vague_search"}:
        return bool(family_tags and behavior_tags)
    if keyword == "compatible_product_search":
        return bool({"compatible_connector", "compatible_hardware", "backup", "color_match", "size_match"} & behavior_tags)
    if keyword == "substitute_product_search":
        return bool(family_tags and (behavior_tags or signature.get("has_url")))
    if keyword == "cheapest_specific_search":
        return bool(family_tags and signature.get("has_numbers"))
    return False


async def _select_cached_template(task: str, keyword: str) -> Optional[Dict[str, Any]]:
    """Return a deterministically compatible cached template, else None."""
    candidates = plan_cache.get(keyword, [])
    if not candidates:
        return None
    signature = _extract_task_signature(task, keyword)
    if not _cache_signature_is_reusable(signature):
        return None
    current_gate_key = _cache_gate_key(signature)
    shape_matches: List[Dict[str, Any]] = []
    for entry in candidates:
        template = (entry.get("template") or "").strip()
        if not template:
            continue
        cached_signature = entry.get("signature", {})
        cached_gate_key = _cache_gate_key(cached_signature)
        if cached_gate_key == current_gate_key:
            shape_matches.append(entry)
    if not shape_matches:
        return None

    best_entry = dict(max(
        shape_matches,
        key=lambda item: (
            int(item.get("support_count", 1) or 1),
            len(item.get("template", "") or ""),
        )
    ))

    best_entry["_current_signature"] = signature
    best_entry["_cache_match_policy"] = CACHE_MATCH_POLICY
    best_entry["_cache_gate_key"] = current_gate_key
    best_entry["_cached_gate_key"] = best_entry.get("gate_key") or _cache_gate_key(best_entry.get("signature", {}))
    return best_entry


def _store_plan_template(keyword: str, task: str, template: str, quality_score: float) -> None:
    """Store or update a reusable template under a keyword with signature metadata."""
    if not template.strip():
        return
    signature = _extract_task_signature(task, keyword)
    entry = {
        "keyword": keyword,
        "template": template,
        "signature": signature,
        "gate_key": _cache_gate_key(signature),
        "quality_score": quality_score,
        "support_count": 1,
        "usage_count": 0,
        "source_task_excerpt": _strip_task_boilerplate(task)[:220],
        "pitfalls": [],
    }
    entries = plan_cache.setdefault(keyword, [])
    merged = False
    for existing in entries:
        existing_gate_key = existing.get("gate_key") or _cache_gate_key(existing.get("signature", {}))
        if existing_gate_key == entry["gate_key"]:
            existing["support_count"] = int(existing.get("support_count", 1) or 1) + 1
            existing["quality_score"] = max(float(existing.get("quality_score", 0.0) or 0.0), quality_score)
            # Prefer the richer template if it is at least as good.
            if len(template) >= len(existing.get("template", "")) and quality_score >= float(existing.get("quality_score", 0.0) or 0.0):
                existing["template"] = template
            existing["signature"] = signature
            existing["gate_key"] = entry["gate_key"]
            merged = True
            break
    if not merged:
        entries.append(entry)
    entries.sort(
        key=lambda x: (
            int(x.get("support_count", 1) or 1),
            len(x.get("template", "")),
        ),
        reverse=True,
    )
    del entries[CACHE_MAX_TEMPLATES_PER_KEYWORD:]


async def extract_keyword(task: str) -> str:
    """APC step 1: classify the task into one of a small fixed vocabulary of
    intent keywords. A closed set is essential for cache hit rate -- free-form
    keywords (paper §3.2 figure 3) become too task-specific and fragment the
    cache. Our 45-task subset maps onto 7 subcategories from Steiner.

    MODIFIED (3-layer classifier):
      A. Deterministic pre-rules catch URL-referenced substitute/compatible
         and cart/checkout actions before any LLM call.
      B. LLM call with boilerplate-stripped task + few-shot examples (incl.
         URL-referenced patterns) and an explicit decision flow.
      C. Hard snap to vocabulary on the LLM's output (fuzzy match fallback)
         so invalid keywords cannot pollute the cache.
    """
    global cache_llm_calls_log
    task_only = _strip_task_boilerplate(task)

    # Layer A: deterministic rules first
    rule_kw = _pre_classify_with_rules(task_only)
    if rule_kw is not None:
        cache_llm_calls_log.append({
            "model": "deterministic_rules",
            "call_type": "keyword_classification_rule",
            "metadata": {"task_excerpt": task_only[:300]},
            "prompt_preview": "",
            "prompt_chars": 0,
            "response_preview": rule_kw,
            "response_raw": rule_kw,
            "parsed_keyword": rule_kw,
            "snapped_keyword": rule_kw,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0,
            "error": None,
        })
        return rule_kw

    # Layer B: LLM classification with examples
    vocab_lines = []
    for k in _KEYWORD_VOCAB:
        examples = _KEYWORD_EXAMPLES.get(k, [])
        ex_str = " | ".join(f'"{e}"' for e in examples)
        vocab_lines.append(f"  - {k}\n      examples: {ex_str}")
    vocab_str = "\n".join(vocab_lines)

    out = await _cache_llm_call(
        "You classify e-commerce agent tasks into EXACTLY ONE of the "
        "intent keywords below. Examples illustrate the pattern -- the "
        "actual task may use different products.\n\n"
        "Decision flow (apply in order):\n"
        "1. Is it a CART or CHECKOUT action? -> add_to_cart_and_checkout\n"
        "2. Does it ask for ALTERNATIVES / SUBSTITUTES to a NAMED product? "
        "-> substitute_product_search\n"
        "3. Does it ask for accessories or parts COMPATIBLE WITH a NAMED "
        "reference product? -> compatible_product_search\n"
        "4. Does it ask for the CHEAPEST / LOWEST-PRICE option?\n"
        "   4a. ...of a NAMED product (no extra constraints) -> "
        "cheapest_product_search\n"
        "   4b. ...matching CONCRETE specs (numbers, sizes, ports) -> "
        "cheapest_specific_search\n"
        "   4c. ...matching a VAGUE use-case ('for gaming', 'for office') "
        "-> cheapest_vague_search\n"
        "5. Does it NAME a concrete product (brand + model)? -> "
        "specific_product_search\n"
        "6. Otherwise (describes desired properties without naming a "
        "product) -> vague_product_search\n\n"
        "Important disambiguations:\n"
        "- 'cheapest', 'lowest price', 'best price' always wins over "
        "specific/vague -- if these words appear, choose a cheapest_* "
        "keyword.\n"
        "- 'alternatives to X' or 'substitutes for X' is NEVER cheapest, "
        "even if price is mentioned secondarily.\n"
        "- A task with 'compact', 'ergonomic', 'best fit', 'for <use-case>' "
        "and NO concrete product model is vague.\n\n"
        f"Allowed keywords:\n{vocab_str}\n\n"
        "Output ONLY the chosen keyword on the last line, nothing else.\n\n"
        f"Task: {task_only}\n\nKeyword:",
        max_tokens=400,
        call_type="keyword_classification",
        metadata={
            "task_excerpt": task_only[:300],
            "allowed_keywords": list(_KEYWORD_VOCAB),
        },
    )
    kw = (out.split("\n")[-1] if out else "").strip().strip("\"'`").lower()
    kw = kw.replace(" ", "_")
    # Layer C: hard snap to vocabulary -- never pollute cache with garbage keywords
    snapped = _snap_keyword_to_vocab(kw)
    if cache_llm_calls_log and cache_llm_calls_log[-1].get("call_type") == "keyword_classification":
        cache_llm_calls_log[-1]["parsed_keyword"] = kw
        cache_llm_calls_log[-1]["snapped_keyword"] = snapped
    return snapped


_CATEGORY_HINTS = {
    "specific_product_search": (
        "STRATEGY: The user names a specific product. Search once with the "
        "exact product name; the same product appears in multiple stores. "
        "Return ALL matching URLs across all 4 webmall stores."
    ),
    "vague_product_search": (
        "STRATEGY: The user describes requirements without naming a product. "
        "Run 2-3 broad searches with different keyword combinations covering "
        "the requirements, then fetch details to verify the specs match. "
        "Return URLs from all matching stores."
    ),
    "cheapest_product_search": (
        "STRATEGY: The user wants the CHEAPEST option of a NAMED product. "
        "The same product is sold in all 4 stores at DIFFERENT prices. You "
        "MUST find the product in every store and compare prices, then "
        "return ONLY the URL(s) with the lowest price. Use search + "
        "get_product_details on the candidates from each store."
    ),
    "cheapest_specific_search": (
        "STRATEGY: The user wants the CHEAPEST product matching CONCRETE "
        "specs (e.g. specific RAM size, port count, exact resolution). "
        "Search broadly first, then enforce the spec constraints via "
        "get_product_details, then pick the lowest-price candidate that "
        "fully matches all specs. Do NOT return products that violate any "
        "concrete spec, even if they are cheaper."
    ),
    "cheapest_vague_search": (
        "STRATEGY: The user wants the CHEAPEST product fitting a VAGUE "
        "use-case ('for streaming', 'for office', 'beginner-friendly'). "
        "Use multiple broad searches to surface candidates, fetch details "
        "to verify the use-case fits, then return the lowest-price "
        "candidate. Vague criteria require checking content, not just "
        "title/summary."
    ),
    "compatible_product_search": (
        "STRATEGY: The user wants accessories/parts compatible with a "
        "reference product. Search for the accessory type combined with the "
        "reference product's brand/model. Return all compatible URLs."
    ),
    "substitute_product_search": (
        "STRATEGY: The user wants alternatives to a named product. Search for "
        "products in the same category with similar specs. Return all "
        "substitute URLs from all stores."
    ),
    "add_to_cart_and_checkout": (
        "STRATEGY: Multi-step transactional task. Search for the requested "
        "items, add them to the appropriate store carts, then checkout each "
        "store. Return the checkout confirmation URLs."
    ),
}


# MODIFIED: the tools the agent actually has. Used to constrain the template
# generator and to strip hallucinated tools (e.g. "compare_prices()") from
# generated templates.
_KNOWN_TOOL_NAMES = {
    "search_products",
    "get_product_details",
    "add_to_cart_webmall_1", "add_to_cart_webmall_2",
    "add_to_cart_webmall_3", "add_to_cart_webmall_4",
    "checkout_webmall_1", "checkout_webmall_2",
    "checkout_webmall_3", "checkout_webmall_4",
}

_TOOL_INVENTORY_PROMPT = (
    "AVAILABLE TOOLS (use ONLY these — do not invent any others):\n"
    "  - search_products(query='<...>', match_count=<int>)\n"
    "  - get_product_details(n_urls=<int>)\n"
    "  - add_to_cart_webmall_<1-4>(...)  [only for transactional tasks]\n"
    "  - checkout_webmall_<1-4>(...)      [only for transactional tasks]\n"
)


def _sanitize_template_steps(template: str) -> str:
    """Drop any 'STEPS:' lines that reference tools not in _KNOWN_TOOL_NAMES,
    then renumber remaining steps. Catches hallucinated tools like
    compare_prices() or return_lowest_price_urls() that the generator LLM
    sometimes invents."""
    if not template:
        return template
    import re as _re
    lines = template.split("\n")
    out: List[str] = []
    in_steps = False
    step_idx = 0
    dropped = 0
    for line in lines:
        if _re.match(r"^\s*STEPS:\s*$", line):
            in_steps = True
            out.append(line)
            continue
        if in_steps and _re.match(r"^\s*(NOTES|STRATEGY)\s*:", line):
            in_steps = False
            out.append(line)
            continue
        if in_steps:
            m = _re.match(r"^\s*\d+\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", line)
            if m:
                tool_name = m.group(1)
                if tool_name not in _KNOWN_TOOL_NAMES:
                    dropped += 1
                    continue  # drop hallucinated tool step
                step_idx += 1
                # renumber: replace leading "<n>." with "<step_idx>."
                line = _re.sub(r"^\s*\d+\.\s*", f"{step_idx}. ", line)
        out.append(line)
    if dropped:
        print(f"  template sanitizer: dropped {dropped} step(s) with unknown tools")
    return "\n".join(out)


async def extract_plan_template(task: str, tool_calls_log: List[Dict],
                                 keyword: str = "") -> str:
    """APC step 2: distill a generalized, reusable plan template from a
    successful execution log. Two-stage filter per paper §3.1:
      (1) rule-based: keep only tool name + key args (drop verbose outputs)
      (2) LLM-based:  generalize away task-specific entities.
    Plus an additional post-filter that drops steps referencing tools the
    agent does not actually have."""
    steps = []
    for call in tool_calls_log:
        name = call.get("tool_name", "?")
        args = call.get("tool_args", {}) or {}
        if name == "search_products":
            steps.append(
                f"search_products(query='{args.get('query','')}', "
                f"match_count={args.get('match_count', 30)})")
        elif name == "get_product_details":
            urls = args.get("urls", []) or []
            steps.append(f"get_product_details(n_urls={len(urls)})")
        else:
            # MODIFIED: rule-based pre-filter -- drop unknown tools at this
            # stage already so the generator never sees them in the log
            if name in _KNOWN_TOOL_NAMES:
                steps.append(f"{name}({list(args.keys())})")
    if not steps:
        return ""
    raw_log = "\n".join(steps)
    hint = _CATEGORY_HINTS.get(keyword, "")
    signature = _extract_task_signature(task, keyword)
    template = await _cache_llm_call(
        "You are building a REUSABLE PLAN TEMPLATE for an e-commerce agent. "
        "The template will be applied to OTHER tasks of the same type, so it "
        "MUST NOT contain any concrete product names, brands, models, colors, "
        "sizes, prices, or other entities from the example task below. "
        "Replace EVERY concrete noun with a placeholder: <PRODUCT>, <BRAND>, "
        "<MODEL>, <COLOR>, <SIZE>, <SPEC>, <PRICE>.\n\n"
        # MODIFIED: explicit tool-inventory constraint to stop the generator
        # from inventing plausible-sounding but non-existent tools.
        f"{_TOOL_INVENTORY_PROMPT}\n"
        "STRICT CONSTRAINTS:\n"
        "- Use ONLY the tool names listed above.\n"
        "- Maximum of 5 STEPS total.\n"
        "- Keep the plan flexible: do NOT hard-code exact brands/models from "
        "the example task.\n"
        "- Prefer ranges/roles over brittle constants. Example: "
        "get_product_details(n_urls=3-8 candidates) is better than a very "
        "specific count copied from one task.\n"
        "- Do NOT invent helper functions like compare_prices(), "
        "return_lowest_price_urls(), filter_results(), etc. -- they do "
        "not exist. Any reasoning, comparison, or selection happens "
        "implicitly inside the agent.\n"
        "- Each step must follow exactly: 'N. tool_name(arg=<placeholder>)'\n\n"
        f"TASK CATEGORY: {keyword}\n"
        f"TASK SIGNATURE: {json.dumps(signature, ensure_ascii=True)}\n"
        f"{hint}\n\n"
        f"EXAMPLE TASK (for context only -- do NOT copy its entities):\n{task}\n\n"
        f"EXAMPLE EXECUTION LOG:\n{raw_log}\n\n"
        "Output format (no commentary, no code fences):\n"
        "STRATEGY: <one or two sentences describing the general approach>\n"
        "STEPS:\n"
        "1. <generalized tool call with placeholders>\n"
        "2. ...\n"
        "NOTES: <any pitfalls, e.g. 'must check all 4 stores for cheapest'>\n",
        max_tokens=1500,
        call_type="template_extraction",
        metadata={
            "keyword": keyword,
            "task_excerpt": _strip_task_boilerplate(task)[:300],
            "raw_execution_steps": steps,
            "signature": signature,
        },
    )
    # MODIFIED: post-filter -- drop any steps that still reference unknown tools
    template = _sanitize_template_steps(template)
    if cache_llm_calls_log and cache_llm_calls_log[-1].get("call_type") == "template_extraction":
        cache_llm_calls_log[-1]["sanitized_template"] = template
        cache_llm_calls_log[-1]["sanitized_template_steps"] = _extract_template_steps_for_log(template)
    return template


@tool
async def search_products(query: str, match_count: int = 30, use_hybrid: bool = True) -> str:
    """Search for products in the webmall database. Use this to find specific products, compatible items, or browse categories.

    Args:
        query: The search query. Be specific and use exact product names when possible.
        match_count: Number of results to retrieve (1-100). Use more results for broad searches, fewer for specific items. Default is 30.
        use_hybrid: Whether to use hybrid search (combines semantic + keyword matching). Generally recommended. Default is True.

    Returns:
        JSON string containing search results with status, query, results count, and product information.
    """
    global search_history, search_results_cache, token_tracker

    # Validate match_count
    match_count = max(1, min(100, match_count))

    print(
        f"\nSEARCH TOOL: Query='{query}', Results={match_count}, Mode={'hybrid' if use_hybrid else 'semantic'}")

    # Get embedding for the query
    query_embedding, embedding_tokens = await get_embedding(query)
    token_tracker["embedding_tokens"] += embedding_tokens

    # Perform search
    if use_hybrid:
        # results = await es_client.hybrid_search(query, query_embedding, match_count)
        # results = await es_client.hybrid_search_content_only(query, query_embedding, match_count)
        results = await es_client.hybrid_search_title_content(query, query_embedding, match_count)
    else:
        results = await es_client.semantic_search(query_embedding, match_count)

    if OPTIMIZATION_METHOD == "filtering":
        results = await filter_with_small_llm(query, results)

    # Store results in cache for easy access
    search_results_cache.append(results)

    # Track search in history
    search_record = {
        "call_index": len(search_history) + 1,
        "tool_call_sequence": _next_tool_call_sequence(),
        "query": query,
        "match_count": match_count,
        "use_hybrid": use_hybrid,
        "results_found": len(results),
        # MODIFIED: log the actual URL list (post-filter if filtering) so that
        # error analysis can reconstruct what the agent saw per search.
        "result_urls": [r.get("url", "") for r in results],
        "timestamp": datetime.now().isoformat()
    }
    search_history.append(search_record)

    print(f"Found {len(results)} results")

    # Return structured response with full results for the agent
    return_string = json.dumps({
        "status": "success",
        "query": query,
        "results_count": len(results),
        "results": [
            {
                "title": r["title"],
                "url": r["url"],
                # "description": r.get("content", "N/A")[:250]
            }
            # Return lightweight results, let the agent fetch details for specific products
            for r in results
        ],
        "search_mode": "hybrid" if use_hybrid else "semantic"
    })
    # print(return_string)
    return return_string


@tool
async def get_product_details(urls: List[str]) -> str:
    """Get detailed information for specific product URLs. Use this after search_products to get full descriptions and details for products you're interested in.

    Args:
        urls: List of product URLs to fetch details for. Maximum 20 URLs per request.

    Returns:
        JSON string containing detailed product information including descriptions, content, summaries, prices, and shop information.
    """
    global token_tracker, details_history

    # Validate and limit URLs
    if not urls:
        return json.dumps({"status": "error", "error": "No URLs provided"})

    urls = urls[:20]  # Limit to 20 URLs to prevent excessive token usage

    print(f"\nDETAILS TOOL: Fetching details for {len(urls)} URLs")

    try:
        # Fetch detailed information from Elasticsearch
        detailed_results = await es_client.get_documents_by_urls(urls)

        details_history.append({
            "call_index": len(details_history) + 1,
            "tool_call_sequence": _next_tool_call_sequence(),
            "requested_urls": list(urls),
            "result_urls": [r.get("url", "") for r in detailed_results if r.get("url")],
            "timestamp": datetime.now().isoformat(),
        })

        print(f"Retrieved details for {len(detailed_results)} products")

        # Return structured response with detailed information
        return_string = json.dumps({
            "status": "success",
            "urls_requested": len(urls),
            "details_found": len(detailed_results),
            "product_details": [
                {
                    "title": r["title"],
                    "url": r["url"],
                    "description": r.get("content", "N/A")
                }
                for r in detailed_results
            ]
        })

        return return_string

    except Exception as e:
        print(f"Error fetching product details: {e}")
        return json.dumps({
            "status": "error",
            "error": str(e),
            "urls_requested": len(urls)
        })


def aggregate_search_results(all_results: List[List[Dict]], expected_urls: List[str]) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Aggregate results from multiple searches, removing duplicates and tracking ranks.

    Returns:
        - Deduplicated results sorted by best score
        - Mapping of expected URLs to their best ranks across all searches
    """
    # Dictionary to track best result for each URL
    url_best_results = {}
    url_best_ranks = {}

    # Process each search result set
    for search_idx, results in enumerate(all_results):
        for rank, doc in enumerate(results, 1):
            url = doc['url']
            score = doc.get('score', doc.get('similarity', 0))

            # Track best result for this URL
            if url not in url_best_results or score > url_best_results[url].get('score', 0):
                url_best_results[url] = doc
                url_best_results[url]['best_search_idx'] = search_idx
                url_best_results[url]['aggregated_score'] = score

            # Track best rank for expected URLs
            normalized_url = normalize_url(url)
            for expected_url in expected_urls:
                if normalize_url(expected_url) == normalized_url:
                    if expected_url not in url_best_ranks or rank < url_best_ranks[expected_url]:
                        url_best_ranks[expected_url] = rank

    # Sort aggregated results by score
    aggregated_results = sorted(
        url_best_results.values(),
        key=lambda x: x.get('aggregated_score', 0),
        reverse=True
    )

    return aggregated_results, url_best_ranks


def extract_urls_from_cart_tool_output(tool_output: str, tool_name: str) -> List[str]:
    """Extract URLs from cart/checkout tool output."""
    try:
        if isinstance(tool_output, str):
            response_data = json.loads(tool_output)
        else:
            response_data = tool_output

        urls = set()  # Use set to avoid duplicates

        # For cart tools, prioritize cart_urls if available
        if "cart_urls" in response_data:
            urls.update(response_data["cart_urls"])
        elif "cart" in response_data and isinstance(response_data["cart"], list):
            # Fallback to extracting from cart items if cart_urls not available
            for item in response_data["cart"]:
                if "url" in item:
                    urls.add(item["url"])

        # For checkout tools, prioritize product_urls if available
        if "product_urls" in response_data:
            urls.update(response_data["product_urls"])
        elif "items" in response_data and isinstance(response_data["items"], list):
            # Fallback to extracting from items if product_urls not available
            for item in response_data["items"]:
                if "url" in item:
                    urls.add(item["url"])

        return list(urls)

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"Warning: Could not extract URLs from {tool_name} output: {e}")
        return []


def message_content_to_text(content: Any) -> str:
    """Normalize LangChain/SSE message content to plain text."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text") or item.get(
                    "content") or item.get("value")
                if isinstance(text_value, str):
                    parts.append(text_value)
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    return str(content)


def parse_model_answer(answer: Any) -> List[str]:
    """Parse the model answer and return the list of URLs."""
    normalized_answer = message_content_to_text(answer)
    import re

    def _regex_fallback(text: str) -> List[str]:
        """MODIFIED: extract any webmall product URLs from free-form text.
        Triggered when the model returns Markdown instead of a JSON array
        (e.g. weaker cache-hit small actor LLMs like gpt-4o-mini)."""
        urls = [u for u in re.findall(r"https?://[^\s\)\]\"'<>]+", text)
                if "webmall" in u]
        # De-dup while preserving order, strip trailing punctuation
        seen, out = set(), []
        for u in urls:
            u = u.rstrip(".,;:)")
            if u not in seen:
                seen.add(u)
                out.append(u)
        return out

    try:
        # Try to extract JSON array from the response
        if normalized_answer.startswith('[') and normalized_answer.endswith(']'):
            return json.loads(normalized_answer)
        else:
            # If response contains JSON within text, try to find it
            json_match = re.search(r'\[.*\]', normalized_answer, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            # MODIFIED: fall back to regex URL extraction before giving up
            urls = _regex_fallback(normalized_answer)
            if urls:
                return urls
            cleaned = normalized_answer.strip()
            return [cleaned] if cleaned.lower() != "done" else ["Done"]
    except json.JSONDecodeError:
        print("Warning: Could not parse JSON response, treating as plain text")
        urls = _regex_fallback(normalized_answer)
        if urls:
            return urls
        return [part.strip() for part in normalized_answer.split("###") if part.strip()]


def create_fallback_result(user_task: str, urls_in_db: List[str], expected_flat: List[str],
                           total_tokens_used: Dict, error_message: str, execution_time: float,
                           preserve_current_logs: bool = True) -> Dict:
    """Create a fallback result structure when the agent fails."""
    print(f"CREATING FALLBACK RESULT: {error_message}")
    error_type = "GraphRecursionError" if "GraphRecursionError" in error_message else "AgentExecutionError"
    partial_tool_log = _build_partial_tool_calls_log() if preserve_current_logs else []
    current_masking_events = list(masking_events_log) if preserve_current_logs else []
    current_filter_decisions = list(filter_decisions_log) if preserve_current_logs else []
    current_filter_llm_calls = list(filter_llm_calls_log) if preserve_current_logs else []
    current_cache_llm_calls = list(cache_llm_calls_log) if preserve_current_logs else []
    masking_summary = (
        _build_masking_summary(expected_flat, parsed_urls=[], error_type=error_type)
        if preserve_current_logs else {}
    )
    cache_summary = (
        _build_cache_summary(expected_flat, parsed_urls=[], error_type=error_type)
        if preserve_current_logs else {}
    )

    return {
        "parsed_urls": [],
        "answer": f"AGENT_FAILED: {error_message}",
        "search_history": list(search_history) if preserve_current_logs else [],
        "total_searches": len(search_history) if preserve_current_logs else 0,
        "aggregated_results": [],
        "rag_exact_url_matches": [],
        "rag_total_matches": 0,
        "rag_coverage": 0.0,
        "url_ranks": {},
        "best_rank": None,
        "avg_rank": None,
        "db_coverage": len(urls_in_db) / len(expected_flat) if expected_flat else 0,
        "search_type": "multi_search_failed",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "execution_time_seconds": execution_time,
        "tool_calls_log": partial_tool_log,
        "cart_checkout_urls": [],
        "filter_decisions": current_filter_decisions,
        "filter_llm_calls": current_filter_llm_calls,
        "filter_model": FILTER_MODEL if preserve_current_logs and OPTIMIZATION_METHOD == "filtering" else None,
        "masking_events": current_masking_events,
        "masking_summary": masking_summary,
        "cache_hit": last_cache_hit if preserve_current_logs else False,
        "agent_model_used": current_agent_model_used if preserve_current_logs else None,
        "cache_keyword_used": cache_keyword_used if preserve_current_logs else None,
        "cache_template_applied": cache_template_applied if preserve_current_logs else None,
        "cache_summary": cache_summary,
        "cache_llm_calls": current_cache_llm_calls,
        "cache_helper_model": CACHE_MODEL if preserve_current_logs and OPTIMIZATION_METHOD == "caching" else None,
        "cache_hit_model": CACHE_HIT_MODEL if preserve_current_logs and last_cache_hit else None,
        "error_occurred": True,
        "error_message": error_message,
        "error_type": error_type
    }


async def get_model_answer(user_task: str, urls_in_db: List[str], expected_flat: List[str],
                           total_tokens_used: Dict, chat_model: Any,
                           model_name: str = "gpt-4") -> Dict:
    """Get model answer using the LangGraph RAG system with tool binding."""

    # Start execution timer
    task_start_time = time.time()

    print("Starting RAG workflow...")

    # Reset global variables for this task
    global search_history, details_history, search_results_cache, tool_call_sequence, filter_token_tracker
    global cache_token_tracker, last_cache_hit
    global filter_decisions_log, filter_llm_calls_log, masking_events_log, _masked_message_ids_logged
    global current_expected_urls_for_masking
    global cache_keyword_used, cache_template_applied, cache_template_metadata
    global cache_llm_calls_log
    global current_agent_model_used
    search_history = []
    details_history = []
    search_results_cache = []
    tool_call_sequence = 0
    # MODIFIED: reset filter token tracker per task so we can attribute
    # filter cost to each individual task in the per-task CSV.
    filter_token_tracker = {"prompt_tokens": 0, "completion_tokens": 0}
    # MODIFIED: reset plan-caching trackers per task
    cache_token_tracker = {"prompt_tokens": 0, "completion_tokens": 0}
    last_cache_hit = False
    # MODIFIED: reset per-task error-analysis logs
    filter_decisions_log = []
    filter_llm_calls_log = []
    masking_events_log = []
    _masked_message_ids_logged = set()
    current_expected_urls_for_masking = {
        _norm_url_for_log(u)
        for u in (expected_flat or [])
        if u
    }
    cache_keyword_used = None
    cache_template_applied = None
    cache_template_metadata = None
    cache_llm_calls_log = []
    current_agent_model_used = model_name

    # Create the system prompt with intelligent search strategy guidance
    system_prompt = """You are an advanced RAG-capable agent that can browse four webshops, find product offers, manage shopping carts, and complete purchases.
You have access to search functions, product detail fetching, cart management tools, and checkout capabilities for all four shops.

AVAILABLE TOOLS:
- search_products: Search for products across all shops (returns title + URL only for token efficiency)
- get_product_details: Get detailed descriptions for specific URLs (use after search to get full info)
- add_to_cart_webmall_1 through add_to_cart_webmall_4: Add products to specific shop carts
- checkout_webmall_1 through checkout_webmall_4: Complete purchases with customer details

EFFICIENT SEARCH WORKFLOW:
1. Use search_products first to get an overview of available products (lightweight: title + URL only)
2. Review search results and identify promising products
3. Use get_product_details for URLs you're interested in to get full descriptions, specs, and pricing
4. Make decisions based on detailed information

TASK-SPECIFIC INSTRUCTIONS:

FOR SEARCH TASKS:
1. OPTIMIZE FOR EFFICIENCY: Use the two-step search approach to minimize token usage.
2. QUERY ANALYSIS: 
   - Simple specific product searches (e.g., "AMD Ryzen 9 5900X"): Start with 10-15 results
   - Complex/compatibility queries (e.g., "cables compatible with..."): May need 30 results
   - Comparison queries (e.g., "better than X"): Break into separate searches
3. Get details only for products that seem relevant from titles
4. After analysis, return a JSON array with the EXACT URLs of all relevant products found.

FOR ADD TO CART TASKS:
1. First search for the requested products
2. Extract the product URLs from search results
3. Group URLs by shop (webmall_1, webmall_2, etc.)
4. Call the appropriate add_to_cart tool for each shop with their respective URLs
5. Return the URLs of products successfully added to carts

FOR CHECKOUT TASKS:
1. If products are already in cart, proceed directly to checkout
2. If starting fresh, first add products to cart
3. Call the checkout tool with all required customer and payment information
4. Return the product URLs from the completed order

RESPONSE FORMAT:
- For all tasks: Return a JSON array containing the EXACT URLs
- Format: ["url1", "url2", ...] or ["Done"] if task completed/no results"""

    # Get cart tools
    cart_tools = get_cart_tools()

    # MODIFIED: Method 2 - Plan Caching: try keyword lookup before invoking
    # the expensive main agent. On cache hit, run a small (gpt-5-nano) agent
    # with the cached template baked into the system prompt.
    cached_template = None
    cached_template_entry = None
    keyword_for_cache = None
    if OPTIMIZATION_METHOD == "caching":
        keyword_for_cache = await extract_keyword(user_task)
        lookup_signature = _extract_task_signature(user_task, keyword_for_cache)
        lookup_gate_key = _cache_gate_key(lookup_signature)
        cached_template_entry = await _select_cached_template(user_task, keyword_for_cache)
        cached_template = cached_template_entry.get("template") if cached_template_entry else None
        # MODIFIED: record keyword used this task (whether or not there was a hit)
        cache_keyword_used = keyword_for_cache
        gate_key = cached_template_entry.get("_cache_gate_key", lookup_gate_key) if cached_template_entry else lookup_gate_key
        print(f"plan-cache keyword='{keyword_for_cache}' "
              f"hit={cached_template is not None} "
              f"policy={CACHE_MATCH_POLICY} "
              f"gate='{gate_key}' "
              f"(cache size={sum(len(v) for v in plan_cache.values())})")

    if cached_template is not None:
        last_cache_hit = True
        # MODIFIED: record the template text that was actually injected
        cache_template_applied = cached_template
        cache_template_metadata = {
            "cache_match_policy": cached_template_entry.get("_cache_match_policy"),
            "cache_gate_key": cached_template_entry.get("_cache_gate_key"),
            "cached_gate_key": cached_template_entry.get("_cached_gate_key"),
            "current_signature": cached_template_entry.get("_current_signature", {}),
            "cached_signature": cached_template_entry.get("signature", {}),
            "quality_score": cached_template_entry.get("quality_score"),
            "support_count": cached_template_entry.get("support_count"),
            "usage_count_before": cached_template_entry.get("usage_count", 0),
            "source_task_excerpt": cached_template_entry.get("source_task_excerpt", ""),
            # MODIFIED: surface pitfalls so error analysis can correlate
            # Caching B improvements with the presence of anti-examples.
            "pitfalls_count": len(cached_template_entry.get("pitfalls", []) or []),
            "pitfalls": cached_template_entry.get("pitfalls", []) or [],
        }
        cached_template_entry["usage_count"] = int(cached_template_entry.get("usage_count", 0) or 0) + 1
        small_model = create_chat_model(
            CACHE_HIT_MODEL,
            reasoning_effort=os.getenv("CACHE_HIT_REASONING", "low"),
            temperature=None if CACHE_HIT_MODEL.startswith("gpt-") else 0.0,
        )
        # MODIFIED: build FAILURE NOTES block from failed cold-run tasks of
        # the same keyword. Empty → block omitted.
        pitfalls_entries = cached_template_entry.get("pitfalls", []) or []
        pitfalls_block = ""
        if pitfalls_entries:
            bullets = []
            for p in pitfalls_entries[:5]:  # cap to keep prompt small
                excerpt = (p.get("task_excerpt") or "").strip()
                desc = (
                    p.get("description") or p.get("error_class") or ""
                ).strip()
                if not excerpt:
                    continue
                line = f"- Task: \"{excerpt}\""
                if desc:
                    line += f"\n  Why it failed: {desc}"
                bullets.append(line)
            if bullets:
                pitfalls_block = (
                    "\n\nFAILURE NOTES (similar tasks that previously failed "
                    "-- avoid these patterns):\n" + "\n".join(bullets)
                )
        hit_system_prompt = system_prompt + (
            "\n\n========================================\n"
            "CACHED PLAN TEMPLATE FROM A PREVIOUS SIMILAR TASK\n"
            "========================================\n"
            f"{cached_template}"
            f"{pitfalls_block}\n"
            "========================================\n"
            "Adapt the placeholders (<PRODUCT>, <BRAND>, <SPEC>, ...) to the "
            "specific items requested in the CURRENT task.\n"
            "Treat the cached template as a strong prior, not as a rigid "
            "script. If the current task shape differs, preserve the overall "
            "strategy but adjust search queries, candidate counts, and detail "
            "fetching to fit the current task."
        )
        agent_kwargs = {"model": small_model, "tools": [search_products, get_product_details, *cart_tools]}
        if OPTIMIZATION_METHOD == "masking":
            agent_kwargs["pre_model_hook"] = _mask_old_observations
        agent = create_react_agent(**agent_kwargs)
        active_system_prompt = hit_system_prompt
        current_agent_model_used = CACHE_HIT_MODEL
    else:
        # Cache miss (or method != caching): create the normal large agent
        agent_kwargs = {"model": chat_model, "tools": [search_products, get_product_details, *cart_tools]}
        if OPTIMIZATION_METHOD == "masking":
            agent_kwargs["pre_model_hook"] = _mask_old_observations
        agent = create_react_agent(**agent_kwargs)
        active_system_prompt = system_prompt
        current_agent_model_used = model_name

    # Run the agent with proper token tracking and error handling
    try:
        with get_usage_metadata_callback() as cb:
            result = await agent.ainvoke(
                {"messages": [SystemMessage(
                    content=active_system_prompt), HumanMessage(content=user_task)]},
                config={"recursion_limit": 50}  # Increase recursion limit
            )
    except GraphRecursionError as e:
        execution_time = time.time() - task_start_time
        error_msg = f"GraphRecursionError: Recursion limit exceeded - {str(e)}"
        print(f"AGENT RECURSION ERROR: {error_msg}")
        print(f"  Failed after {execution_time:.2f} seconds")
        return create_fallback_result(user_task, urls_in_db, expected_flat, total_tokens_used, error_msg, execution_time)
    except Exception as e:
        execution_time = time.time() - task_start_time
        error_msg = f"Unexpected agent error: {str(e)}"
        print(f"AGENT UNEXPECTED ERROR: {error_msg}")
        print(f"  Failed after {execution_time:.2f} seconds")
        return create_fallback_result(user_task, urls_in_db, expected_flat, total_tokens_used, error_msg, execution_time)

    # Extract token usage from callback
    usage_data = cb.usage_metadata
    print(f"Token Usage: {usage_data}")

    for _, usage in usage_data.items():
        total_tokens_used["prompt_tokens"] += usage.get("input_tokens", 0)
        total_tokens_used["completion_tokens"] += usage.get("output_tokens", 0)
        total_tokens_used["total_tokens"] += usage.get("total_tokens", 0)

    # Get search results from our cache (populated by the tool calls)
    all_search_results = search_results_cache.copy()

    # Track tool calls and extract cart/checkout URLs
    agent_messages = result.get("messages", [])
    tool_calls_log = []
    cart_checkout_urls = set()

    # Add search history to tool calls log
    for search_record in search_history:
        tool_calls_log.append({
            "tool_name": "search_products",
            "tool_args": {
                "call_index": search_record["call_index"],
                "tool_call_sequence": search_record.get("tool_call_sequence"),
                "query": search_record["query"],
                "match_count": search_record["match_count"],
                "use_hybrid": search_record["use_hybrid"]
            },
            "tool_output": {
                "results_found": search_record["results_found"],
                "status": "success",
                # MODIFIED: carry URL list so error analysis can see what the agent saw
                "result_urls": search_record.get("result_urls", []),
            },
            "timestamp": search_record["timestamp"],
            "tool_type": "search"
        })

    # Process all messages to extract tool calls
    details_call_counter_for_log = 0
    for msg in agent_messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_output = None
                tool_output_parsed = None

                # Find corresponding tool result message
                for result_msg in agent_messages:
                    if hasattr(result_msg, 'tool_call_id') and result_msg.tool_call_id == tool_call.get("id"):
                        tool_output = result_msg.content
                        # Try to parse JSON output for structured data
                        try:
                            if tool_output and tool_output.strip().startswith('{'):
                                tool_output_parsed = json.loads(tool_output)
                        except json.JSONDecodeError:
                            pass
                        break

                # Determine tool type
                tool_type = "unknown"
                if tool_name == "search_products":
                    tool_type = "search"
                elif tool_name == "get_product_details":
                    tool_type = "details"
                    details_call_counter_for_log += 1
                elif tool_name.startswith("add_to_cart_"):
                    tool_type = "cart"
                elif tool_name.startswith("checkout_"):
                    tool_type = "checkout"

                # Log tool call with enhanced information
                tool_call_entry = {
                    "tool_name": tool_name,
                    "tool_type": tool_type,
                    "tool_args": tool_call.get("args", {}),
                    "tool_output_raw": tool_output,
                    "timestamp": datetime.now().isoformat()
                }

                # Add parsed output if available
                if tool_output_parsed:
                    tool_call_entry["tool_output_parsed"] = tool_output_parsed

                    # Add specific metrics for different tool types
                    if tool_type == "cart" and "cart" in tool_output_parsed:
                        tool_call_entry["items_in_cart"] = len(
                            tool_output_parsed.get("cart", []))
                        tool_call_entry["total_quantity"] = tool_output_parsed.get(
                            "total_items", 0)
                    elif tool_type == "checkout" and "items" in tool_output_parsed:
                        tool_call_entry["items_purchased"] = len(
                            tool_output_parsed.get("items", []))
                        tool_call_entry["order_id"] = tool_output_parsed.get(
                            "order_id", "")
                        tool_call_entry["total_amount"] = tool_output_parsed.get(
                            "total", "0.00")
                if tool_type == "details":
                    tool_call_entry.setdefault("tool_args", {})
                    tool_call_entry["tool_args"]["call_index"] = details_call_counter_for_log
                    detail_record = (
                        details_history[details_call_counter_for_log - 1]
                        if details_call_counter_for_log <= len(details_history)
                        else {}
                    )
                    tool_call_entry["tool_args"]["tool_call_sequence"] = detail_record.get("tool_call_sequence")

                # Skip search_products as they're already added above
                if tool_name != "search_products":
                    tool_calls_log.append(tool_call_entry)

                # Extract URLs from cart/checkout tools
                if tool_name.startswith(("add_to_cart_", "checkout_")) and tool_output:
                    urls = extract_urls_from_cart_tool_output(
                        tool_output, tool_name)
                    cart_checkout_urls.update(urls)
                    print(
                        f"Extracted {len(urls)} URLs from {tool_name}: {urls}")

    # Get final answer from the agent's last message
    final_message = agent_messages[-1] if agent_messages else None
    answer = final_message.content if final_message and hasattr(
        final_message, 'content') else "No answer provided"

    # Aggregate all search results
    aggregated_results, url_ranks = aggregate_search_results(
        all_search_results, expected_flat)

    # Parse the agent's final answer directly
    parsed_urls = parse_model_answer(answer)
    masking_summary = _build_masking_summary(expected_flat, parsed_urls, error_type=None)
    cache_summary = _build_cache_summary(expected_flat, parsed_urls, error_type=None)

    print(f"\nTOOL EXECUTION SUMMARY:")
    search_tools = [t for t in tool_calls_log if t.get(
        "tool_type") == "search"]
    details_tools = [t for t in tool_calls_log if t.get(
        "tool_type") == "details"]
    cart_tools = [t for t in tool_calls_log if t.get("tool_type") == "cart"]
    checkout_tools = [t for t in tool_calls_log if t.get(
        "tool_type") == "checkout"]

    print(f"  - Total tool calls: {len(tool_calls_log)}")
    print(f"  - Search tools: {len(search_tools)}")
    print(f"  - Details tools: {len(details_tools)}")
    print(f"  - Cart tools: {len(cart_tools)}")
    print(f"  - Checkout tools: {len(checkout_tools)}")
    print(f"  - Total unique products found: {len(aggregated_results)}")
    print(f"  - Agent returned {len(parsed_urls)} URLs")

    if search_tools:
        print(
            f"  - Search queries: {[t['tool_args'].get('query') for t in search_tools]}")
    if cart_checkout_urls:
        print(f"  - Cart/Checkout URLs: {len(cart_checkout_urls)} URLs")

    # Check coverage
    found_urls = [r['url'] for r in aggregated_results]
    found_normalized = [normalize_url(url) for url in found_urls]

    exact_url_matches = []
    for expected_url in expected_flat:
        if normalize_url(expected_url) in found_normalized:
            exact_url_matches.append(expected_url)

    rag_coverage = len(exact_url_matches) / \
        len(expected_flat) if expected_flat else 0

    # Display coverage and ranking info
    if url_ranks:
        best_rank = min(url_ranks.values())
        avg_rank = sum(url_ranks.values()) / len(url_ranks)

    # Calculate execution time
    execution_time = time.time() - task_start_time
    print(f"  Execution time: {execution_time:.2f} seconds")

    # Calculate retrieval metrics
    db_coverage = len(urls_in_db) / len(expected_flat) if expected_flat else 0

    # MODIFIED: Method 2 - Plan Caching: on a successful cache miss, extract a
    # generalized plan template from the execution log and persist it under
    # the task keyword for future reuse.
    if (OPTIMIZATION_METHOD == "caching"
            and not CACHE_FREEZE
            and not last_cache_hit
            and keyword_for_cache
            and parsed_urls
            and tool_calls_log):
        try:
            exact_task_success = 1.0 if set(map(normalize_url, parsed_urls)) == set(map(normalize_url, expected_flat)) else 0.0
            f1_like = (
                (2 * len(exact_url_matches)) / max(len(parsed_urls) + len(expected_flat), 1)
                if (parsed_urls or expected_flat) else 0.0
            )
            quality_score = max(exact_task_success, f1_like)
            if quality_score >= CACHE_MIN_STORE_F1:
                tmpl = await extract_plan_template(user_task, tool_calls_log, keyword_for_cache)
                if tmpl:
                    _store_plan_template(keyword_for_cache, user_task, tmpl, quality_score)
                    _save_plan_cache(plan_cache)
                    print(
                        f"stored plan template for keyword='{keyword_for_cache}' "
                        f"(templates for keyword={len(plan_cache.get(keyword_for_cache, []))}, "
                        f"total templates={sum(len(v) for v in plan_cache.values())})"
                    )
            else:
                print(
                    f"skipped cache-store for keyword='{keyword_for_cache}' "
                    f"(quality_score={quality_score:.2f} < {CACHE_MIN_STORE_F1:.2f})"
                )
        except Exception as e:
            print(f"[WARN] template extraction failed: {e}")

    # Return comprehensive results
    return {
        "parsed_urls": parsed_urls,
        "answer": answer,
        "search_history": search_history,
        "total_searches": len(all_search_results),
        "aggregated_results": aggregated_results,
        "rag_exact_url_matches": exact_url_matches,
        "rag_total_matches": len(exact_url_matches),
        "rag_coverage": rag_coverage,
        "url_ranks": url_ranks,
        "best_rank": min(url_ranks.values()) if url_ranks else None,
        "avg_rank": sum(url_ranks.values()) / len(url_ranks) if url_ranks else None,
        "db_coverage": db_coverage,
        "search_type": "multi_search",
        "prompt_tokens": total_tokens_used["prompt_tokens"],
        "completion_tokens": total_tokens_used["completion_tokens"],
        "total_tokens": total_tokens_used["total_tokens"],
        # MODIFIED: per-task small-LLM filter token usage (filtering optimization)
        "filter_prompt_tokens": filter_token_tracker["prompt_tokens"],
        "filter_completion_tokens": filter_token_tracker["completion_tokens"],
        # MODIFIED: per-task plan-caching cheap-LLM token usage + hit flag
        "cache_prompt_tokens": cache_token_tracker["prompt_tokens"],
        "cache_completion_tokens": cache_token_tracker["completion_tokens"],
        "cache_hit": last_cache_hit,
        "agent_model_used": current_agent_model_used,
        # MODIFIED: per-task error-analysis fields
        "filter_decisions": list(filter_decisions_log),
        "filter_llm_calls": list(filter_llm_calls_log),
        "filter_model": FILTER_MODEL if OPTIMIZATION_METHOD == "filtering" else None,
        "masking_events": list(masking_events_log),
        "masking_summary": masking_summary,
        "cache_keyword_used": cache_keyword_used,
        "cache_template_applied": cache_template_applied,
        "cache_summary": cache_summary,
        "cache_llm_calls": list(cache_llm_calls_log),
        "cache_helper_model": CACHE_MODEL if OPTIMIZATION_METHOD == "caching" else None,
        "cache_hit_model": CACHE_HIT_MODEL if last_cache_hit else None,
        "execution_time_seconds": execution_time,
        "tool_calls_log": tool_calls_log,
        "cart_checkout_urls": list(cart_checkout_urls)
    }


# Load benchmark JSON file
# MODIFIED: Use 45-task challenging subset instead of full 91 tasks
# BENCHMARK_JSON_PATH = "task_sets/task_sets.json"
# MODIFIED: env-driven so cold runs can use a disjoint warming set
# (task_sets_warming.json) while warm runs evaluate on the regular subset.
BENCHMARK_JSON_PATH = os.getenv(
    "BENCHMARK_JSON_PATH", "task_sets/task_sets_subset.json"
)

with open(BENCHMARK_JSON_PATH, "r", encoding="utf-8") as f:
    benchmark = json.load(f)
print(f"[benchmark] loaded {sum(len(ts['tasks']) for ts in benchmark)} task(s) "
      f"from {BENCHMARK_JSON_PATH}")


async def process_benchmark(model_name: str, chat_model: Any):
    """Process benchmark tasks using the LangGraph RAG workflow."""
    print("\n" + "=" * 60)
    print("BENCHMARK - RAG SYSTEM")
    print("=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)

    reasoning_effort = extract_reasoning_effort(chat_model)
    # MODIFIED: separate output dir per optimization method so baseline and
    # optimized runs do not get mixed up in the same folder
    interface_label = "rag" if OPTIMIZATION_METHOD == "none" else f"rag-{OPTIMIZATION_METHOD}"
    results_output_dir = interface_results_dir(
        __file__, interface_label, model_name, reasoning_effort)
    # Create a run-unique timestamp early so stream files are consistent
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Prepare incremental output files (streaming)
    incremental_csv_file = results_output_dir / \
        f"benchmark_metrics_{current_timestamp}_stream.csv"
    incremental_jsonl_file = results_output_dir / \
        f"benchmark_results_{current_timestamp}.jsonl"

    # Initialize the streaming CSV with header
    with incremental_csv_file.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "category",
            "task_id",
            "task_completion_rate",
            "avg_precision",
            "avg_recall",
            "f1_score",
            "prompt_tokens",
            "completion_tokens",
            "filter_prompt_tokens",
            "filter_completion_tokens",
            "cache_prompt_tokens",
            "cache_completion_tokens",
            "cache_hit",
            "execution_duration"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    # Reset all carts before starting benchmark
    reset_all_carts()
    print("Reset all shopping carts before starting benchmark")

    # Initialize results list
    results = []

    # Initialize counters for overall statistics
    total_urls_not_in_db = 0
    total_expected_urls = 0
    total_searches_performed = 0
    total_execution_time = 0.0
    total_tool_calls = 0
    total_cart_tools = 0
    total_checkout_tools = 0

    # Error tracking
    failed_tasks = 0
    recursion_errors = 0
    other_errors = 0

    # Ranking metrics
    total_rank_sum = 0
    total_ranked_results = 0
    top3_count = 0
    top10_count = 0

    # Initialize token tracking
    total_tokens_used = {
        "embedding_tokens": 0,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0
    }

    # Process tasks
    for task_set in benchmark:
        for task in task_set["tasks"]:
            # if task["id"] != "Webmall_Add_To_Cart_Task4":
            #    continue

            print(f"\n=== TASK {task['id']} ===")
            # Reset carts before each task to ensure clean state
            reset_all_carts()
            # Track the start time for the whole task (in case preprocessing fails)
            task_preprocess_start = time.time()

            # Defaults to ensure we can still log on failure
            expected_flat = []
            urls_in_db = []
            urls_not_in_db = []
            user_task = ""
            task_category = task.get("category", "Search")
            preprocessing_failed = False

            try:
                # Check if correct answers exist in the database
                correct_answer = task.get(
                    "correct_answer", {}).get("answers", [])
                expected_flat = [fill_urls(x, URLS) for x in correct_answer]

                # Check database for exact URL matches
                for expected_url in expected_flat:
                    if not expected_url.endswith("/"):
                        expected_url += "/"
                    if await es_client.check_url_exists(expected_url):
                        urls_in_db.append(expected_url)
                    else:
                        urls_not_in_db.append(expected_url)

                total_urls_not_in_db += len(urls_not_in_db)

                if len(urls_not_in_db) > 0:
                    print(
                        f"    WARNING: {len(urls_not_in_db)} correct answer(s) not found in database!")

                total_expected_urls += len(expected_flat)

                # Get task category for evaluation logic
                task_category = task.get("category", "Search")
                print(f"  Task Category: {task_category}")

                # Extract user task
                user_task = task["task"] if "task" in task else None
                if not user_task:
                    start = task["instruction"].find("<task>")
                    end = task["instruction"].find("</task>") + len("</task>")
                    user_task = task["instruction"][start:end]

                user_task = fill_urls(user_task, URLS)
                user_task = user_task.replace(
                    "<task>", "").replace("</task>", "")

                if "{{product_url}}" in user_task:
                    user_task = user_task.replace(
                        "{{product_url}}", str(expected_flat))

                if "{{email}}" in user_task:
                    user_details = task["user_details"]
                    user_task = user_task.replace(
                        "{{name}}", user_details["name"])
                    user_task = user_task.replace(
                        "{{email}}", user_details["email"])
                    user_task = user_task.replace(
                        "{{street}}", user_details["street"])
                    user_task = user_task.replace(
                        "{{house_number}}", user_details["house_number"])
                    user_task = user_task.replace(
                        "{{zip}}", user_details["zip"])
                    user_task = user_task.replace(
                        "{{city}}", user_details["city"])
                    user_task = user_task.replace(
                        "{{state}}", user_details["state"])
                    user_task = user_task.replace(
                        "{{country}}", user_details["country"])

                    # Replace payment info placeholders
                    payment_info = task["payment_info"]
                    user_task = user_task.replace(
                        "{{card}}", payment_info["card"])
                    user_task = user_task.replace(
                        "{{cvv}}", payment_info["cvv"])
                    user_task = user_task.replace(
                        "{{expiry_date}}", payment_info["expiry_date"])

                print(f"User task: {user_task}")
            except Exception as e:
                # If preprocessing fails, ensure we still log a result for this task
                preprocessing_failed = True
                error_msg = f"TaskProcessingError: {str(e)}"
                print(
                    f"PREPROCESSING ERROR in task {task['id']}: {error_msg}")
                execution_time = time.time() - task_preprocess_start
                # Create a fallback result so downstream logging works
                model_result = create_fallback_result(
                    user_task or "",
                    urls_in_db,
                    expected_flat,
                    total_tokens_used,
                    error_msg,
                    execution_time,
                    preserve_current_logs=False,
                )
                failed_tasks += 1
                other_errors += 1

            # Capture token usage before this task
            tokens_before_task = {
                "prompt_tokens": total_tokens_used["prompt_tokens"],
                "completion_tokens": total_tokens_used["completion_tokens"],
                "total_tokens": total_tokens_used["total_tokens"]
            }

            # Get model answer using LangGraph system with error handling
            if not preprocessing_failed:
                try:
                    model_result = await get_model_answer(
                        user_task,
                        urls_in_db,
                        expected_flat,
                        total_tokens_used,
                        chat_model=chat_model,
                        model_name=model_name
                    )

                    # Check if this was a failed result
                    if model_result.get("error_occurred", False):
                        failed_tasks += 1
                        error_type = model_result.get("error_type", "unknown")
                        if error_type == "GraphRecursionError":
                            recursion_errors += 1
                        else:
                            other_errors += 1
                        print(
                            f"  Task {task['id']} failed with error: {model_result.get('error_message', 'Unknown error')}")

                except Exception as e:
                    # Fallback error handling if even the error handling fails
                    execution_time = time.time() - task_preprocess_start
                    error_msg = f"Critical benchmark error: {str(e)}"
                    print(
                        f"CRITICAL ERROR in task {task['id']}: {error_msg}")
                    model_result = create_fallback_result(
                        user_task, urls_in_db, expected_flat, total_tokens_used, error_msg, execution_time)
                    failed_tasks += 1
                    other_errors += 1

            # Calculate per-task token usage
            task_tokens = {
                "prompt_tokens": total_tokens_used["prompt_tokens"] - tokens_before_task["prompt_tokens"],
                "completion_tokens": total_tokens_used["completion_tokens"] - tokens_before_task["completion_tokens"],
                "total_tokens": total_tokens_used["total_tokens"] - tokens_before_task["total_tokens"]
            }

            # Track total searches and execution time
            total_searches_performed += model_result["total_searches"]
            total_execution_time += model_result["execution_time_seconds"]

            # Track tool call statistics
            tool_history = model_result.get("tool_calls_log", [])
            total_tool_calls += len(tool_history)
            total_cart_tools += len(
                [t for t in tool_history if t.get("tool_type") == "cart"])
            total_checkout_tools += len(
                [t for t in tool_history if t.get("tool_type") == "checkout"])

            # Extract values from the result
            parsed_urls = model_result["parsed_urls"]
            url_ranks = model_result["url_ranks"]
            best_rank = model_result["best_rank"]
            avg_rank = model_result["avg_rank"]
            cart_checkout_urls = model_result.get("cart_checkout_urls", [])

            # Determine which URLs to use for evaluation based on task category
            if task_category in ["Add_To_Cart", "Checkout", "FindAndOrder"]:
                # For cart/checkout tasks, use URLs from cart/checkout operations
                evaluation_urls = [normalize_url(url)
                                   for url in cart_checkout_urls]
                print(
                    f"Using cart/checkout URLs for evaluation: {cart_checkout_urls}")
            else:
                # For search tasks, use the final answer
                evaluation_urls = [normalize_url(url.strip())
                                   for url in parsed_urls if url.strip().lower() != "done"]
                print(
                    f"Using final answer URLs for evaluation: {parsed_urls}")

            # Ranking metrics
            if best_rank is not None:
                total_rank_sum += best_rank
                total_ranked_results += 1

                if best_rank <= 3:
                    top3_count += 1
                if best_rank <= 10:
                    top10_count += 1

            # Calculate accuracy metrics using evaluation URLs
            expected_normalized = [normalize_url(url) for url in expected_flat]

            # Handle failed tasks with empty metrics
            if model_result.get("error_occurred", False):
                # For failed tasks, set all metrics to 0
                correct_model_answers = []
                additional_urls = []
                missing_urls = expected_flat  # All expected URLs are missing
                metrics = {
                    "task_completion_rate": 0,
                    "avg_precision": 0.0,
                    "avg_recall": 0.0,
                    "f1_score": 0.0
                }
                print(
                    f"FAILED TASK METRICS: All metrics set to 0 for task {task['id']}")
            else:
                correct_model_answers = [
                    url for url in expected_flat if normalize_url(url) in evaluation_urls]
                additional_urls = [
                    url for url in evaluation_urls if url not in expected_normalized]
                missing_urls = [
                    url for url in expected_normalized if url not in evaluation_urls]

                metrics = calculation_results(expected_flat, evaluation_urls)

            print("Metrics:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.2f}" if isinstance(
                    v, float) else f"  {k}: {v}")

            # Save results with collected metrics
            task_result = {
                "task_id": task["id"],
                "user_task": user_task,
                "metrics": metrics,
                "parsed_urls": parsed_urls,
                "db_urls_found": urls_in_db,
                "db_urls_missing": urls_not_in_db,
                "db_coverage": model_result["db_coverage"],
                "tool_history": model_result["tool_calls_log"],
                "total_searches": model_result["total_searches"],
                "rag_exact_url_matches": model_result["rag_exact_url_matches"],
                "rag_total_matches": model_result["rag_total_matches"],
                "rag_coverage": model_result["rag_coverage"],

                # Ranking details
                "search_type": model_result["search_type"],
                "url_rank_details": url_ranks,
                "best_rank": best_rank,
                "avg_rank": avg_rank,
                "multi_search": True,

                # Execution time
                "execution_time_seconds": model_result["execution_time_seconds"],

                "correct_answers": expected_normalized,
                "correct_model_answers": correct_model_answers,
                "additional_urls": additional_urls,
                "missing_urls": missing_urls,
                "parsed_model_response": parsed_urls,
                "model_response": model_result["answer"],
                "task_category": task["id"].split("_Task")[0],
                "evaluation_urls": evaluation_urls,
                "cart_checkout_urls": cart_checkout_urls,
                "prompt_tokens": task_tokens["prompt_tokens"],
                "completion_tokens": task_tokens["completion_tokens"],
                "total_tokens": task_tokens["total_tokens"],
                # MODIFIED: carry small-LLM filter token usage into task_result
                "filter_prompt_tokens": model_result.get("filter_prompt_tokens", 0),
                "filter_completion_tokens": model_result.get("filter_completion_tokens", 0),
                # MODIFIED: carry plan-caching cheap-LLM token usage + hit flag
                "cache_prompt_tokens": model_result.get("cache_prompt_tokens", 0),
                "cache_completion_tokens": model_result.get("cache_completion_tokens", 0),
                "cache_hit": model_result.get("cache_hit", False),
                "agent_model_used": model_result.get("agent_model_used"),
                # MODIFIED: carry error-analysis logs into per-task jsonl
                "filter_decisions": model_result.get("filter_decisions", []),
                "filter_llm_calls": model_result.get("filter_llm_calls", []),
                "filter_model": model_result.get("filter_model"),
                "masking_events": model_result.get("masking_events", []),
                "masking_summary": model_result.get("masking_summary", {}),
                "cache_keyword_used": model_result.get("cache_keyword_used"),
                "cache_template_applied": model_result.get("cache_template_applied"),
                "cache_summary": model_result.get("cache_summary", {}),
                "cache_llm_calls": model_result.get("cache_llm_calls", []),
                "cache_helper_model": model_result.get("cache_helper_model"),
                "cache_hit_model": model_result.get("cache_hit_model"),
                "error_occurred": model_result.get("error_occurred", False),
                "error_message": model_result.get("error_message"),
                "error_type": model_result.get("error_type")
            }

            results.append(task_result)

            # Append per-task result to JSONL for crash-safe logging
            try:
                with incremental_jsonl_file.open("a", encoding="utf-8") as jf:
                    jf.write(json.dumps(task_result) + "\n")
            except Exception as e:
                print(f"  Failed to append JSONL for task {task['id']}: {e}")

            # Append a row to the streaming CSV
            try:
                row = {
                    "category": task.get("category", "Unknown"),
                    "task_id": task.get("id", ""),
                    "task_completion_rate": metrics.get("task_completion_rate", 0),
                    "avg_precision": metrics.get("avg_precision", 0.0),
                    "avg_recall": metrics.get("avg_recall", 0.0),
                    "f1_score": metrics.get("f1_score", 0.0),
                    "prompt_tokens": 0 if model_result.get("error_occurred", False) else task_tokens.get("prompt_tokens", 0),
                    "completion_tokens": 0 if model_result.get("error_occurred", False) else task_tokens.get("completion_tokens", 0),
                    "filter_prompt_tokens": model_result.get("filter_prompt_tokens", 0),
                    "filter_completion_tokens": model_result.get("filter_completion_tokens", 0),
                    "cache_prompt_tokens": model_result.get("cache_prompt_tokens", 0),
                    "cache_completion_tokens": model_result.get("cache_completion_tokens", 0),
                    "cache_hit": model_result.get("cache_hit", False),
                    "execution_duration": model_result.get("execution_time_seconds", 0)
                }
                with incremental_csv_file.open("a", newline="", encoding="utf-8") as csvfile:
                    fieldnames = [
                        "category",
                        "task_id",
                        "task_completion_rate",
                        "avg_precision",
                        "avg_recall",
                        "f1_score",
                        "prompt_tokens",
                        "completion_tokens",
                        "filter_prompt_tokens",
                        "filter_completion_tokens",
                        "cache_prompt_tokens",
                        "cache_completion_tokens",
                        "cache_hit",
                        "execution_duration"
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writerow(row)
            except Exception as e:
                print(f"  Failed to append CSV for task {task['id']}: {e}")
            # break
    # Generate results file

    # Calculate execution time statistics
    avg_execution_time = total_execution_time / len(results) if results else 0
    min_execution_time = min(r["execution_time_seconds"]
                             for r in results) if results else 0
    max_execution_time = max(r["execution_time_seconds"]
                             for r in results) if results else 0

    # Enhanced benchmark summary with detailed metrics
    benchmark_summary = {
        "benchmark_metadata": {
            "timestamp": current_timestamp,
            "version": "langgraph",
            "model": model_name,
            "reasoning_effort": reasoning_effort,
            "results_directory": str(results_output_dir),
            "total_tasks": len(results),
            "total_searches_performed": total_searches_performed,
            "avg_searches_per_task": total_searches_performed / len(results) if results else 0,
            "total_tool_calls": total_tool_calls,
            "total_cart_tools": total_cart_tools,
            "total_checkout_tools": total_checkout_tools,
            "avg_tools_per_task": total_tool_calls / len(results) if results else 0,
            "token_usage": total_tokens_used,
            "execution_time_stats": {
                "total_seconds": total_execution_time,
                "average_seconds": avg_execution_time,
                "min_seconds": min_execution_time,
                "max_seconds": max_execution_time
            }
        },
        "performance_summary": {
            "ranking_metrics": {
                "total_tasks_with_results": total_ranked_results,
                "avg_best_rank": total_rank_sum / total_ranked_results if total_ranked_results > 0 else None,
                "top3_success_rate": top3_count / len(results) if results else 0,
                "top10_success_rate": top10_count / len(results) if results else 0,
                "top3_count": top3_count,
                "top10_count": top10_count
            }
        },
        "results_summary": {
            "total_tasks": len(results),
            "total_urls_not_in_db": total_urls_not_in_db,
            "total_expected_urls": total_expected_urls,
            "total_searches": total_searches_performed
        },
        "error_summary": {
            "failed_tasks": failed_tasks,
            "successful_tasks": len(results) - failed_tasks,
            "recursion_errors": recursion_errors,
            "other_errors": other_errors,
            "success_rate": (len(results) - failed_tasks) / len(results) if results else 0
        },
        "results": results
    }

    results_file = results_output_dir / \
        f"benchmark_results_{current_timestamp}.json"

    # Save results
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Generate compact CSV metrics using external calculation function
    csv_data = []

    # Calculate metrics for each individual task
    for result in results:
        # Get benchmark solution (expected URLs) for this task
        benchmark_solution = result.get("correct_answers", [])

        # Get model solution (evaluation URLs) for this task
        model_solution = result.get("evaluation_urls", [])
        error_occurred = result.get("error_occurred", False)

        if not error_occurred and benchmark_solution and model_solution:
            metrics = calculation_results(benchmark_solution, model_solution)
        else:
            metrics_source = result.get("metrics", {})
            metrics = {
                "task_completion_rate": metrics_source.get("task_completion_rate", 0),
                "avg_precision": metrics_source.get("avg_precision", 0.0),
                "avg_recall": metrics_source.get("avg_recall", 0.0),
                "f1_score": metrics_source.get("f1_score", 0.0)
            }

        csv_data.append({
            "category": result.get("task_category", "Unknown"),
            "task_id": result.get("task_id", ""),
            "task_completion_rate": metrics["task_completion_rate"],
            "avg_precision": metrics["avg_precision"],
            "avg_recall": metrics["avg_recall"],
            "f1_score": metrics["f1_score"],
            "prompt_tokens": 0 if error_occurred else result.get("prompt_tokens", 0),
            "completion_tokens": 0 if error_occurred else result.get("completion_tokens", 0),
            "filter_prompt_tokens": result.get("filter_prompt_tokens", 0),
            "filter_completion_tokens": result.get("filter_completion_tokens", 0),
            "cache_prompt_tokens": result.get("cache_prompt_tokens", 0),
            "cache_completion_tokens": result.get("cache_completion_tokens", 0),
            "cache_hit": result.get("cache_hit", False),
            "execution_duration": result.get("execution_time_seconds", 0)
        })

    csv_file = results_output_dir / \
        f"benchmark_metrics_{current_timestamp}.csv"

    # Write CSV file
    with csv_file.open("w", newline="", encoding="utf-8") as csvfile:
        if csv_data:
            fieldnames = [
                "category",
                "task_id",
                "task_completion_rate",
                "avg_precision",
                "avg_recall",
                "f1_score",
                "prompt_tokens",
                "completion_tokens",
                "filter_prompt_tokens",
                "filter_completion_tokens",
                "cache_prompt_tokens",
                "cache_completion_tokens",
                "cache_hit",
                "execution_duration"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_data)

    print(f"CSV metrics saved to {csv_file}")

    # Print statistics
    print("\n" + "=" * 60)
    print("PERFORMANCE STATISTICS")
    print("=" * 60)
    print(f"Total searches performed: {total_searches_performed}")
    print(
        f"Average searches per task: {total_searches_performed/len(results):.1f}")
    print(f"  Total tool calls: {total_tool_calls}")
    print(f"Total cart operations: {total_cart_tools}")
    print(f"Total checkout operations: {total_checkout_tools}")
    print(f"  Average tools per task: {total_tool_calls/len(results):.1f}")

    # Error statistics
    success_rate = (len(results) - failed_tasks) / \
        len(results) if results else 0
    print(f"\nERROR STATISTICS:")
    print(f"  - Total tasks: {len(results)}")
    print(f"  - Successful tasks: {len(results) - failed_tasks}")
    print(f"  - Failed tasks: {failed_tasks}")
    print(f"  - Success rate: {success_rate:.1%}")
    if failed_tasks > 0:
        print(f"  - Recursion errors: {recursion_errors}")
        print(f"  - Other errors: {other_errors}")


    # Print execution time statistics
    print(f"\n  EXECUTION TIME METRICS:")
    print(f"  - Total: {total_execution_time:.2f} seconds")
    print(f"  - Average per task: {avg_execution_time:.2f} seconds")
    print(f"  - Min: {min_execution_time:.2f} seconds")
    print(f"  - Max: {max_execution_time:.2f} seconds")

    # Print overall performance metrics
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE METRICS")
    print("=" * 60)

    # Token usage and cost summary
    print("\n" + "=" * 60)
    print("TOKEN USAGE SUMMARY")
    print("=" * 60)
    print(f"Embedding Tokens: {total_tokens_used['embedding_tokens']:,}")
    print(f"Prompt Tokens: {total_tokens_used['prompt_tokens']:,}")
    print(f"Completion Tokens: {total_tokens_used['completion_tokens']:,}")
    print(f"Total Tokens Used: {total_tokens_used['total_tokens']:,}")


# SAIA API Configuration (GWDG HPC)
# Base URL for SAIA OpenAI-compatible API
SAIA_BASE_URL = "https://chat-ai.academiccloud.de/v1"

# Available SAIA models:
# - openai-gpt-oss-120b: Large GPT model
# - meta-llama-3.1-8b-instruct: Llama 3.1 8B
# - llama-3.3-70b-instruct: Llama 3.3 70B
# - mistral-large-instruct: Mistral Large
# - qwen3-32b: Qwen 3 32B
# - qwen2.5-coder-32b-instruct: Qwen 2.5 Coder
# See https://docs.hpc.gwdg.de/services/saia/ for full list


def create_saia_model(model_name: str = "openai-gpt-oss-120b", temperature: float = 0.0) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance configured for SAIA API.

    Args:
        model_name: SAIA model name (default: openai-gpt-oss-120b)
        temperature: Model temperature (default: 0.0)

    Returns:
        ChatOpenAI instance configured for SAIA

    Requires:
        GOAI_API_KEY environment variable to be set with your SAIA API key
    """
    api_key = os.getenv("GOAI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOAI_API_KEY environment variable not set. "
            "Get your API key from https://kisski.gwdg.de/ (KISSKI LLM Service)"
        )

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=SAIA_BASE_URL,
        temperature=temperature
    )


# Main execution
async def main():
    """Main function with proper cleanup"""
    try:
        # You can easily switch models here
        # Examples: "gpt-4", "gpt-3.5-turbo", "claude-3-opus-20240229", "claude-3-sonnet-20240229"

        # === SAIA Models (GWDG HPC) ===
        # Uncomment to use SAIA API with openai-gpt-oss-120b or other models
        # Requires GOAI_API_KEY environment variable

        # model_name = "codestral-22b"
        # chat_model = create_saia_model(model_name=model_name, temperature=0.0)

        # Other SAIA models you can try:
        # model_name = "llama-3.3-70b-instruct"
        # model_name = "mistral-large-instruct"
        # model_name = "qwen3-32b"

        # MODIFIED: model is env-driven (MAIN_MODEL / MAIN_REASONING_EFFORT)
        # so the PowerShell drivers can run all method permutations without
        # editing the source.
        model_name = os.getenv("MAIN_MODEL", "gpt-5-mini")
        reasoning = os.getenv("MAIN_REASONING_EFFORT")
        chat_model = create_chat_model(
            model_name,
            reasoning_effort=reasoning,
            temperature=0.0,
        )

        await process_benchmark(model_name=model_name, chat_model=chat_model)
    finally:
        # Clean up Elasticsearch client
        await es_client.close()


if __name__ == "__main__":
    asyncio.run(main())
