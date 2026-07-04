import asyncio
import signal
import os
from contextlib import asynccontextmanager

import aiohttp
from engine import SGlangEngine
from utils import process_response
import runpod

# Initialize the engine
engine = SGlangEngine()

# Global aiohttp session (connection pool + reuse)
_session: aiohttp.ClientSession = None

# Request timeout: connect=5s, total=300s (generous for VLM generation)
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=int(os.getenv("REQUEST_TIMEOUT", "300")), connect=5)


@asynccontextmanager
async def get_session():
    """Yield the shared aiohttp session, creating it on first use."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            timeout=REQUEST_TIMEOUT,
            connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=30),
        )
    try:
        yield _session
    except Exception:
        # Session-level errors are handled per-request below
        raise


def validate_job_input(job_input):
    """Basic request validation to catch malformed payloads early."""
    if not isinstance(job_input, dict):
        return False, "Input must be a JSON object"
    return True, None


async def post_with_retry(session, url, headers, json_data, max_retries=1):
    """
    Send POST request with timeout and basic retry on connection errors.
    Only retries on connection failures (not on server errors like 4xx/5xx).
    """
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            async with session.post(url, headers=headers, json=json_data) as resp:
                return resp
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_error = e
            if attempt < max_retries:
                print(
                    f"[handler] Request attempt {attempt + 1} failed: {e}. Retrying...",
                    flush=True,
                )
                await asyncio.sleep(1)
            else:
                print(
                    f"[handler] Request failed after {max_retries + 1} attempts: {e}",
                    flush=True,
                )
                raise
    raise last_error


async def async_handler(job):
    """Handle requests asynchronously using aiohttp (non-blocking)."""
    job_input = job["input"]

    # Basic request validation
    valid, err = validate_job_input(job_input)
    if not valid:
        yield {"error": err}
        return

    async with get_session() as session:
        # Case 1: full OpenAI style payload where caller specifies the route.
        if job_input.get("openai_route"):
            openai_route = job_input.get("openai_route")
            openai_input = job_input.get("openai_input")
            openai_url = f"{engine.base_url}" + openai_route
            headers = {"Content-Type": "application/json"}

            resp = await post_with_retry(session, openai_url, headers, openai_input)

            if openai_input.get("stream", False):
                # Stream: read SSE lines from async response
                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()
                    if decoded:
                        from utils import format_chunk
                        yield format_chunk(decoded)
            else:
                # Non-streaming: parse single JSON response
                yield await resp.json()

        # Case 2: payload looks like OpenAI chat/completions but omits the wrapper.
        elif "messages" in job_input:
            openai_url = f"{engine.base_url}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}

            if "model" not in job_input:
                job_input["model"] = engine.model or "default"

            resp = await post_with_retry(session, openai_url, headers, job_input)

            if job_input.get("stream", False):
                async for line in resp.content:
                    decoded = line.decode("utf-8").strip()
                    if decoded:
                        from utils import format_chunk
                        yield format_chunk(decoded)
            else:
                yield await resp.json()

        # Case 3: assume user meant the native /generate endpoint.
        else:
            generate_url = f"{engine.base_url}/generate"
            headers = {"Content-Type": "application/json"}
            resp = await post_with_retry(session, generate_url, headers, job_input)

            if resp.status == 200:
                yield await resp.json()
            else:
                error_text = await resp.text()
                yield {
                    "error": f"Generate request failed with status code {resp.status}",
                    "details": error_text,
                }


def cleanup():
    """Graceful shutdown: close aiohttp session and stop sglang server."""
    global _session
    print("[handler] Shutting down...", flush=True)
    if _session and not _session.closed:
        asyncio.get_event_loop().run_until_complete(_session.close())
    engine.shutdown()
    print("[handler] Shutdown complete.", flush=True)


# Start the server and wait for it to be ready
engine.start_server()
engine.wait_for_server()

# Warmup: send a lightweight request to prime CUDA graphs and KV cache
engine.warmup()

# Register signal handlers for graceful shutdown
for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, lambda s, f: cleanup())

runpod.serverless.start(
    {
        "handler": async_handler,
        "concurrency_modifier": lambda: int(os.getenv("MAX_CONCURRENCY", "10")),
        "return_aggregate_stream": True,
    }
)