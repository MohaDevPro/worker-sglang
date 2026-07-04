import subprocess
import time
import os
import sys


class SGlangEngine:
    def __init__(
        self,
        model=os.getenv("MODEL_NAME"),
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "30000")),
    ):
        self.model = model
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
        self.process = None

    def start_server(self):
        command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        # Dictionary of all possible options and their corresponding env var names
        options = {
            "MODEL_NAME": "--model-path",
            "TOKENIZER_PATH": "--tokenizer-path",
            "TOKENIZER_MODE": "--tokenizer-mode",
            "LOAD_FORMAT": "--load-format",
            "DTYPE": "--dtype",
            "CONTEXT_LENGTH": "--context-length",
            "QUANTIZATION": "--quantization",
            "SERVED_MODEL_NAME": "--served-model-name",
            "CHAT_TEMPLATE": "--chat-template",
            "MEM_FRACTION_STATIC": "--mem-fraction-static",
            "MAX_RUNNING_REQUESTS": "--max-running-requests",
            "MAX_TOTAL_TOKENS": "--max-total-tokens",
            "CHUNKED_PREFILL_SIZE": "--chunked-prefill-size",
            "MAX_PREFILL_TOKENS": "--max-prefill-tokens",
            "SCHEDULE_POLICY": "--schedule-policy",
            "SCHEDULE_CONSERVATIVENESS": "--schedule-conservativeness",
            "TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
            "STREAM_INTERVAL": "--stream-interval",
            "RANDOM_SEED": "--random-seed",
            "LOG_LEVEL": "--log-level",
            "LOG_LEVEL_HTTP": "--log-level-http",
            "API_KEY": "--api-key",
            "FILE_STORAGE_PATH": "--file-storage-path",
            "DATA_PARALLEL_SIZE": "--data-parallel-size",
            "LOAD_BALANCE_METHOD": "--load-balance-method",
            "ATTENTION_BACKEND": "--attention-backend",
            "SAMPLING_BACKEND": "--sampling-backend",
            "TOOL_CALL_PARSER": "--tool-call-parser",
            "REASONING_PARSER": "--reasoning-parser",
            # VLM-specific options (added for Qwen3-VL support)
            "MM_ATTENTION_BACKEND": "--mm-attention-backend",
        }

        # Boolean flags
        boolean_flags = [
            "SKIP_TOKENIZER_INIT",
            "TRUST_REMOTE_CODE",
            "LOG_REQUESTS",
            "SHOW_TIME_COST",
            "DISABLE_RADIX_CACHE",
            "DISABLE_CUDA_GRAPH",
            "DISABLE_OUTLINES_DISK_CACHE",
            "ENABLE_TORCH_COMPILE",
            "ENABLE_P2P_CHECK",
            "ENABLE_FLASHINFER_MLA",
            "TRITON_ATTENTION_REDUCE_IN_FP32",
            # VLM-specific boolean flags (added for Qwen3-VL support)
            "ENABLE_MULTIMODAL",
            "KEEP_MM_FEATURE_ON_DEVICE",
            "DISABLE_FAST_IMAGE_PROCESSOR",
        ]

        # Add options from environment variables only if they are set
        for env_var, option in options.items():
            value = os.getenv(env_var)
            if value is not None and value != "":
                command.extend([option, value])

        # Add boolean flags only if they are set to true
        for flag in boolean_flags:
            if os.getenv(flag, "").lower() in ("true", "1", "yes"):
                command.append(f"--{flag.lower().replace('_', '-')}")

        # Log the full command for debugging
        print(f"Starting sglang server with command: {' '.join(command)}", flush=True)

        # Capture stdout/stderr so we can see server logs and detect crashes
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(f"Server started with PID: {self.process.pid}", flush=True)

    def wait_for_server(self, timeout=900, interval=5):
        """
        Wait for the sglang server to be ready.
        Monitors the subprocess for crashes while waiting,
        and logs server output for debugging.
        """
        import urllib.request
        import urllib.error

        start_time = time.time()
        last_output_time = start_time

        while time.time() - start_time < timeout:
            # Check if the server process has crashed
            poll = self.process.poll()
            if poll is not None:
                # Process has exited - read any remaining output
                remaining = self.process.stdout.read().decode("utf-8", errors="replace")
                print(f"Server process exited with code {poll}", flush=True)
                if remaining:
                    print(f"Server output:\n{remaining}", flush=True)
                raise RuntimeError(
                    f"Server process crashed with exit code {poll}. "
                    f"Check the logs above for details."
                )

            # Read and print server output for debugging (non-blocking)
            try:
                import selectors
                sel = selectors.DefaultSelector()
                sel.register(self.process.stdout, selectors.EVENT_READ)
                ready = sel.select(timeout=0.1)
                for key, _ in ready:
                    line = key.fileobj.readline()
                    if line:
                        decoded = line.decode("utf-8", errors="replace").rstrip()
                        print(f"[sglang] {decoded}", flush=True)
                        last_output_time = time.time()
                sel.close()
            except Exception:
                pass

            # Check if server is ready
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/v1/models",
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        print("Server is ready!", flush=True)
                        return True
            except (urllib.error.URLError, Exception):
                pass

            time.sleep(interval)

        raise TimeoutError(
            f"Server failed to start within {timeout} seconds. "
            f"Last server output was {time.time() - last_output_time:.0f}s ago."
        )

    def warmup(self):
        """
        Send a lightweight warmup request to prime CUDA graphs and KV cache.
        This avoids the first real user request being significantly slower.
        Uses urllib to avoid pulling in requests/aiohttp at module level.
        """
        import urllib.request
        import urllib.error
        import json

        print("[engine] Sending warmup request...", flush=True)
        warmup_payload = json.dumps({
            "model": self.model or "default",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "stream": False,
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=warmup_payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status == 200:
                    print("[engine] Warmup successful!", flush=True)
                else:
                    print(
                        f"[engine] Warmup returned status {resp.status}, "
                        "continuing anyway.",
                        flush=True,
                    )
        except Exception as e:
            # Warmup failure is non-fatal; the server is still running.
            print(f"[engine] Warmup failed (non-fatal): {e}", flush=True)

    def shutdown(self):
        """Gracefully terminate the sglang server process."""
        if self.process and self.process.poll() is None:
            print("[engine] Sending SIGTERM to sglang server...", flush=True)
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
                print("[engine] Server exited gracefully.", flush=True)
            except subprocess.TimeoutExpired:
                print("[engine] Server did not exit in 30s, sending SIGKILL...", flush=True)
                self.process.kill()
                self.process.wait()
                print("[engine] Server killed.", flush=True)
        else:
            print("[engine] Server process already stopped.", flush=True)