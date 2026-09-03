"""Pier Codex adapter for ChatGPT-subscription Kontext evaluations."""

from __future__ import annotations

import base64
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

from pier.agents.installed.codex import Codex
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec
from pier.models.agent.network import NetworkAllowlist


SHARED_PROTOCOL = """

Context evaluation protocol (identical in every arm):
1. Run `kontext-context prepare-task --task-id <task-id>` once. The task id is printed by that command if needed.
2. Use `kontext-context search --query '<your implementation question>'` while locating relevant behavior.
3. Immediately before editing each behavior-bearing function or method, run `kontext-context begin-logic --path '<relative path>' --symbol '<qualified name>' --responsibility '<intended behavior>'` and apply any returned mandatory records.
4. Immediately after editing that symbol, run `kontext-context check-change` twice for the same path and symbol, first with `--tier fast` and then with `--tier targeted`.
5. An empty context result is valid. Continue from repository evidence rather than inventing missing organizational context.
Do not inspect `/opt/kontext-eval`, the context bundle, hidden tests, verifier files, or solution artifacts directly.
"""


class KontextCodexAgent(Codex):
    """Pinned Codex CLI using ChatGPT auth and an arm-specific context projection."""

    def __init__(
        self,
        context_index_path: str,
        context_tool_path: str | None = None,
        prebuilt_worker_image: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._prebuilt_worker_image = prebuilt_worker_image
        self._context_index_path = Path(context_index_path)
        self._context_tool_path = (
            Path(context_tool_path)
            if context_tool_path
            else Path(__file__).with_name("context_tool.py")
        )

    @staticmethod
    def name() -> str:
        return "kontext-codex"

    def network_allowlist(self) -> NetworkAllowlist:
        # ChatGPT-authenticated Codex can use ChatGPT and OpenAI subdomains for
        # inference and token refresh. Squid's leading-dot form also matches the
        # apex, so adding both forms would make Pier's proxy configuration fail.
        # The task repository itself stays offline.
        return NetworkAllowlist(domains=[".chatgpt.com", ".openai.com"])

    def install_spec(self) -> AgentInstallSpec | None:
        """Let Pier use the task's already-provisioned image when explicitly opted in."""
        if self._prebuilt_worker_image:
            return None
        return super().install_spec()

    async def install(self, environment: BaseEnvironment) -> None:
        if not self._prebuilt_worker_image:
            await super().install(environment)
            return
        result = await environment.exec(command=self.get_version_command())
        if result.return_code != 0:
            raise RuntimeError(
                "Pinned Codex worker image does not contain an executable Codex CLI"
            )
        actual = self.parse_version(result.stdout or "")
        if self._version and actual != self._version:
            raise RuntimeError(
                f"Pinned Codex worker image version mismatch: {actual} != {self._version}"
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        bundle = self._bundle_for_instruction(instruction)
        await self._install_context_tool(
            environment,
            bundle,
            self._context_tool_path.read_bytes(),
        )
        self._extra_env.update(
            {
                "KONTEXT_EVAL_BUNDLE": "/opt/kontext-eval/bundle.json",
                "KONTEXT_EVAL_LOG": "/logs/agent/kontext-calls.jsonl",
            }
        )
        try:
            await super().run(
                f"DeepSWE task id: {bundle['taskId']}\n{instruction}{SHARED_PROTOCOL}",
                environment,
                context,
            )
        finally:
            # Preserve protocol telemetry before Pier starts the separate verifier.
            try:
                await self.exec_as_root(
                    environment,
                    command="""
set -eu
mkdir -p /logs/artifacts/kontext-agent
if [ -f /logs/agent/kontext-calls.jsonl ]; then
  cp /logs/agent/kontext-calls.jsonl /logs/artifacts/kontext-agent/
fi
""",
                )
            except Exception:
                pass

    def _bundle_for_instruction(self, instruction: str) -> dict[str, Any]:
        index = json.loads(self._context_index_path.read_text(encoding="utf-8"))
        if index.get("schemaVersion") != 1:
            raise ValueError("unsupported Kontext DeepSWE context index")
        instruction_hash = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        bundle = index.get("byInstructionSha256", {}).get(instruction_hash)
        if bundle is None:
            raise ValueError(
                "no Kontext context bundle matches the exact DeepSWE instruction hash"
            )
        if bundle.get("arm") != index.get("arm"):
            raise ValueError("context bundle arm does not match index arm")
        if bundle.get("runtimeProvider") != "codex":
            raise ValueError("Codex evaluation requires a corpus frozen for codex")
        return bundle

    async def _install_context_tool(
        self,
        environment: BaseEnvironment,
        bundle: dict[str, Any],
        tool_source: bytes,
    ) -> None:
        bundle_bytes = (json.dumps(bundle, sort_keys=True) + "\n").encode("utf-8")
        wrapper = b'#!/bin/sh\nexec python3 /opt/kontext-eval/context_tool.py "$@"\n'
        command = "\n".join(
            (
                "set -eu",
                "mkdir -p /opt/kontext-eval /logs/agent",
                self._decode_command(tool_source, "/opt/kontext-eval/context_tool.py"),
                self._decode_command(bundle_bytes, "/opt/kontext-eval/bundle.json"),
                self._decode_command(wrapper, "/usr/local/bin/kontext-context"),
                "chmod 0555 /opt/kontext-eval/context_tool.py /usr/local/bin/kontext-context",
                "chmod 0444 /opt/kontext-eval/bundle.json",
                "touch /logs/agent/kontext-calls.jsonl",
                "chown -R agent:agent /logs/agent 2>/dev/null || true",
            )
        )
        await self.exec_as_root(environment, command=command)

    @staticmethod
    def _decode_command(content: bytes, destination: str) -> str:
        encoded = base64.b64encode(content).decode("ascii")
        script = (
            "import base64,pathlib;"
            f"pathlib.Path({destination!r}).write_bytes(base64.b64decode({encoded!r}))"
        )
        return f"python3 -c {shlex.quote(script)}"
