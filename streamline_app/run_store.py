from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


DEFAULT_RUN_ROOT = Path("runs") / "generated"


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    prompt_path: Path
    retrieval_path: Path
    metadata_path: Path
    llm_response_path: Path
    code_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "prompt_path": str(self.prompt_path),
            "retrieval_path": str(self.retrieval_path),
            "metadata_path": str(self.metadata_path),
            "llm_response_path": str(self.llm_response_path),
            "code_path": str(self.code_path),
        }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_run_artifacts(
    *,
    prompt: str,
    retrieval_payload: Mapping[str, Any],
    dataset_metadata: Mapping[str, Any],
    user_request: Mapping[str, Any],
    llm_response: str,
    code: str,
    run_root: str | Path = DEFAULT_RUN_ROOT,
) -> RunArtifacts:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(run_root) / f"{timestamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = run_dir / "final_prompt.md"
    retrieval_path = run_dir / "retrieval.json"
    metadata_path = run_dir / "dataset_metadata.json"
    llm_response_path = run_dir / "llm_response.md"
    code_path = run_dir / "generated_visualization.py"

    prompt_path.write_text(prompt, encoding="utf-8")
    _write_json(
        retrieval_path,
        {
            "user_request": user_request,
            "retrieval": retrieval_payload,
        },
    )
    _write_json(metadata_path, dataset_metadata)
    llm_response_path.write_text(llm_response, encoding="utf-8")
    code_path.write_text(code, encoding="utf-8")

    return RunArtifacts(
        run_dir=run_dir,
        prompt_path=prompt_path,
        retrieval_path=retrieval_path,
        metadata_path=metadata_path,
        llm_response_path=llm_response_path,
        code_path=code_path,
    )

