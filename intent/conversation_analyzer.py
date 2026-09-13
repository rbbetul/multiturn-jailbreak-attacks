"""Core TCA conversation analyzer: LLM analysis + risk scoring + CSV export."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from config.tca_config import TCAConfig
from llms.llm_call_logger import LLMCallLogger
from llms.an_llm_manager import LLMManager
from prompts.prompt_templates import PromptManager
from tca.risk_calculator import RiskCalculator
from tca.security_engine import SecurityDecisionEngine


class ConversationAnalyzer:
    """Multi-turn risk analyzer using an OpenAI-compatible LLM backend and TCA scoring."""

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        llm_type: str = "gpt",
        model_name: str | None = None,
        llm_provider: str | None = None,
        verbose: bool = True,
        llm_calls_log: Path | None = None,
        human_only: bool = False,
    ):
        load_dotenv(find_dotenv())
        self.config = TCAConfig(config_path)
        self.llm_type = llm_type
        self.verbose = verbose
        self.human_only = human_only
        self.llm_call_logger = (
            LLMCallLogger(llm_calls_log, verbose=verbose) if llm_calls_log else None
        )
        self.llm_manager = LLMManager(
            prompt_manager=PromptManager(),
            config_path=config_path,
            model_name=model_name,
            provider=llm_provider,
            call_logger=self.llm_call_logger,
        )
        self.risk_calculator = RiskCalculator()
        self.decision_engine = SecurityDecisionEngine()
        self.historical_risk = 0.0
        self.recorded_results: List[Dict[str, Any]] = []

    @property
    def active_model(self) -> str:
        return self.llm_manager.config.model_name

    def _log(self, *args: object) -> None:
        if self.verbose:
            print(*args)

    @staticmethod
    def _parse_llm_content(content: str) -> dict:
        cleaned = content.strip("`").replace("```json", "").replace("```", "").strip()
        if not cleaned:
            raise ValueError("Empty LLM response")

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            missing_brackets = cleaned.count("[") - cleaned.count("]")
            if missing_brackets > 0 and cleaned.endswith("}"):
                repaired = cleaned[:-1] + ("]" * missing_brackets) + "}"
                return json.loads(repaired)
            raise

    def _valid_pair(self, pair: Tuple[str, str]) -> bool:
        if not pair or len(pair) != 2:
            return False
        human_raw = pair[1]
        human = "" if human_raw is None else str(human_raw).strip()
        return bool(human)

    def _build_record(
        self,
        content_data: dict,
        row_idx: int,
        pair_idx: int,
        historical_before: float,
        pattern_risk: float,
        progressive_risk: float,
        decision: dict,
    ) -> Dict[str, Any]:
        progression = content_data.get("overall_progression_summary", {})
        patterns = content_data.get("patterns", {})
        recommendations = content_data.get("recommendations", [])
        intent_shift = content_data.get("intent_shift", {})
        prompt_attack = content_data.get("prompt_attack", {})

        record: Dict[str, Any] = {
            "row_index": row_idx,
            "transition_index": pair_idx,
            "interaction_risk": progression.get("final_risk_level", 0),
            "initial_risk_level": progression.get("initial_risk_level", ""),
            "risk_progression": progression.get("risk_progression", "N/A"),
            "pattern_risk": pattern_risk,
            "historical_risk_before": historical_before,
            "progressive_risk": progressive_risk,
            "historical_risk_after": progressive_risk,
            "decision": decision["decision"],
            "decision_message": decision["message"],
            "decision_reasoning": decision["reasoning"],
            "warn_threshold": self.config.risk.warn_threshold,
            "block_threshold": self.config.risk.block_threshold,
            "intent_shift_detected": intent_shift.get("shift_detected"),
            "intent_shift_risk_level": intent_shift.get("risk_level"),
            "prompt_attack_detected": prompt_attack.get("detected"),
            "recommendations": " | ".join(recommendations) if recommendations else "",
            "model": self.active_model,
        }
        for name, details in patterns.items():
            if isinstance(details, dict):
                record[f"pattern_{name}_detected"] = details.get("detected")
        return record

    def process_response(
        self,
        response: Dict[str, Any],
        row_idx: int,
        pair_idx: int,
    ) -> Dict[str, Any] | None:
        try:
            content_data = self._parse_llm_content(response.get("content", "{}"))
        except (ValueError, json.JSONDecodeError) as e:
            self._log(f"Error parsing LLM content (row {row_idx}, transition {pair_idx}): {e}")
            content_data = {}

        progression = content_data.get("overall_progression_summary", {})
        interaction_risk = progression.get("final_risk_level", 0)
        historical_before = self.historical_risk
        pattern_risk = self.risk_calculator.calculate_pattern_risk(
            content_data.get("patterns", {})
        )
        progressive_risk = self.risk_calculator.calculate_progressive_risk(
            historical_before, interaction_risk, pattern_risk
        )
        self.historical_risk = progressive_risk
        decision = self.decision_engine.evaluate_risk(progressive_risk, historical_before)

        self._log(
            f"Row {row_idx} transition {pair_idx}: "
            f"progressive_risk={progressive_risk:.2f} -> {decision['decision'].upper()}"
        )

        return self._build_record(
            content_data, row_idx, pair_idx,
            historical_before, pattern_risk, progressive_risk, decision,
        )

    async def analyze_conversation_row(
        self,
        row_index: int,
        llm_pairs: List[Tuple[str, str]],
        metadata: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        metadata = metadata or {}
        n_transitions = max(0, len(llm_pairs) - 1)
        self._log(
            f"Row {row_index}: {len(llm_pairs)} turns, {n_transitions} transitions "
            f"| model={self.active_model}"
        )

        if len(llm_pairs) < 2:
            self._log(f"Skipping row {row_index}: need at least 2 turns.")
            return []

        self.historical_risk = 0.0
        row_results: List[Dict[str, Any]] = []

        for t_idx in range(1, len(llm_pairs)):
            prev_pair, current_pair = llm_pairs[t_idx - 1], llm_pairs[t_idx]
            if not self._valid_pair(prev_pair) or not self._valid_pair(current_pair):
                self._log(f"Skipping invalid pair at row {row_index} transition {t_idx}")
                continue

            try:
                response = await self.llm_manager.analyze_conversation(
                    self.llm_type,
                    prev_pair,
                    current_pair,
                    call_context={
                        "row_index": row_index,
                        "transition_index": t_idx,
                        "llm_type": self.llm_type,
                    },
                )
                record = self.process_response(response, row_index, t_idx)
                if record:
                    record.update(metadata)
                    row_results.append(record)
                    self.recorded_results.append(record)
            except Exception as e:
                self._log(f"Error row {row_index} transition {t_idx}: {e}")

        return row_results

    def close(self) -> None:
        if self.llm_call_logger is not None:
            self.llm_call_logger.close()

    def save_results_csv(self, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.recorded_results).to_csv(output_path, index=False)
        print(f"Wrote {len(self.recorded_results)} transition records to {output_path}")
        return output_path


if __name__ == "__main__":
    from intent.cli import main

    main()
