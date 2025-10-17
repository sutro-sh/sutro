"""
GEPA Prompt Optimizer for Sutro Batch Inference

Example: Optimize patent analysis prompt from GPT-4o to Qwen-3-4B
"""

import json
import re
from typing import Any, TypedDict

import litellm
import polars as pl

import gepa
from gepa.core.adapter import EvaluationBatch, GEPAAdapter


# ============================================================================
# Data Types
# ============================================================================

class DataInst(TypedDict):
    input: str
    golden_output: str


class RolloutOutput(TypedDict):
    generated: str
    score: float


# ============================================================================
# Step 1: Create Golden Dataset
# ============================================================================

def create_golden_dataset_from_parquet(
    parquet_path: str,
    input_column: str,
    n_samples: int,
    golden_model: str,
    system_prompt: str,
    output_file: str = "golden_examples.jsonl",
    seed: int = 42,
) -> list[DataInst]:
    """
    Create golden dataset by sampling from parquet and running through strong model.
    """
    print("Creating golden dataset:")
    print(f"  Loading from: {parquet_path}")
    print(f"  Sampling {n_samples} rows")
    print(f"  Using {golden_model} for golden outputs")
    print()

    # Load and sample
    df = pl.read_parquet(parquet_path)
    sampled_df = df.sample(n=min(n_samples, len(df)), seed=seed)

    # Prepare messages
    messages_batch = []
    for row in sampled_df.iter_rows(named=True):
        input_text = str(row[input_column])
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        messages_batch.append((input_text, msgs))

    # Generate golden outputs
    print("Generating golden outputs...")
    examples = []

    responses = litellm.batch_completion(
        model=golden_model,
        messages=[msgs for _, msgs in messages_batch],
        max_workers=10,
    )

    print('Got all responses back')

    for (input_text, _), response in zip(messages_batch, responses):
        golden_output = response.choices[0].message.content
        examples.append({
            "input": input_text,
            "golden_output": golden_output,
        })

    # Save to JSONL
    with open(output_file, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

    print(f"✓ Created {len(examples)} examples")
    print(f"✓ Saved to {output_file}\n")

    return examples


# ============================================================================
# Step 2: LLM Judge (3-point scale)
# ============================================================================

class LLMJudge:
    def __init__(self, judge_model: str, golden_model_name: str ):
        self.judge_model = judge_model
        self.golden_model_name = golden_model_name

    def score(self, input_text: str, golden: str, generated: str) -> tuple[float, str]:
        """Score generated output. Returns (score, reasoning)."""
        prompt = f"""You are evaluating a generated response against a reference response.

**Input:** {input_text[:500]}...

**Reference (from {self.golden_model_name}):** {golden}

**Generated:** {generated}

**Scoring:**
CORRECT (1.0): Contains all key information, no significant errors
PARTIAL (0.5): Some correct info but missing key elements or minor errors
WRONG (0.0): Major errors, missing most information, or unusable

Format: SCORE: [0.0, 0.5, or 1.0]
REASON: [one sentence]

SCORE:"""

        try:
            import litellm
            response = litellm.completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            ).choices[0].message.content
        except Exception as e:
            return 0.0, f"Judge failed: {e}"

        # Parse score
        score_match = re.search(r'SCORE:\s*(0\.0|0\.5|1\.0)', response)
        score = float(score_match.group(1)) if score_match else 0.0

        # Parse reasoning
        reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        reasoning = reason_match.group(1).strip() if reason_match else response[:200]

        return score, reasoning

    def generate_feedback(self, input_text: str, golden: str, generated: str, score: float) -> str:
        """Generate actionable feedback for GEPA."""
        if score == 1.0:
            return "CORRECT: Response matches reference quality."

        prompt = f"""The generated response scored {score} on a discrete scale [0.0,0.5,1.0].

**Input:** {input_text[:500]}...
**Reference:** {golden}
**Generated:** {generated}

Provide 2-3 specific points on what needs to improve.

Issues:"""



        feedback = (
            litellm.completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            .choices[0]
            .message.content
        )

        prefix = "PARTIAL: " if score == 0.5 else "WRONG: "
        return prefix + feedback


# ============================================================================
# Step 3: GEPA Adapter
# ============================================================================

class SutroPromptAdapter(GEPAAdapter[DataInst, dict, RolloutOutput]):
    def __init__(
        self,
        target_model: str,
        golden_model_name: str,
        judge_model: str ,
        max_workers: int = 10,
    ):
        self.target_model = target_model
        self.judge = LLMJudge(judge_model, golden_model_name)
        self.max_workers = max_workers

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict, RolloutOutput]:
        system_prompt = candidate.get("system_prompt", "")

        # Build messages
        messages_batch = []
        for data in batch:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data["input"]}
            ]
            messages_batch.append(msgs)

        # Execute target model
        try:
            import litellm
            responses = [
                resp.choices[0].message.content
                for resp in litellm.batch_completion(
                    model=self.target_model,
                    messages=messages_batch,
                    max_workers=self.max_workers,
                )
            ]
        except Exception:
            return EvaluationBatch(
                outputs=[{"generated": "", "score": 0.0}] * len(batch),
                scores=[0.0] * len(batch),
                trajectories=None,
            )

        # Score each response
        outputs = []
        scores = []
        traces = [] if capture_traces else None

        for data, response in zip(batch, responses):
            score, reasoning = self.judge.score(data["input"], data["golden_output"], response)

            outputs.append({"generated": response, "score": score})
            scores.append(score)

            if capture_traces:
                traces.append({
                    "input": data["input"],
                    "golden": data["golden_output"],
                    "generated": response,
                    "score": score,
                    "reasoning": reasoning,
                })

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traces)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict, RolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        items = []

        for trace in eval_batch.trajectories:
            feedback = self.judge.generate_feedback(
                trace["input"],
                trace["golden"],
                trace["generated"],
                trace["score"],
            )

            items.append({
                "Input": trace["input"][:500] + "...",  # Truncate for readability
                "Expected Output": trace["golden"],
                "Generated Output": trace["generated"],
                "Score": f"{trace['score']}/1.0",
                "Feedback": feedback,
            })

        return {components_to_update[0]: items}


# ============================================================================
# Step 4: Optimize
# ============================================================================

def optimize_prompt(
    examples: list[DataInst],
    seed_prompt: str,
    target_model: str,
    golden_model_name: str ,
    judge_model: str ,
    reflection_lm: str ,
    max_metric_calls: int = 150,
    output_dir: str = "./gepa_output",
) -> dict:
    """Run GEPA optimization."""

    # Split train/val
    split_idx = int(len(examples) * 0.8)
    trainset = examples[:split_idx]
    valset = examples[split_idx:]

    print("="*80)
    print("OPTIMIZING PROMPT")
    print("="*80)
    print(f"Target model: {target_model}")
    print(f"Golden model: {golden_model_name}")
    print(f"Train: {len(trainset)}, Val: {len(valset)}")
    print(f"Budget: {max_metric_calls} evaluations")
    print()

    # Create adapter
    adapter = SutroPromptAdapter(
        target_model=target_model,
        golden_model_name=golden_model_name,
        judge_model=judge_model,
    )

    # Run GEPA
    opt_result = gepa.optimize(
        seed_candidate={"system_prompt": seed_prompt},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=3,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        perfect_score=1.0,
        run_dir=output_dir,
        display_progress_bar=True,
    )

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"Baseline:   {opt_result.program_scores[0]:.2%}")
    print(f"Optimized:  {opt_result.best_score:.2%}")
    print(f"Improvement: +{(opt_result.best_score - opt_result.program_scores[0]):.2%}")
    print()
    print("OPTIMIZED PROMPT:")
    print("-"*80)
    print(opt_result.best_candidate["system_prompt"])
    print("-"*80)

    return {
        "optimized_prompt": opt_result.best_candidate["system_prompt"],
        "best_score": opt_result.best_score,
        "baseline_score": opt_result.program_scores[0],
        "improvement": opt_result.best_score - opt_result.program_scores[0],
    }


# ============================================================================
# Example: Patent Analysis
# ============================================================================

if __name__ == "__main__":
    # Your patent analysis prompt
    SYSTEM_PROMPT = """You are a patent analyst. You will be given a patent filing.

Generate the following:
1.  **Title**: The main title of the patent.
2.  **Short Summary**: A brief summary of the patent.
3.  **Keywords**: A list of relevant keywords.
4.  **Key Quirks**: Identify a few notable or unusual aspects (`SimplifiedQuirk`). For each, provide an ID, description, and an optional humor rating (0.0-1.0).
5.  **Core Interpretations**: Provide some core interpretations (`SimplifiedInterpretation`) of the patent's claims or purpose. Include an ID, the interpretation text, and a confidence score (0.0-1.0).
6.  **Suggested Application Areas**: List potential application areas.
7.  **Overall Novelty Score**: Rate the patent's novelty on a scale of 1 to 10.

Please output each criteria on a separate line."""

    # Configuration
    PARQUET_PATH = "processed_patents_filtered/batch_001.parquet"
    INPUT_COLUMN = "description_localized"
    N_SAMPLES = 50
    FRONTIER_MODEL = "gemini/gemini-2.5-flash"
    TARGET_MODEL = "openrouter/qwen/qwen3-30b-a3b-thinking-2507"
    max_metric_calls = 50

    print("="*80)
    print("GEPA OPTIMIZATION: PATENT ANALYSIS")
    print("="*80)
    print(f"Goal: Optimize prompt from {FRONTIER_MODEL} → {TARGET_MODEL}")
    print()

    # Step 1: Create golden dataset
    examples = create_golden_dataset_from_parquet(
        parquet_path=PARQUET_PATH,
        input_column=INPUT_COLUMN,
        n_samples=N_SAMPLES,
        golden_model="gemini/gemini-2.5-pro",
        system_prompt=SYSTEM_PROMPT,
        output_file="patent_golden_examples.jsonl",
    )

    # Step 2: Optimize prompt
    result = optimize_prompt(
        examples=examples,
        seed_prompt=SYSTEM_PROMPT,
        target_model=TARGET_MODEL,
        golden_model_name=FRONTIER_MODEL,
        judge_model=FRONTIER_MODEL,
        reflection_lm=FRONTIER_MODEL,
        max_metric_calls=max_metric_calls,
        output_dir="./patent_optimization",
    )

    # Step 3: Save results
    with open("patent_optimization/final_result.json", 'w') as f:
        json.dump(result, f, indent=2)

    print("\n✓ Results saved to patent_optimization/final_result.json")
    print(f"\n🎯 Next: Test optimized prompt in Sutro with {TARGET_MODEL}")