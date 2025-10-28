"""
GEPA Prompt Optimizer for Sutro Batch Inference

Example: Optimize patent analysis prompt from GPT-4o to Qwen-3-4B
"""

import json
import re
import time
from datetime import timedelta
from typing import Any, TypedDict, Tuple

import litellm
import polars as pl

import gepa
from datasets import load_dataset
from dotenv import load_dotenv
from gepa import GEPAResult, TimeoutStopCondition
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
    use_existing: bool = True,
) -> list[DataInst]:
    """
    Create golden dataset by sampling from parquet and running through strong model.

    Args:
        use_existing: If True and output_file exists, load from file instead of regenerating
    """
    import os

    # Check if we can use existing file
    if use_existing and os.path.exists(output_file):
        print(f"✓ Found existing golden dataset: {output_file}")
        print(f"  Loading {output_file} (use_existing=True)")

        examples = []
        with open(output_file, 'r') as f:
            for line in f:
                examples.append(json.loads(line))

        print(f"✓ Loaded {len(examples)} examples from file\n")
        return examples

    # Generate new dataset
    print("Creating golden dataset:")
    print(f"  Loading from: {parquet_path}")
    print(f"  Sampling {n_samples} rows")
    print(f"  Using {golden_model} for golden outputs")
    print()

    # Load and sample
    df = pl.read_parquet(parquet_path)
    sampled_df = df.sample(n=min(n_samples, len(df)), seed=seed)
    print(len(df))

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
        max_workers=15,
    )

    print('Got all responses back')

    for (input_text, _), response in zip(messages_batch, responses):
        if isinstance(response, litellm.APIError):
            print("ERROR", response)
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


class LLMJudge:
    def __init__(self, judge_model: str, golden_model_name: str ):
        self.judge_model = judge_model
        self.golden_model_name = golden_model_name

    def score_batch(
        self, system_prompt: str, inputs: list[str], goldens: list[str], generateds: list[str]
    ) -> list[Tuple[float, str]]:
        """Score multiple outputs in parallel using batch_completion."""
        # Create prompts for all items
        messages_list = [
            [
                {
                    "role": "user",
                    "content": f"""You are evaluating a generated response against a reference response. The generated response was generated after being given the task instructions and the given inmput.
                    
**Task Instructions:** {system_prompt}

**Input:** {inp}

**Reference:** {gold}

**Generated:** {gen}

**Scoring:**
CORRECT (1.0): Contains all key information, no significant errors, closely matches reference response
PARTIAL (0.5): Some correct info but missing key elements or minor errors, generally doesn't match reference response
WRONG (0.0): Major errors, missing most information, or unusable

Format: SCORE: [0.0, 0.5, or 1.0]
REASON: [one sentence]

SCORE:""",
                }
            ]
            for inp, gold, gen in zip(inputs, goldens, generateds)
        ]

        # Batch call to LiteLLM
        responses = litellm.batch_completion(
            model=self.judge_model,
            messages=messages_list,
            temperature=0,
            max_workers=10
        )

        # Parse all responses
        results = []
        for response in responses:
            if isinstance(response, litellm.APIError):
                print("ERROR", response)
            content = response.choices[0].message.content

            # Parse score
            score_match = re.search(r"SCORE:\s*(0\.0|0\.5|1\.0)", content)
            score = float(score_match.group(1)) if score_match else 0.0

            # Parse reasoning
            reason_match = re.search(
                r"REASON:\s*(.+?)(?:\n|$)", content, re.IGNORECASE
            )
            reasoning = (
                reason_match.group(1).strip() if reason_match else content[:200]
            )

            results.append((score, reasoning))

        return results

    def generate_feedback(self, system_prompt: str, input_text: str, golden: str, generated: str, score: float) -> str:
        """Generate actionable feedback for GEPA."""
        if score == 1.0:
            return "CORRECT: Response matches reference quality."

        # TODO(cooper) it would be good to make this prompt dynamically aware of the
        #  size/strength of the target model
        prompt = f"""The generated response scored {score} on a discrete scale [0.0,0.5,1.0].
        
**Task Instructions:** {system_prompt}

**Input:** {input_text}

**Reference:** {golden}

**Generated:** {generated}

Provide 2-3 specific points that would help drive improvement on future iterations. These points will be taken as feedback to improve the instructions given to future assistants.

Keep in mind that this feedback may be incorporated into a prompt for a weaker assistant than yourself, thus it can get overwhelmed by mega prompts with excruciating detail.

Issues and improvements:"""

        feedback = (
            litellm.completion(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            .choices[0]
            .message.content
        )

        prefix = "PARTIAL: " if score == 0.5 else "WRONG: "
        return prefix + feedback


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
        responses = [
            # TODO need to pass in default or user provider sampling params here
            resp.choices[0].message.content
            for resp in litellm.batch_completion(
                model=self.target_model,
                messages=messages_batch,
                max_workers=self.max_workers,
            )
        ]

        # Score each response
        outputs = []
        scores = []
        traces = [] if capture_traces else None

        inputs = [data["input"] for data in batch]
        goldens = [data["golden_output"] for data in batch]
        generateds = responses
        batch_results = self.judge.score_batch(system_prompt, inputs, goldens, generateds)

        # Process results
        for data, response, (score, reasoning) in zip(batch, responses, batch_results):
            outputs.append({"generated": response, "score": score})
            scores.append(score)

            if capture_traces:
                traces.append(
                    {
                        "input": data["input"],
                        "golden": data["golden_output"],
                        "generated": response,
                        "score": score,
                        "reasoning": reasoning,
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=traces)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict, RolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        items = []

        system_prompt = next(iter(candidate.values()))

        for trace in eval_batch.trajectories:
            feedback = self.judge.generate_feedback(
                system_prompt,
                trace["input"],
                trace["golden"],
                trace["generated"],
                trace["score"],
            )

            it = {
                "Input": trace["input"],
                "Expected Output": trace["golden"],
                "Generated Output": trace["generated"],
                "Score": f"{trace['score']}/1.0",
                "Feedback": feedback,
            }

            items.append(it)

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
    max_metric_calls: int = None,
    max_runtime_minutes: int | None = None,
    output_dir: str = "./gepa_output",
) -> dict:
    """
    Run GEPA optimization.

    Speed optimization options:
    - max_metric_calls: Lower = faster (50 = ~30min, 150 = ~2hrs, 300 = ~4hrs)
    - max_runtime_minutes: Hard timeout if available (requires gepa.utils.TimeoutStopper)
    - Reduce n_samples in golden dataset creation
    - Use smaller val set (automatically 20% of examples)

    The main speed knob is max_metric_calls (budget). Lower budget = faster but less optimization.
    """

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
    if max_runtime_minutes:
        print(f"Max runtime: {max_runtime_minutes} minutes")
    print(f"\n💡 Tip: Create '{output_dir}/gepa.stop' file to gracefully stop early")
    print(f"💡 Tip: If interrupted, will resume from {output_dir}\n")

    # Create adapter
    adapter = SutroPromptAdapter(
        target_model=target_model,
        golden_model_name=golden_model_name,
        judge_model=judge_model,
    )

    # Setup stop conditions (if timeout specified)
    stop_callbacks = None
    if max_runtime_minutes:
        timeout_stopper = TimeoutStopCondition(max_runtime_minutes * 60)  # noqa: F821
        stop_callbacks = [timeout_stopper]
        print(f"⏱️  Timeout enabled: will stop after {max_runtime_minutes} minutes")


    # Run GEPA
    # Note: If interrupted, GEPA will resume from run_dir automatically
    start_time = time.time()

    opt_result: GEPAResult = gepa.optimize(
        seed_candidate={"system_prompt": seed_prompt},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        reflection_minibatch_size=5,
        candidate_selection_strategy="pareto",
        skip_perfect_score=True,
        perfect_score=1.0,
        run_dir=output_dir,
        display_progress_bar=True,
        stop_callbacks=stop_callbacks,
        use_wandb=True,
        wandb_init_kwargs={
            "entity": "coopslarhette-n-a",
            "project":"my-awesome-project",
        }
    )

    # Calculate duration
    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60

    # Display in readable format
    print(f"Optimization completed in: {timedelta(minutes=duration_minutes)}")
    print(f"Optimization completed in: {duration_minutes:.2f} minutes")

    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print()
    print("OPTIMIZED PROMPT:")
    print("-"*80)
    print(opt_result.best_candidate["system_prompt"])
    print(opt_result.val_aggregate_scores[opt_result.best_idx])
    print("-"*80)

    return {
        "optimized_prompt": opt_result.best_candidate["system_prompt"],
    }


# ============================================================================
# Example: Patent Analysis
# ============================================================================

if __name__ == "__main__":

    # Your patent analysis prompt

    SYSTEM_PROMPT = """Classify the following patent abstract as one of the following classes:
CLASSIFICATION CLASSES:
0: Human Necessities - food, agriculture, medicine, healthcare, personal care
1: Performing Operations; Transporting - manufacturing, industrial processes, transportation
2: Chemistry; Metallurgy - chemical processes, materials science, metallurgy, polymers
3: Textiles; Paper - textile production, paper making, fiber processing
4: Fixed Constructions - buildings, roads, bridges, construction, civil engineering
5: Mechanical Engineering; Lightning; Heating; Weapons; Blasting - mechanical systems, engines
6: Physics - measurement instruments, optics, nuclear physics, scientific analysis
7: Electricity - electrical circuits, power generation, electronics, telecommunications  
8: General tagging of new or cross-sectional technology - AI, nanotechnology, interdisciplinary

Abstract:
"""
#     SYSTEM_PROMPT = """
#
# You are a precise document classifier.
#
# # Objective
# Your task is to classify the provided text into one of the labels below.
#
# # Guidelines
# - Classify the text based on its primary topic. The classification should reflect what the document is predominantly about. If the majority of the document discusses topics unrelated to the available labels, but only a small portion mentions content that would fit one of the defined labels, classify the document as OTHER.
#
# # Labels
# CYBERSECURITY — Security content proper: vulnerabilities/CVEs, exploits, malware/reverse engineering, threat intel, pentest/red team, blue team/IR/SOC, forensics, AppSec/secure coding, cloud security/IAM, ICS/OT/IoT security, policy/compliance when written for practitioners.
#
# COMPUTER_SYSTEMS — Linux/Unix/Windows admin, filesystems, processes, services, systemd, Docker/K8s, virtualization, Terraform/Ansible used operationally.
#
# SOFTWARE_DEVELOPMENT — SDLC/CI/CD, code review, dependency management, secure coding practices when general, Git usage impacting code integrity.
#
# CRYPTOGRAPHY — Practical cryptography: hashing, encryption, signatures, TLS, key management (not pure math proofs).
#
# OS_COMMAND — Terminal/PowerShell/CMD usage, shell pipelines, admin scripts, package managers, config snippets focused on operating systems behavior.
#
# NETWORK_PROTOCOL — TCP/IP, DNS, HTTP/TLS, SSH, SMTP, BGP/OSPF—protocol mechanics, packet structure, handshake/state machines.
#
# PROGRAMMING — General coding without a security angle (Python/Go/C/C++/JS snippets, API usage, tutorials not tied to security).
#
# OTHER — Everything else.
#
# Classify the input text provided by the user into one of the specified labels and explain your reasoning.
#
# """

    # Configuration
    PARQUET_PATH = "/Users/cooperlarhette/Downloads/jobs_2025-10-07_user-a0d28ecc-10c9-489f-9b2d-51658e3f12aa_job-47518aa2-45a9-4865-a398-a6af39e06e6c_inputs_inputs_part_0.snappy.parquet"
    INPUT_COLUMN = "SKYSIGHT_PROMPTS"
    FRONTIER_MODEL = "openrouter/openai/gpt-5"
    GOLDEN_MODEL = "openrouter/openai/gpt-5"
    TARGET_MODEL = "openrouter/openai/gpt-oss-20b"

    # Speed modes - choose one:
    # The BUDGET (max_metric_calls) is the main speed control
    MODE = "fast"  # Change to "balanced" or "thorough"

    if MODE == "fast":
        # ~30-45 minutes total
        N_SAMPLES = 450        # Fewer examples = faster golden creation
        MAX_METRIC_CALLS = 2000           # Fewer evaluations = faster optimization
        MAX_RUNTIME_MINUTES = 30      # 1 hour safety limit
    elif MODE == "balanced":
        # ~1-2 hours total (recommended)
        N_SAMPLES = 100
        MAX_METRIC_CALLS = 150
        MAX_RUNTIME_MINUTES = 120
    else:  # thorough
        # ~3-4 hours total
        N_SAMPLES = 200
        MAX_METRIC_CALLS = 300
        MAX_RUNTIME_MINUTES = 240

    print("="*80)
    print("GEPA OPTIMIZATION: PATENT ANALYSIS")
    print("="*80)
    print(f"Goal: Optimize prompt from {FRONTIER_MODEL} → {TARGET_MODEL}")
    print(f"Mode: {MODE.upper()}")
    print(f"  - Samples: {N_SAMPLES}")
    print(f"  - Budget: {MAX_METRIC_CALLS} evaluations")
    print(f"  - Max runtime: {MAX_RUNTIME_MINUTES} minutes")
    print()

    # Step 1: Create golden dataset (or load if exists)
    # examples = create_golden_dataset_from_parquet(
    #     parquet_path=PARQUET_PATH,
    #     input_column=INPUT_COLUMN,
    #     n_samples=N_SAMPLES,
    #     golden_model=GOLDEN_MODEL,
    #     system_prompt=SYSTEM_PROMPT,
    #     output_file="patent_golden_examples.jsonl",
    #     use_existing=True,  # Set to False to force regeneration
    # )


    load_dotenv()
    dataset = load_dataset("ccdv/patent-classification", "abstract")

    # Convert to pandas dataframe (using the train split)
    df = dataset["train"].to_pandas()

    # Get first 400 examples in the required format
    examples: list[DataInst] = [
        {
            "input": row["text"],
            "golden_output": row["label"],
        }
        for _, row in df.head(400).iterrows()
    ]

    # Step 2: Optimize prompt
    result = optimize_prompt(
        examples=examples,
        seed_prompt=SYSTEM_PROMPT,
        target_model=TARGET_MODEL,
        golden_model_name=FRONTIER_MODEL,
        judge_model=FRONTIER_MODEL,
        reflection_lm=FRONTIER_MODEL,
        # max_metric_calls=MAX_METRIC_CALLS,
        output_dir="./patent_optimization",
        max_runtime_minutes=MAX_RUNTIME_MINUTES
    )

    # Step 3: Save results
    with open("patent_optimization/final_result.json", 'w') as f:
        json.dump(result, f, indent=2)

    print("\n✓ Results saved to patent_optimization/final_result.json")
    print(f"\n🎯 Next: Test optimized prompt in Sutro with {TARGET_MODEL}")