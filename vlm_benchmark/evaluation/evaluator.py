"""
Attack Evaluator

Main orchestrator for evaluating adversarial attacks on VLMs.
Combines VLM inference, semantic similarity scoring, and metrics calculation.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .vlm_inference import VLMInference, VLMResponse
from .semantic_scorer import SemanticSimilarityScorer
from .metrics import calculate_metrics, print_metrics, EvaluationMetrics


class AttackEvaluator:
    """
    Complete evaluation pipeline for adversarial attacks.

    Workflow:
        1. Query VLM with adversarial images
        2. Compute semantic similarity to target description
        3. Calculate ASR and other metrics
        4. Save results

    Example:
        evaluator = AttackEvaluator(
            vlm_model="gpt-4o-mini-2024-07-18",
            scorer_model="gpt-4o-mini-2024-07-18",
            api_key="..."
        )

        results = evaluator.evaluate(
            image_dir="experiments/physpatch/run1",
            query="Describe the main object in the scene.",
            reference_text="A stop sign is visible",
            output_dir="experiments/physpatch/run1/evaluation"
        )
    """

    def __init__(
        self,
        vlm_model: str = "gpt-4o-mini-2024-07-18",
        scorer_model: str = "gpt-4o-mini-2024-07-18",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        vlm_api_base: Optional[str] = None,
        threshold: float = 0.5,
        max_workers: int = 10
    ):
        """
        Initialize attack evaluator.

        Args:
            vlm_model: VLM for inference (e.g., "gpt-4o-mini-2024-07-18")
            scorer_model: Model for semantic scoring (e.g., "gpt-4o-mini-2024-07-18")
            api_key: API key for both VLM and scorer
            api_base: Optional custom API base (used for scorer if vlm_api_base not set)
            vlm_api_base: Optional VLM-specific API base (e.g., local vllm server URL)
            threshold: Success threshold for ASR (default: 0.5)
        """
        self.vlm = VLMInference(
            model=vlm_model,
            api_key=api_key,
            api_base=vlm_api_base or api_base,
            max_workers=max_workers
        )

        self.scorer = SemanticSimilarityScorer(
            model=scorer_model,
            api_key=api_key,
            api_base=api_base
        )

        self.threshold = threshold
        self.max_workers = max_workers

    def evaluate(
        self,
        image_dir: str,
        query: str,
        reference_text: str,
        output_dir: Optional[str] = None,
        save_responses: bool = True,
        verbose: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate adversarial attack.

        Args:
            image_dir: Directory containing adversarial images
            query: Query to ask VLM (e.g., "Describe the main object...")
            reference_text: Target description (e.g., "A stop sign is visible")
            output_dir: Directory to save results (default: same as image_dir)
            save_responses: Save VLM responses and scores to files
            verbose: Print progress and results

        Returns:
            EvaluationMetrics with ASR and similarity scores
        """
        if verbose:
            print("=" * 70)
            print("Attack Evaluation Pipeline")
            print("=" * 70)
            print(f"Image dir:      {image_dir}")
            print(f"VLM model:      {self.vlm.model}")
            print(f"Scorer model:   {self.scorer.model}")
            print(f"Query:          {query}")
            print(f"Reference:      {reference_text}")
            print(f"Threshold:      {self.threshold}")
            print("=" * 70)
            print()

        # Step 1: Get image paths
        image_paths = self._get_image_paths(image_dir)

        if not image_paths:
            print(f"[Warning] No images found in {image_dir}")
            return calculate_metrics([], self.threshold)

        if verbose:
            print(f"[1/3] Found {len(image_paths)} images\n")

        # Step 2: Query VLM
        if verbose:
            print("[2/3] Querying VLM with adversarial images...")

        responses = self.vlm.query_images(
            image_paths=image_paths,
            query=query,
            verbose=verbose
        )

        if verbose:
            print(f"\n✓ Got {len(responses)} VLM responses\n")

        # Step 3: Compute semantic similarity (parallel)
        if verbose:
            print("[3/3] Computing semantic similarity scores...")

        similarity_scores = [0.0] * len(responses)
        completed = [0]

        def _score(args):
            idx, response = args
            if response.error:
                return idx, 0.0
            return idx, self.scorer.compute_similarity(response.response, reference_text, verbose=False)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_score, (i, r)): i
                for i, r in enumerate(responses)
            }
            for future in as_completed(futures):
                idx, score = future.result()
                similarity_scores[idx] = score
                completed[0] += 1
                if verbose:
                    r = responses[idx]
                    if r.error:
                        print(f"  [{completed[0]}/{len(responses)}] Skipping (error): {Path(r.image_path).name}")
                    else:
                        success = "✓" if score > self.threshold else "✗"
                        print(f"  [{completed[0]}/{len(responses)}] {Path(r.image_path).name}: {score:.4f} {success}")

        # Step 4: Calculate metrics
        metrics = calculate_metrics(similarity_scores, self.threshold)

        if verbose:
            print()
            print_metrics(metrics, verbose=False)

        # Step 5: Save results
        if save_responses and output_dir:
            self._save_results(
                responses=responses,
                similarity_scores=similarity_scores,
                metrics=metrics,
                output_dir=output_dir,
                query=query,
                reference_text=reference_text
            )

        return metrics

    def _get_image_paths(self, image_dir: str) -> List[str]:
        """Get sorted list of image paths from directory."""
        image_dir_path = Path(image_dir)
        if not image_dir_path.exists():
            return []

        image_extensions = {'.png', '.jpg', '.jpeg'}
        image_paths = sorted([
            str(p) for p in image_dir_path.iterdir()
            if p.suffix.lower() in image_extensions
        ])

        return image_paths

    def _save_results(
        self,
        responses: List[VLMResponse],
        similarity_scores: List[float],
        metrics: EvaluationMetrics,
        output_dir: str,
        query: str,
        reference_text: str
    ):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save VLM responses
        responses_file = output_path / "vlm_responses.jsonl"
        with open(responses_file, 'w') as f:
            for response, score, success in zip(
                responses,
                similarity_scores,
                metrics.per_sample_success
            ):
                entry = {
                    "image_path": response.image_path,
                    "model": response.model_name,
                    "query": response.query,
                    "response": response.response,
                    "error": response.error,
                    "similarity_score": score,
                    "success": success,
                    "metadata": response.metadata
                }
                f.write(json.dumps(entry) + '\n')

        # Save evaluation metrics
        metrics_file = output_path / "evaluation_metrics.json"
        metrics_data = {
            "vlm_model": self.vlm.model,
            "scorer_model": self.scorer.model,
            "query": query,
            "reference_text": reference_text,
            "threshold": self.threshold,
            "asr": metrics.asr,
            "avg_similarity": metrics.avg_similarity,
            "num_successful": metrics.num_successful,
            "num_total": metrics.num_total,
            "per_sample_scores": metrics.per_sample_scores,
            "per_sample_success": metrics.per_sample_success
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        print(f"\n✓ Results saved to {output_dir}/")
        print(f"  - vlm_responses.jsonl")
        print(f"  - evaluation_metrics.json")
