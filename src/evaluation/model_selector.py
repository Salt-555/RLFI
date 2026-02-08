"""
Model Selector - Intelligent model selection based on eval scores, learning trend,
and grokking analysis. Only promotes models that show both good performance AND
evidence of actual learning (not just memorization).

Grokking Check: After evaluating learning curves, we run spectral analysis on the
model's weight matrices and eval curve phase detection to verify the model has
genuinely internalized trading patterns vs memorizing training data.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json
import os


@dataclass
class ModelEvaluation:
    """Stores evaluation metrics for a trained model."""
    model_id: str
    model_path: str
    eval_rewards: List[float]  # Rewards at each eval checkpoint
    eval_timesteps: List[int]  # Timesteps at each eval checkpoint
    eval_stds: List[float]  # Standard deviations at each checkpoint
    final_reward: float
    best_reward: float
    trend_score: float  # Positive = improving, negative = declining
    consistency_score: float  # Lower variance = more consistent
    learning_score: float  # Combined metric for model selection
    passed_selection: bool
    # Grokking analysis results
    grokking_score: float = 0.0
    grokking_phase: str = 'unknown'
    has_grokked: bool = False
    grokking_reason: str = ''
    grokking_details: Dict = field(default_factory=dict)


class ModelSelector:
    """
    Selects models based on evaluation performance and learning trend.
    
    Selection criteria:
    1. Final reward must be positive (model is profitable)
    2. Trend must be positive or stable (model learned, not just lucky)
    3. Consistency must be reasonable (not erratic)
    """
    
    def __init__(
        self,
        min_final_reward: float = 0.0,
        min_trend_score: float = -0.1,  # Allow slight decline if converged
        max_variance_ratio: float = 0.5,  # Std/mean ratio
        top_k: Optional[int] = None  # If set, only keep top K models
    ):
        self.min_final_reward = min_final_reward
        self.min_trend_score = min_trend_score
        self.max_variance_ratio = max_variance_ratio
        self.top_k = top_k
        self.evaluations: List[ModelEvaluation] = []
    
    def calculate_trend(self, rewards: List[float], timesteps: List[int]) -> float:
        """
        Calculate learning trend using linear regression slope.
        Normalized by reward magnitude for comparability.
        
        Returns:
            Positive value = improving over time
            Negative value = declining over time
            Near zero = stable/converged
        """
        if len(rewards) < 3:
            return 0.0
        
        # Normalize timesteps to [0, 1] range
        t = np.array(timesteps, dtype=float)
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        
        r = np.array(rewards, dtype=float)
        
        # Linear regression: r = slope * t + intercept
        # slope = cov(t, r) / var(t)
        slope = np.cov(t, r)[0, 1] / (np.var(t) + 1e-8)
        
        # Normalize by reward range for comparability
        reward_range = r.max() - r.min()
        if reward_range > 0:
            normalized_slope = slope / reward_range
        else:
            normalized_slope = 0.0
        
        return normalized_slope
    
    def calculate_consistency(self, rewards: List[float], stds: List[float]) -> float:
        """
        Calculate consistency score based on reward variance.
        
        Returns:
            Value between 0 and 1, higher = more consistent
        """
        if len(rewards) < 2:
            return 1.0
        
        # Use coefficient of variation (std/mean) as inconsistency measure
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        if abs(mean_reward) < 1e-6:
            return 0.5  # Can't calculate CV, return neutral
        
        cv = std_reward / abs(mean_reward)
        
        # Convert to consistency score (lower CV = higher consistency)
        # Cap CV at 2.0 for scoring purposes
        consistency = max(0, 1 - min(cv, 2.0) / 2.0)
        
        return consistency
    
    def calculate_learning_score(
        self, 
        final_reward: float, 
        trend_score: float, 
        consistency_score: float
    ) -> float:
        """
        Calculate combined learning score for model ranking.
        
        Weights:
        - 50% final performance (are we making money?)
        - 30% trend (did we learn or get lucky?)
        - 20% consistency (is the strategy stable?)
        """
        # Normalize final reward to roughly 0-1 scale (assuming rewards in -100 to 500 range)
        normalized_reward = np.clip(final_reward / 300, -1, 1)
        
        # Trend is already normalized, but clip for safety
        normalized_trend = np.clip(trend_score, -1, 1)
        
        # Consistency is already 0-1
        
        learning_score = (
            0.5 * normalized_reward +
            0.3 * normalized_trend +
            0.2 * consistency_score
        )
        
        return learning_score
    
    def evaluate_model(
        self,
        model_id: str,
        model_path: str,
        eval_rewards: List[float],
        eval_timesteps: List[int],
        eval_stds: Optional[List[float]] = None,
        loaded_model=None
    ) -> ModelEvaluation:
        """
        Evaluate a single model and determine if it passes selection criteria.
        
        If loaded_model is provided, also runs grokking analysis (weight matrix
        spectral analysis + eval curve phase detection) to verify the model has
        genuinely learned vs just memorized training data.
        
        Args:
            model_id: Unique model identifier
            model_path: Path to saved model file
            eval_rewards: Rewards at each eval checkpoint
            eval_timesteps: Timesteps at each eval checkpoint  
            eval_stds: Standard deviations at each eval checkpoint
            loaded_model: Optional loaded SB3 model for grokking analysis
        """
        if eval_stds is None:
            eval_stds = [0.0] * len(eval_rewards)
        
        final_reward = eval_rewards[-1] if eval_rewards else 0.0
        best_reward = max(eval_rewards) if eval_rewards else 0.0
        
        trend_score = self.calculate_trend(eval_rewards, eval_timesteps)
        consistency_score = self.calculate_consistency(eval_rewards, eval_stds)
        learning_score = self.calculate_learning_score(final_reward, trend_score, consistency_score)
        
        # Run grokking analysis if model is provided
        grokking_score = 0.0
        grokking_phase = 'unknown'
        has_grokked = False
        grokking_reason = 'no model provided for analysis'
        grokking_details = {}
        
        if loaded_model is not None:
            try:
                from src.evaluation.grokking_detector import GrokkingDetector
                
                detector = GrokkingDetector()
                analysis = detector.analyze_model(
                    model=loaded_model,
                    eval_rewards=eval_rewards,
                    eval_timesteps=eval_timesteps,
                    eval_stds=eval_stds,
                )
                
                grokking_score = analysis.grokking_score
                grokking_phase = analysis.phase
                has_grokked = analysis.has_grokked
                grokking_reason = analysis.reason
                grokking_details = detector.to_metadata_dict(analysis)
                
            except Exception as e:
                grokking_reason = f'grokking analysis failed: {e}'
                print(f"  [{model_id}] Warning: Grokking analysis failed: {e}")
        
        # Determine if model passes selection criteria
        # Original criteria (performance-based)
        performance_passed = (
            final_reward >= self.min_final_reward and
            trend_score >= self.min_trend_score and
            consistency_score >= (1 - self.max_variance_ratio)
        )
        
        # Grokking gate: if we ran grokking analysis, require it to pass
        # If no model was provided (legacy path), skip grokking gate
        if loaded_model is not None:
            grokking_passed = has_grokked
            passed = performance_passed and grokking_passed
        else:
            passed = performance_passed
        
        # Boost learning_score with grokking score for better ranking
        if loaded_model is not None and grokking_score > 0:
            # Blend: 60% original learning score, 40% grokking score
            learning_score = 0.6 * learning_score + 0.4 * grokking_score
        
        evaluation = ModelEvaluation(
            model_id=model_id,
            model_path=model_path,
            eval_rewards=eval_rewards,
            eval_timesteps=eval_timesteps,
            eval_stds=eval_stds,
            final_reward=final_reward,
            best_reward=best_reward,
            trend_score=trend_score,
            consistency_score=consistency_score,
            learning_score=learning_score,
            passed_selection=passed,
            grokking_score=grokking_score,
            grokking_phase=grokking_phase,
            has_grokked=has_grokked,
            grokking_reason=grokking_reason,
            grokking_details=grokking_details
        )
        
        self.evaluations.append(evaluation)
        return evaluation
    
    def select_models(self) -> List[ModelEvaluation]:
        """
        Select models that pass criteria, optionally limited to top K.
        
        Returns:
            List of ModelEvaluation for models that passed selection,
            sorted by learning_score descending.
        """
        # Filter to models that passed
        passed_models = [e for e in self.evaluations if e.passed_selection]
        
        # Sort by learning score
        passed_models.sort(key=lambda x: x.learning_score, reverse=True)
        
        # Limit to top K if specified
        if self.top_k is not None and len(passed_models) > self.top_k:
            passed_models = passed_models[:self.top_k]
        
        return passed_models
    
    def get_selection_report(self) -> str:
        """Generate a human-readable selection report."""
        lines = [
            "=" * 60,
            "MODEL SELECTION REPORT",
            "=" * 60,
            f"Total models evaluated: {len(self.evaluations)}",
            f"Selection criteria:",
            f"  - Min final reward: {self.min_final_reward}",
            f"  - Min trend score: {self.min_trend_score}",
            f"  - Max variance ratio: {self.max_variance_ratio}",
            ""
        ]
        
        # Sort all by learning score for display
        sorted_evals = sorted(self.evaluations, key=lambda x: x.learning_score, reverse=True)
        
        for i, e in enumerate(sorted_evals, 1):
            status = "✓ PASSED" if e.passed_selection else "✗ FAILED"
            lines.append(f"\n{i}. {e.model_id} [{status}]")
            lines.append(f"   Final Reward: {e.final_reward:.2f} | Best: {e.best_reward:.2f}")
            lines.append(f"   Trend: {e.trend_score:+.3f} | Consistency: {e.consistency_score:.2f}")
            lines.append(f"   Learning Score: {e.learning_score:.3f}")
            
            # Grokking analysis results
            if e.grokking_phase != 'unknown':
                grok_status = "✓" if e.has_grokked else "✗"
                lines.append(f"   Grokking: {grok_status} {e.grokking_phase} "
                           f"(score: {e.grokking_score:.3f})")
                if e.grokking_reason:
                    lines.append(f"   Grokking Detail: {e.grokking_reason}")
            
            # Show why it failed if applicable
            if not e.passed_selection:
                reasons = []
                if e.final_reward < self.min_final_reward:
                    reasons.append(f"low final reward ({e.final_reward:.2f} < {self.min_final_reward})")
                if e.trend_score < self.min_trend_score:
                    reasons.append(f"negative trend ({e.trend_score:.3f} < {self.min_trend_score})")
                if e.consistency_score < (1 - self.max_variance_ratio):
                    reasons.append(f"inconsistent ({e.consistency_score:.2f})")
                if e.grokking_phase != 'unknown' and not e.has_grokked:
                    reasons.append(f"failed grokking check ({e.grokking_phase})")
                lines.append(f"   Reason: {', '.join(reasons)}")
        
        selected = self.select_models()
        lines.append("\n" + "=" * 60)
        lines.append(f"SELECTED FOR TESTING: {len(selected)} models")
        lines.append("=" * 60)
        
        for e in selected:
            lines.append(f"  - {e.model_id} (score: {e.learning_score:.3f})")
        
        return "\n".join(lines)
    
    def save_results(self, filepath: str):
        """Save evaluation results to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        def to_builtin(obj):
            """Convert numpy/frozendict types to JSON-serializable Python types."""
            try:
                import numpy as _np
            except Exception:
                _np = None
            
            # Handle frozendict-like objects
            if hasattr(obj, "items") and not isinstance(obj, dict):
                try:
                    obj = dict(obj)
                except Exception:
                    pass
            
            if isinstance(obj, dict):
                return {k: to_builtin(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [to_builtin(v) for v in obj]
            if _np is not None:
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                if isinstance(obj, _np.generic):
                    return obj.item()
            return obj
        
        results = {
            "criteria": {
                "min_final_reward": self.min_final_reward,
                "min_trend_score": self.min_trend_score,
                "max_variance_ratio": self.max_variance_ratio,
                "top_k": self.top_k
            },
            "evaluations": [
                {
                    "model_id": e.model_id,
                    "model_path": e.model_path,
                    "final_reward": float(e.final_reward),
                    "best_reward": float(e.best_reward),
                    "trend_score": float(e.trend_score),
                    "consistency_score": float(e.consistency_score),
                    "learning_score": float(e.learning_score),
                    "passed_selection": bool(e.passed_selection),
                    "eval_rewards": [float(r) for r in e.eval_rewards],
                    "eval_timesteps": [int(t) for t in e.eval_timesteps],
                    "grokking_score": float(e.grokking_score),
                    "grokking_phase": e.grokking_phase,
                    "has_grokked": bool(e.has_grokked),
                    "grokking_reason": e.grokking_reason,
                    "grokking_details": e.grokking_details
                }
                for e in self.evaluations
            ],
            "selected_models": [e.model_id for e in self.select_models()]
        }
        
        with open(filepath, 'w') as f:
            json.dump(to_builtin(results), f, indent=2)
        
        print(f"Selection results saved to {filepath}")
    
    @staticmethod
    def load_eval_from_npz(npz_path: str) -> Tuple[List[float], List[int], List[float]]:
        """Load evaluation history from SB3's evaluations.npz file."""
        data = np.load(npz_path)
        
        timesteps = data['timesteps'].tolist()
        results = data['results']  # Shape: (n_evals, n_episodes)
        
        # Calculate mean and std per evaluation
        rewards = [float(np.mean(r)) for r in results]
        stds = [float(np.std(r)) for r in results]
        
        return rewards, timesteps, stds
