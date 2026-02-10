"""
Grokking Detector - Determines whether a trained model has genuinely learned
generalizable patterns vs merely memorizing training data.

Grokking is the phenomenon where a model transitions from memorization to true
generalization after extended training. This module detects which phase a model
is in using three complementary approaches:

1. Weight Matrix Spectral Analysis - Low effective rank = structured learning
2. Eval Curve Phase Detection - Hockey-stick improvement = grokking signature
3. Generalization Gap Analysis - Train vs eval divergence = memorization signal

Integration Points:
- Called by ModelSelector after training completes (Gate 1: pre-backtest)
- Results stored in model metadata for lifecycle tracking
- Models that fail grokking checks are blocked from backtesting
"""
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class GrokkingAnalysis:
    """Complete grokking analysis results for a trained model."""
    # Phase detection
    phase: str  # 'pre_grok', 'grokking', 'post_grok', 'memorizing', 'insufficient_data'
    phase_confidence: float  # 0.0 to 1.0
    
    # Weight matrix health
    avg_effective_rank_ratio: float  # 0.0 to 1.0, lower = more structured
    avg_weight_norm: float
    weight_norm_trend: float  # Negative = simplifying (good)
    
    # Eval curve analysis
    generalization_gap: float  # train - eval performance
    eval_improvement_rate: float  # Late-stage eval improvement
    eval_stability: float  # 0.0 to 1.0, higher = more stable
    
    # Overall verdict
    has_grokked: bool
    grokking_score: float  # 0.0 to 1.0, composite score
    reason: str  # Human-readable explanation
    
    # Raw data for metadata storage
    layer_details: Dict[str, Any] = field(default_factory=dict)


class GrokkingDetector:
    """
    Analyzes trained RL models to detect whether genuine learning (grokking)
    has occurred vs pure memorization.
    
    For RL trading models, "grokking" means the model learned generalizable
    market patterns (mean reversion, momentum, risk management) rather than
    memorizing specific price sequences from training data.
    """
    
    # Thresholds tuned for SB3 PPO/SAC TRADING models
    # IMPORTANT: Day trading stocks is NOISY and NON-STATIONARY compared to algorithmic tasks!
    # Research papers (Yunis et al., Power et al.) focused on clean math problems (modular arithmetic).
    # Stock markets have:
    #   - High noise-to-signal ratio
    #   - Non-stationary distributions (regime shifts)
    #   - External shocks (earnings, news, macro)
    #   - No clean underlying mathematical structure
    # 
    # Therefore, trading thresholds MUST be more permissive than algorithmic task thresholds.
    # A model with 0.50 grokking score on stocks may be excellent even if it would fail on modular arithmetic.
    
    # RELAXED for trading reality:
    EFFECTIVE_RANK_RATIO_THRESHOLD = 0.70  # Below this = structured learning (relaxed from 0.65 - trading is noisy)
    WEIGHT_NORM_GROWTH_THRESHOLD = 0.35  # Above this = weights still growing (relaxed from 0.3)
    MIN_EVAL_IMPROVEMENT = 0.15  # Minimum late-stage improvement ratio (realistic for noisy markets)
    MAX_GENERALIZATION_GAP = 0.5  # Max acceptable train-eval divergence (restored to 0.5 - markets diverge)
    MIN_EVAL_STABILITY = 0.35  # Minimum eval stability score (relaxed from 0.3 - markets are volatile)
    
    # Composite score weights - balanced approach for noisy data
    WEIGHT_SPECTRAL = 0.30  # Weight matrix structure
    WEIGHT_EVAL_CURVE = 0.40  # Eval curve shape (important but not everything in noisy markets)
    WEIGHT_STABILITY = 0.30  # Consistency of performance (important for trading)
    
    def __init__(self):
        pass
    
    def analyze_model(
        self,
        model,
        eval_rewards: List[float],
        eval_timesteps: List[int],
        eval_stds: Optional[List[float]] = None,
        train_rewards: Optional[List[float]] = None,
    ) -> GrokkingAnalysis:
        """
        Run complete grokking analysis on a trained model.
        
        Args:
            model: Trained SB3 model (PPO, SAC, etc.)
            eval_rewards: Evaluation rewards at each checkpoint
            eval_timesteps: Timesteps at each eval checkpoint
            eval_stds: Standard deviations at each eval checkpoint
            train_rewards: Training rewards (if available)
        
        Returns:
            GrokkingAnalysis with complete diagnosis
        """
        if eval_stds is None:
            eval_stds = [0.0] * len(eval_rewards)
        
        # 1. Spectral analysis of weight matrices
        spectral_results = self._analyze_weight_matrices(model)
        
        # 2. Eval curve phase detection
        phase_results = self._detect_eval_phase(
            eval_rewards, eval_timesteps, eval_stds
        )
        
        # 3. Generalization gap (if train rewards available)
        gen_gap = self._compute_generalization_gap(
            train_rewards, eval_rewards
        ) if train_rewards else 0.0
        
        # 4. Eval stability
        eval_stability = self._compute_eval_stability(eval_rewards, eval_stds)
        
        # 5. Compute composite grokking score
        spectral_score = self._score_spectral(spectral_results)
        curve_score = self._score_eval_curve(phase_results, eval_rewards)
        stability_score = eval_stability
        
        grokking_score = (
            self.WEIGHT_SPECTRAL * spectral_score +
            self.WEIGHT_EVAL_CURVE * curve_score +
            self.WEIGHT_STABILITY * stability_score
        )
        
        # 6. Determine verdict
        has_grokked, reason = self._determine_verdict(
            phase_results['phase'],
            phase_results['confidence'],
            spectral_results,
            grokking_score,
            gen_gap,
            eval_stability,
            eval_rewards
        )
        
        return GrokkingAnalysis(
            phase=phase_results['phase'],
            phase_confidence=phase_results['confidence'],
            avg_effective_rank_ratio=spectral_results['avg_rank_ratio'],
            avg_weight_norm=spectral_results['avg_weight_norm'],
            weight_norm_trend=spectral_results['norm_trend'],
            generalization_gap=gen_gap,
            eval_improvement_rate=phase_results.get('late_improvement', 0.0),
            eval_stability=eval_stability,
            has_grokked=has_grokked,
            grokking_score=grokking_score,
            reason=reason,
            layer_details=spectral_results.get('layers', {})
        )
    
    def _analyze_weight_matrices(self, model) -> Dict[str, Any]:
        """
        Perform spectral analysis on all weight matrices in the policy network.
        
        Key insight from research: grokking coincides with weight matrices
        discovering low-rank solutions. Models that have genuinely learned
        will have lower effective rank ratios (more structured weights).
        
        Returns dict with:
            avg_rank_ratio: Average effective rank / full rank across layers
            avg_weight_norm: Average L2 norm of weight matrices
            norm_trend: Whether norms are growing or shrinking (by layer depth)
            layers: Per-layer breakdown
        """
        layers = {}
        rank_ratios = []
        weight_norms = []
        spectral_norms = []
        
        try:
            # Access the policy network parameters
            policy = model.policy
            
            for name, param in policy.named_parameters():
                # Only analyze weight matrices (skip biases, 1D params)
                if 'weight' not in name or len(param.shape) < 2:
                    continue
                
                weight = param.data.detach().cpu().float()
                
                # Skip very small layers (e.g., final output)
                if weight.shape[0] < 4 or weight.shape[1] < 4:
                    continue
                
                try:
                    # SVD analysis
                    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                    
                    # Effective rank via participation ratio
                    # High effective rank = uniformly distributed singular values = random/unstructured
                    # Low effective rank = few dominant singular values = structured learning
                    S_squared = S ** 2
                    total = S_squared.sum()
                    if total > 0:
                        effective_rank = (total ** 2) / (S_squared ** 2).sum()
                        full_rank = min(weight.shape)
                        rank_ratio = effective_rank.item() / full_rank
                    else:
                        rank_ratio = 1.0
                    
                    # Weight norm (L2)
                    w_norm = torch.norm(weight, p='fro').item()
                    
                    # Spectral norm (largest singular value)
                    s_norm = S[0].item() if len(S) > 0 else 0.0
                    
                    # Condition number (ratio of largest to smallest singular value)
                    # Very high = poorly conditioned = may not have converged
                    s_min = S[-1].item() if len(S) > 0 else 1e-8
                    condition = S[0].item() / (s_min + 1e-8) if len(S) > 0 else float('inf')
                    
                    rank_ratios.append(rank_ratio)
                    weight_norms.append(w_norm)
                    spectral_norms.append(s_norm)
                    
                    layers[name] = {
                        'shape': list(weight.shape),
                        'effective_rank_ratio': round(rank_ratio, 4),
                        'weight_norm': round(w_norm, 4),
                        'spectral_norm': round(s_norm, 4),
                        'condition_number': round(min(condition, 1e6), 2),
                        'top_3_sv': [round(s, 4) for s in S[:3].tolist()],
                    }
                    
                except Exception as e:
                    # SVD can fail on degenerate matrices
                    layers[name] = {'error': str(e)}
                    continue
            
        except Exception as e:
            return {
                'avg_rank_ratio': 1.0,
                'avg_weight_norm': 0.0,
                'norm_trend': 0.0,
                'layers': {},
                'error': str(e)
            }
        
        # Compute aggregates
        avg_rank_ratio = np.mean(rank_ratios) if rank_ratios else 1.0
        avg_weight_norm = np.mean(weight_norms) if weight_norms else 0.0
        
        # Norm trend: compare early layers to later layers
        # If later layers have smaller norms, model is simplifying (good)
        if len(weight_norms) >= 2:
            mid = len(weight_norms) // 2
            early_norm = np.mean(weight_norms[:mid])
            late_norm = np.mean(weight_norms[mid:])
            norm_trend = (late_norm - early_norm) / (early_norm + 1e-8)
        else:
            norm_trend = 0.0
        
        return {
            'avg_rank_ratio': avg_rank_ratio,
            'avg_weight_norm': avg_weight_norm,
            'norm_trend': norm_trend,
            'layers': layers
        }
    
    def _detect_eval_phase(
        self,
        eval_rewards: List[float],
        eval_timesteps: List[int],
        eval_stds: List[float]
    ) -> Dict[str, Any]:
        """
        Detect which phase of learning the model is in based on the eval curve shape.
        
        Grokking signature: flat/declining eval followed by sharp improvement.
        
        Phases:
        - 'post_grok': Eval improved significantly in later training -> GOOD
        - 'grokking': Eval is actively improving -> PROMISING  
        - 'improving': Steady improvement throughout -> GOOD (normal learning)
        - 'memorizing': Eval stagnant/declining while presumably training converged -> BAD
        - 'pre_grok': Not enough improvement yet -> NEEDS MORE TRAINING
        - 'insufficient_data': Too few eval points
        """
        if len(eval_rewards) < 5:
            return {
                'phase': 'insufficient_data',
                'confidence': 0.0,
                'late_improvement': 0.0
            }
        
        rewards = np.array(eval_rewards, dtype=float)
        n = len(rewards)
        
        # Smooth with moving average to reduce noise
        window = max(3, n // 5)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
        
        if len(smoothed) < 4:
            smoothed = rewards
        
        # Split into segments for analysis
        third = len(smoothed) // 3
        early = smoothed[:third] if third > 0 else smoothed[:1]
        middle = smoothed[third:2*third] if third > 0 else smoothed[1:2]
        late = smoothed[2*third:] if third > 0 else smoothed[2:]
        
        early_mean = np.mean(early)
        middle_mean = np.mean(middle)
        late_mean = np.mean(late)
        
        # Overall trend
        overall_improvement = late_mean - early_mean
        late_improvement = late_mean - middle_mean
        
        # Reward range for normalization
        reward_range = rewards.max() - rewards.min()
        if reward_range < 1e-6:
            reward_range = abs(np.mean(rewards)) + 1e-6
        
        # Normalized improvements
        norm_overall = overall_improvement / reward_range if reward_range > 0 else 0
        norm_late = late_improvement / reward_range if reward_range > 0 else 0
        
        # Check for hockey-stick pattern (classic grokking)
        # Early flat/declining, then sharp late improvement
        early_to_mid_change = (middle_mean - early_mean) / (reward_range + 1e-8)
        mid_to_late_change = (late_mean - middle_mean) / (reward_range + 1e-8)
        
        # Detect phase
        if mid_to_late_change > 0.3 and early_to_mid_change < 0.1:
            # Hockey stick: flat early, sharp late improvement
            phase = 'post_grok'
            confidence = min(1.0, mid_to_late_change)
        elif norm_overall > 0.3 and norm_late > 0.15:
            # Still actively improving with good late-stage gains
            phase = 'grokking'
            confidence = min(1.0, norm_late * 2)
        elif norm_overall > 0.2:
            # Steady improvement throughout (normal healthy learning)
            phase = 'improving'
            confidence = min(1.0, norm_overall)
        elif norm_overall < -0.1:
            # Getting worse over time
            phase = 'memorizing'
            confidence = min(1.0, abs(norm_overall))
        elif abs(norm_overall) < 0.1 and late_mean > 0:
            # Flat but positive - could still be useful, converged early
            phase = 'pre_grok'
            confidence = 0.5
        else:
            # Flat and not positive
            phase = 'memorizing'
            confidence = 0.6
        
        return {
            'phase': phase,
            'confidence': round(confidence, 3),
            'late_improvement': round(norm_late, 4),
            'overall_improvement': round(norm_overall, 4),
            'early_mean': round(float(early_mean), 4),
            'middle_mean': round(float(middle_mean), 4),
            'late_mean': round(float(late_mean), 4),
        }
    
    def _compute_generalization_gap(
        self,
        train_rewards: List[float],
        eval_rewards: List[float]
    ) -> float:
        """
        Compute the generalization gap between training and evaluation performance.
        
        A large gap (train >> eval) indicates the model has memorized training data
        rather than learning generalizable patterns.
        
        Returns:
            Gap ratio in [0, 1+]. 0 = no gap (perfect generalization).
            > 0.5 = concerning gap. > 1.0 = severe memorization.
        """
        if not train_rewards or not eval_rewards:
            return 0.0
        
        # Use the last 20% of each curve
        n_train = max(1, len(train_rewards) // 5)
        n_eval = max(1, len(eval_rewards) // 5)
        
        train_late = np.mean(train_rewards[-n_train:])
        eval_late = np.mean(eval_rewards[-n_eval:])
        
        # Normalize by training performance
        if abs(train_late) < 1e-6:
            return 0.0
        
        gap = (train_late - eval_late) / (abs(train_late) + 1e-8)
        return max(0.0, gap)  # Only care about train > eval gap
    
    def _compute_eval_stability(
        self,
        eval_rewards: List[float],
        eval_stds: List[float]
    ) -> float:
        """
        Measure how stable the eval performance is in the final portion of training.
        
        A model that has grokked should show stable, consistent eval performance
        in its later stages. Erratic eval = hasn't settled into a generalizable strategy.
        
        Returns:
            Stability score in [0, 1]. Higher = more stable.
        """
        if len(eval_rewards) < 3:
            return 0.5  # Neutral if insufficient data
        
        # Focus on the last 40% of eval checkpoints
        cutoff = max(3, int(len(eval_rewards) * 0.6))
        late_rewards = np.array(eval_rewards[cutoff:], dtype=float)
        
        if len(late_rewards) < 2:
            late_rewards = np.array(eval_rewards[-3:], dtype=float)
        
        # Coefficient of variation (lower = more stable)
        mean_r = np.mean(late_rewards)
        std_r = np.std(late_rewards)
        
        if abs(mean_r) < 1e-6:
            return 0.3  # Near-zero mean is concerning
        
        cv = std_r / (abs(mean_r) + 1e-8)
        
        # Also check within-evaluation variance (how consistent across episodes)
        late_stds = eval_stds[cutoff:] if len(eval_stds) > cutoff else eval_stds[-3:]
        avg_within_std = np.mean(late_stds) if late_stds else 0
        
        # Combine: lower CV and lower within-eval variance = higher stability
        # CV typically ranges 0 to 2+ for RL rewards
        stability_from_cv = max(0, 1.0 - min(cv, 2.0) / 2.0)
        
        # Within-eval variance relative to mean
        if abs(mean_r) > 1e-6:
            within_ratio = avg_within_std / (abs(mean_r) + 1e-8)
            stability_from_within = max(0, 1.0 - min(within_ratio, 2.0) / 2.0)
        else:
            stability_from_within = 0.3
        
        # Weighted combination
        stability = 0.6 * stability_from_cv + 0.4 * stability_from_within
        return round(stability, 4)
    
    def _score_spectral(self, spectral_results: Dict[str, Any]) -> float:
        """
        Score weight matrix quality from spectral analysis.
        
        Returns 0.0 to 1.0. Higher = better (more structured weights).
        """
        rank_ratio = spectral_results['avg_rank_ratio']
        norm_trend = spectral_results['norm_trend']
        
        # Lower rank ratio = more structure = better
        # Typical range: 0.3 (very structured) to 0.9 (random)
        rank_score = max(0, 1.0 - rank_ratio)
        
        # Negative norm trend = weights simplifying over depth = good
        # Positive = growing = potentially unstable
        if norm_trend < 0:
            trend_score = min(1.0, 0.5 + abs(norm_trend))
        else:
            trend_score = max(0, 0.5 - norm_trend)
        
        return 0.7 * rank_score + 0.3 * trend_score
    
    def _score_eval_curve(
        self,
        phase_results: Dict[str, Any],
        eval_rewards: List[float]
    ) -> float:
        """
        Score the eval curve quality.
        
        Returns 0.0 to 1.0. Higher = more evidence of genuine learning.
        """
        phase = phase_results['phase']
        confidence = phase_results['confidence']
        late_improvement = phase_results.get('late_improvement', 0.0)
        
        # Phase-based scoring
        phase_scores = {
            'post_grok': 0.95,     # Best - clear grokking happened
            'grokking': 0.80,      # Active improvement
            'improving': 0.70,     # Steady learning (good)
            'pre_grok': 0.30,      # Hasn't gotten there yet
            'memorizing': 0.10,    # Stuck memorizing
            'insufficient_data': 0.40,  # Unknown
        }
        
        base_score = phase_scores.get(phase, 0.3)
        
        # Modulate by confidence
        score = base_score * (0.5 + 0.5 * confidence)
        
        # Bonus for positive late improvement
        if late_improvement > 0:
            score = min(1.0, score + 0.1 * late_improvement)
        
        # Check if final eval rewards are actually positive
        if eval_rewards:
            final_rewards = eval_rewards[-3:] if len(eval_rewards) >= 3 else eval_rewards
            if np.mean(final_rewards) <= 0:
                # Unprofitable models get a penalty regardless of curve shape
                score *= 0.5
        
        return round(min(1.0, max(0.0, score)), 4)
    
    def _determine_verdict(
        self,
        phase: str,
        phase_confidence: float,
        spectral_results: Dict[str, Any],
        grokking_score: float,
        gen_gap: float,
        eval_stability: float,
        eval_rewards: List[float]
    ) -> Tuple[bool, str]:
        """
        Determine final grokking verdict with human-readable explanation.
        
        A model has "grokked" if it shows evidence of having learned
        generalizable patterns. This is NOT just about the score threshold -
        we also check for disqualifying factors.
        
        Returns:
            (has_grokked: bool, reason: str)
        """
        reasons = []
        disqualifiers = []
        
        # Check disqualifying factors first
        if gen_gap > self.MAX_GENERALIZATION_GAP:
            disqualifiers.append(
                f"large generalization gap ({gen_gap:.2f}) indicates memorization"
            )
        
        if phase == 'memorizing' and phase_confidence > 0.5:
            disqualifiers.append(
                f"eval curve shows memorization pattern (confidence: {phase_confidence:.2f})"
            )
        
        rank_ratio = spectral_results['avg_rank_ratio']
        if rank_ratio > 0.85:
            disqualifiers.append(
                f"weight matrices lack structure (rank ratio: {rank_ratio:.2f}, "
                f"expected < {self.EFFECTIVE_RANK_RATIO_THRESHOLD})"
            )
        
        # Check if final eval performance is actually profitable
        if eval_rewards and len(eval_rewards) >= 3:
            final_mean = np.mean(eval_rewards[-3:])
            if final_mean < 0:
                disqualifiers.append(
                    f"final eval reward is negative ({final_mean:.2f})"
                )
        
        # Qualifying factors
        if phase in ['post_grok', 'grokking']:
            reasons.append(f"eval curve shows {phase} pattern")
        elif phase == 'improving':
            reasons.append("steady learning improvement throughout training")
        
        if rank_ratio < self.EFFECTIVE_RANK_RATIO_THRESHOLD:
            reasons.append(
                f"structured weight matrices (rank ratio: {rank_ratio:.2f})"
            )
        
        if eval_stability > 0.6:
            reasons.append(f"stable eval performance ({eval_stability:.2f})")
        
        norm_trend = spectral_results['norm_trend']
        if norm_trend < -0.05:
            reasons.append("weights are simplifying (good generalization signal)")
        
        # Final decision
        # Need score above threshold AND no hard disqualifiers
        # TRADING-SPECIFIC: 0.45 is realistic for noisy stock markets (not 0.60!)
        # Algorithmic tasks (clean math) can demand 0.60+, but stocks are messy.
        score_passes = grokking_score >= 0.45
        has_hard_disqualifier = len(disqualifiers) >= 2 or (
            len(disqualifiers) >= 1 and grokking_score < 0.55
        )
        
        has_grokked = score_passes and not has_hard_disqualifier
        
        # Build explanation
        if has_grokked:
            reason = f"PASSED (score: {grokking_score:.3f}). " + "; ".join(reasons[:3])
        else:
            all_issues = disqualifiers + [f"low grokking score ({grokking_score:.3f})"] if not score_passes else disqualifiers
            reason = f"FAILED (score: {grokking_score:.3f}). " + "; ".join(all_issues[:3])
            if reasons:
                reason += f". Positives: " + "; ".join(reasons[:2])
        
        return has_grokked, reason
    
    def to_metadata_dict(self, analysis: GrokkingAnalysis) -> Dict[str, Any]:
        """
        Convert GrokkingAnalysis to a serializable dict for YAML metadata storage.
        """
        return {
            'phase': analysis.phase,
            'phase_confidence': round(analysis.phase_confidence, 4),
            'has_grokked': analysis.has_grokked,
            'grokking_score': round(analysis.grokking_score, 4),
            'reason': analysis.reason,
            'avg_effective_rank_ratio': round(analysis.avg_effective_rank_ratio, 4),
            'avg_weight_norm': round(analysis.avg_weight_norm, 4),
            'weight_norm_trend': round(analysis.weight_norm_trend, 4),
            'generalization_gap': round(analysis.generalization_gap, 4),
            'eval_improvement_rate': round(analysis.eval_improvement_rate, 4),
            'eval_stability': round(analysis.eval_stability, 4),
        }
