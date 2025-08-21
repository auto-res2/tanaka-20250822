import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from preprocess import Claim, SyntheticFeatureBuilder, MatryoshkaRetrieverSim, ESCALATION_STEPS
from train import ConsistencyHead, ConformalG


@dataclass
class PlanResult:
    plan: Optional[Dict]
    bound: float
    features: Dict[str, float]
    answered: bool
    abstained: bool
    correct_if_answered: bool
    citations: List[int]
    plan_level: int
    latency_ms: float


class RealityRAGSystem:
    def __init__(self, head: ConsistencyHead, feat_builder: SyntheticFeatureBuilder,
                 conformal_g: ConformalG, retriever: MatryoshkaRetrieverSim,
                 device: str = 'cpu', with_registers: bool = True):
        self.head = head.to(device)
        self.feat_builder = feat_builder
        self.g = conformal_g
        self.retriever = retriever
        self.device = device
        self.with_registers = with_registers

    def _head_predict(self, feat_vec: np.ndarray) -> Tuple[float, np.ndarray]:
        with torch.no_grad():
            x = torch.tensor(feat_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.head(x)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pS = float(probs[0])
        return pS, probs

    def _citations(self, claim: Claim, plan: Dict, seed: int = 0) -> List[int]:
        rng = np.random.RandomState(seed + claim.id)
        base = rng.randn(self.retriever.D_full).astype('float32')
        if claim.label == 0:
            base[:16] += 0.5
        elif claim.label == 2:
            base[16:32] += 0.5
        base = base / (np.linalg.norm(base) + 1e-8)
        ids = self.retriever.search(base, d=plan['dim'], k=min(plan['k'], 5), seed=seed)
        return ids

    def _compute_cost_proxy(self, plan: Dict, M_tokens: int = 8) -> Tuple[int, int, float]:
        retrieval_calls = plan['k'] * (1 + plan['hops']) * plan['ens']
        context_tokens = 256 + int(0.02 * plan['dim'] * plan['k']) + M_tokens
        latency_ms = 5.0 + 0.02 * retrieval_calls + 0.0015 * context_tokens
        return retrieval_calls, context_tokens, latency_ms

    def answer_with_certification(self, claim: Claim, tau: float, seed: int = 0,
                                  micro_ensemble_m: int = 1) -> PlanResult:
        prev_pS = None
        chosen = None
        last_feats = None
        last_bound = 1.0
        
        for plan in ESCALATION_STEPS:
            feat_vec, aux = self.feat_builder.build_features(claim, plan, seed=seed, prev_pS=prev_pS)
            
            pS, probs = self._head_predict(feat_vec)
            
            if micro_ensemble_m > 1:
                pS_vals = [pS]
                for m in range(1, micro_ensemble_m):
                    feat_vec_m, _ = self.feat_builder.build_features(claim, plan, seed=seed+m, prev_pS=prev_pS)
                    pS_m, _ = self._head_predict(feat_vec_m)
                    pS_vals.append(pS_m)
                pS = float(np.mean(pS_vals))
                ensemble_var = float(np.var(pS_vals))
                feat_vec, aux = self.feat_builder.build_features(claim, plan, seed=seed, prev_pS=prev_pS, ensemble_var=ensemble_var)
            
            plan_bucket = (plan['dim'], plan['k'], plan['hops'], tuple(sorted(plan['sources'])), plan['ens'])
            score = 1.0 - pS
            bound = self.g.bound(float(score), plan_bucket)
            
            if bound <= tau:
                chosen = plan
                last_feats = aux
                last_bound = bound
                break
            
            prev_pS = pS
            last_feats = aux
            last_bound = bound
        
        if chosen is None:
            return PlanResult(
                plan=None, bound=last_bound, features=last_feats or {},
                answered=False, abstained=True, correct_if_answered=False,
                citations=[], plan_level=-1, latency_ms=0.0
            )
        
        citations = self._citations(claim, chosen, seed=seed)
        _, _, latency_ms = self._compute_cost_proxy(chosen)
        correct = (claim.label == 0)
        
        return PlanResult(
            plan=chosen, bound=last_bound, features=last_feats or {},
            answered=True, abstained=False, correct_if_answered=correct,
            citations=citations, plan_level=chosen['level'], latency_ms=latency_ms
        )


def save_pdf(fig, filename: str):
    if not filename.endswith('.pdf'):
        filename = filename + '.pdf'
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


def experiment_a_conformal_certification(system: RealityRAGSystem, claims: List[Claim], 
                                       output_dir: str = ".research/iteration1/images") -> Dict:
    print("Running Experiment A: Conformal Certification Accuracy")
    
    taus = [0.05, 0.10, 0.15, 0.20, 0.25]
    results = {tau: {'empirical_risk': [], 'certified_bound': [], 'abstention_rate': []} for tau in taus}
    
    for tau in taus:
        print(f"  Testing tau = {tau}")
        empirical_fails = 0
        total_answered = 0
        total_claims = 0
        bounds = []
        
        for claim in claims[:50]:
            result = system.answer_with_certification(claim, tau, seed=42)
            total_claims += 1
            
            if result.answered:
                total_answered += 1
                if not result.correct_if_answered:
                    empirical_fails += 1
            
            bounds.append(result.bound)
        
        empirical_risk = empirical_fails / max(total_answered, 1)
        abstention_rate = (total_claims - total_answered) / total_claims
        avg_bound = np.mean(bounds)
        
        results[tau]['empirical_risk'].append(empirical_risk)
        results[tau]['certified_bound'].append(avg_bound)
        results[tau]['abstention_rate'].append(abstention_rate)
        
        print(f"    Empirical risk: {empirical_risk:.3f}, Avg bound: {avg_bound:.3f}, Abstention: {abstention_rate:.3f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    tau_vals = list(results.keys())
    emp_risks = [results[tau]['empirical_risk'][0] for tau in tau_vals]
    cert_bounds = [results[tau]['certified_bound'][0] for tau in tau_vals]
    abst_rates = [results[tau]['abstention_rate'][0] for tau in tau_vals]
    
    ax1.plot(tau_vals, emp_risks, 'o-', label='Empirical Risk', linewidth=2)
    ax1.plot(tau_vals, cert_bounds, 's-', label='Certified Bound', linewidth=2)
    ax1.plot(tau_vals, tau_vals, '--', alpha=0.7, label='Target τ')
    ax1.set_xlabel('Risk Target τ')
    ax1.set_ylabel('Risk')
    ax1.set_title('Conformal Certification Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(tau_vals, abst_rates, 'o-', color='red', linewidth=2)
    ax2.set_xlabel('Risk Target τ')
    ax2.set_ylabel('Abstention Rate')
    ax2.set_title('Abstention vs Risk Target')
    ax2.grid(True, alpha=0.3)
    
    save_pdf(fig, f"{output_dir}/experiment_a_conformal_certification.pdf")
    print(f"  Saved plot to {output_dir}/experiment_a_conformal_certification.pdf")
    
    return results


def experiment_b_matryoshka_efficiency(system: RealityRAGSystem, claims: List[Claim],
                                     output_dir: str = ".research/iteration1/images") -> Dict:
    print("Running Experiment B: Matryoshka Retrieval Efficiency")
    
    results = {'dims': [], 'accuracy': [], 'latency': [], 'cost': []}
    
    for plan in ESCALATION_STEPS:
        print(f"  Testing plan level {plan['level']} (dim={plan['dim']}, k={plan['k']})")
        
        correct = 0
        total_latency = 0
        total_cost = 0
        
        for claim in claims[:30]:
            result = system.answer_with_certification(claim, tau=0.15, seed=42)
            if result.answered and result.correct_if_answered:
                correct += 1
            total_latency += result.latency_ms
            _, _, cost = system._compute_cost_proxy(plan)
            total_cost += cost
        
        accuracy = correct / 30
        avg_latency = total_latency / 30
        avg_cost = total_cost / 30
        
        results['dims'].append(plan['dim'])
        results['accuracy'].append(accuracy)
        results['latency'].append(avg_latency)
        results['cost'].append(avg_cost)
        
        print(f"    Accuracy: {accuracy:.3f}, Latency: {avg_latency:.1f}ms, Cost: {avg_cost:.1f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    dims = results['dims']
    
    ax1.plot(dims, results['accuracy'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Matryoshka Dimension')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Retrieval Dimension')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(dims, results['latency'], 's-', color='orange', linewidth=2, markersize=8)
    ax2.set_xlabel('Matryoshka Dimension')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency vs Retrieval Dimension')
    ax2.grid(True, alpha=0.3)
    
    save_pdf(fig, f"{output_dir}/experiment_b_matryoshka_efficiency.pdf")
    print(f"  Saved plot to {output_dir}/experiment_b_matryoshka_efficiency.pdf")
    
    return results


def experiment_c_micro_ensemble_stability(system: RealityRAGSystem, claims: List[Claim],
                                        output_dir: str = ".research/iteration1/images") -> Dict:
    print("Running Experiment C: Micro-Ensemble Stability")
    
    ensemble_sizes = [1, 2, 3, 4, 5]
    results = {'sizes': [], 'variance': [], 'accuracy': []}
    
    for ens_size in ensemble_sizes:
        print(f"  Testing ensemble size {ens_size}")
        
        variances = []
        correct = 0
        
        for claim in claims[:20]:
            predictions = []
            for seed in range(10):
                result = system.answer_with_certification(claim, tau=0.15, seed=seed, micro_ensemble_m=ens_size)
                if result.answered:
                    predictions.append(1.0 if result.correct_if_answered else 0.0)
            
            if predictions:
                variances.append(np.var(predictions))
                correct += np.mean(predictions)
        
        avg_variance = np.mean(variances) if variances else 0.0
        accuracy = correct / 20
        
        results['sizes'].append(ens_size)
        results['variance'].append(avg_variance)
        results['accuracy'].append(accuracy)
        
        print(f"    Variance: {avg_variance:.4f}, Accuracy: {accuracy:.3f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sizes = results['sizes']
    
    ax1.plot(sizes, results['variance'], 'o-', color='green', linewidth=2, markersize=8)
    ax1.set_xlabel('Ensemble Size')
    ax1.set_ylabel('Prediction Variance')
    ax1.set_title('Stability vs Ensemble Size')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(sizes, results['accuracy'], 's-', color='purple', linewidth=2, markersize=8)
    ax2.set_xlabel('Ensemble Size')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Ensemble Size')
    ax2.grid(True, alpha=0.3)
    
    save_pdf(fig, f"{output_dir}/experiment_c_micro_ensemble_stability.pdf")
    print(f"  Saved plot to {output_dir}/experiment_c_micro_ensemble_stability.pdf")
    
    return results


def evaluate_system(head: ConsistencyHead, conformal_g: ConformalG, claims: List[Claim], 
                   retriever: MatryoshkaRetrieverSim, device: str = 'cpu', n_test: int = 100) -> Dict:
    """Main evaluation function running all experiments."""
    print("=== System Evaluation ===")
    
    feat_builder = SyntheticFeatureBuilder()
    system = RealityRAGSystem(head, feat_builder, conformal_g, retriever, device=device)
    
    test_claims = claims[:n_test]
    
    results = {}
    results['experiment_a'] = experiment_a_conformal_certification(system, test_claims)
    results['experiment_b'] = experiment_b_matryoshka_efficiency(system, test_claims)
    results['experiment_c'] = experiment_c_micro_ensemble_stability(system, test_claims)
    
    print("=== All experiments completed ===")
    return results
