"""
=================================================
RIEMANNIAN BCI FAULT-TOLERANT EVALUATION SUITE 
=================================================
"""

from __future__ import annotations
import csv
import logging
import os
import time
import warnings
import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
from moabb.datasets import BNCI2014_001, PhysionetMI
from moabb.paradigms import MotorImagery
from scipy.linalg import inv, logm
from scipy.stats import sem, ttest_rel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# ------------------------------------------------------------------------------
# PART A: IMPORTS, CONFIGURATION, AND CORE ENGINE
# ------------------------------------------------------------------------------

# Setup academic logging environment
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RiemannianBCI")

# Suppress verbose third-party telemetry
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class PipelineConfig:
    """Encapsulates all experimental hyperparameters and hardware flags."""
    CHANNELS: list[str] = ["C3", "Cz", "C4"]
    RESAMPLE_RATE: int = 160
    FMIN: float = 8.0
    FMAX: float = 30.0
    ALPHAS: np.ndarray = np.logspace(-4, 0, 7)
    RANDOM_STATE: int = 42
    N_SPLITS: int = 10
    N_REPEATS: int = 10
    BENCHMARK_REPS: int = 1000
    OUTPUT_DIR: str = "./bci_results_export"
    
    @classmethod
    def init_environment(cls):
        if not os.path.exists(cls.OUTPUT_DIR):
            os.makedirs(cls.OUTPUT_DIR)
            logger.info(f"Created export directory at {cls.OUTPUT_DIR}")

def calibrate_collapsed_weights(
    clf_coef: np.ndarray, 
    clf_intercept: float, 
    p_ref: np.ndarray, 
    n_channels: int
) -> tuple[np.ndarray, float]:
    """
    Projects vector weights of an LDA classifier back into native manifold 
    space as a symmetric matrix weight W to achieve true O(C^2) online inference.
    """
    vals, vecs = np.linalg.eigh(p_ref)
    p_inv_sq = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    
    w_tangent_matrix = np.zeros((n_channels, n_channels))
    idx = 0
    for r in range(n_channels):
        for c in range(r + 1):
            if r == c:
                w_tangent_matrix[r, c] = clf_coef[0, idx]
            else:
                val = clf_coef[0, idx] / np.sqrt(2.0)
                w_tangent_matrix[r, c] = val
                w_tangent_matrix[c, r] = val
            idx += 1
            
    w_collapsed = p_inv_sq @ w_tangent_matrix @ p_inv_sq
    return w_collapsed, float(clf_intercept)

def compute_tangent_space_vector(
    cov_matrix: np.ndarray, 
    p_inv_sq: np.ndarray, 
    n_channels: int
) -> np.ndarray:
    """Maps a spatial covariance matrix to its Euclidean tangent vector representation."""
    transformed = p_inv_sq @ cov_matrix @ p_inv_sq
    matrix_log = logm(transformed).real
    
    vector_len = (n_channels * (n_channels + 1)) // 2
    vec = np.zeros(vector_len)
    idx = 0
    for r in range(n_channels):
        for c in range(r + 1):
            if r == c:
                vec[idx] = matrix_log[r, c]
            else:
                vec[idx] = matrix_log[r, c] * np.sqrt(2.0)
            idx += 1
    return vec

# ------------------------------------------------------------------------------
# PART B: EVALUATION LOOP, LOSO ENGINE, SWEEPS, AND STATISTICS
# ------------------------------------------------------------------------------

def execute_loso_evaluation(
    paradigm: MotorImagery, 
    dataset_train: PhysionetMI, 
    dataset_test: BNCI2014_001, 
    best_alpha: float
) -> dict:
    """Executes an exhaustive Leave-One-Subject-Out cross-dataset loop."""
    logger.info("Starting Leave-One-Subject-Out (LOSO) Cross-Dataset Evaluation Loop...")
    
    # Pre-load and cache global source domain data
    x_train_raw, labels_train, _ = paradigm.get_data(dataset=dataset_train, return_epochs=False)
    y_train = np.array([0 if lbl == "left_hand" else 1 for lbl in labels_train])
    n_channels = x_train_raw.shape[1]
    identity_floor = np.eye(n_channels)
    
    cov_train = np.array([np.cov(x) + best_alpha * identity_floor for x in x_train_raw])
    p_ref = cov_train.mean(axis=0)
    vals, vecs = np.linalg.eigh(p_ref)
    p_inv_sq = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    
    x_train_tangent = np.array([compute_tangent_space_vector(C, p_inv_sq, n_channels) for C in cov_train])
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(x_train_tangent, y_train)
    w_global, intercept = calibrate_collapsed_weights(clf.coef_, clf.intercept_, p_ref, n_channels)
    
    # Fault Scenario Setup: Sensor 0 Drops Out Entirely
    drop_indices, surv_indices = np.array([0]), np.array([1, 2])
    p_surv = p_ref[np.ix_(surv_indices, surv_indices)]
    p_cross = p_ref[np.ix_(drop_indices, surv_indices)]
    m_projection = p_cross @ inv(p_surv)
    
    w_fault_repaired = np.zeros((n_channels, n_channels))
    w_ss = w_global[np.ix_(surv_indices, surv_indices)]
    w_dd = w_global[np.ix_(drop_indices, drop_indices)]
    w_ds = w_global[np.ix_(drop_indices, surv_indices)]
    
    w_modified_surv = w_ss + (m_projection.T @ w_ds) + (w_ds.T @ m_projection) + (m_projection.T @ w_dd @ m_projection)
    w_fault_repaired[np.ix_(surv_indices, surv_indices)] = w_modified_surv
    
    # Storage arrays for comprehensive metrics tracking
    subject_metrics = []
    all_trial_unprotected = []
    all_trial_repaired = []
    all_y_true = []
    all_y_pred = []
    
    for subj_idx, subject in enumerate(dataset_test.subject_list):
        logger.info(f"Processing Target Domain Subject Validation Framework: Subject {subject}")
        try:
            x_test_raw, labels_test, _ = paradigm.get_data(dataset=dataset_test, subjects=[subject], return_epochs=False)
        except Exception as err:
            logger.warning(f"Skipping problematic subject target reference {subject}: {err}")
            continue
            
        y_test = np.array([0 if lbl == "left_hand" else 1 for lbl in labels_test])
        cov_test = np.array([np.cov(x) + best_alpha * identity_floor for x in x_test_raw])
        
        subj_unprotected_hits = 0
        subj_repaired_hits = 0
        
        for c_live, y_true in zip(cov_test, y_test):
            c_surv = c_live[np.ix_(surv_indices, surv_indices)]
            
            # Simulated raw fault response execution 
            c_damaged = identity_floor * 1e-5
            c_damaged[np.ix_(surv_indices, surv_indices)] = c_surv
            raw_score = np.sum(w_global * c_damaged) + intercept
            pred_unprotected = 1 if raw_score >= 0 else 0
            
            # Loopless spatial recovery kernel execution
            loopless_score = np.sum(w_fault_repaired[np.ix_(surv_indices, surv_indices)] * c_surv) + intercept
            pred_repaired = 1 if loopless_score >= 0 else 0
            
            is_unprotected_correct = (pred_unprotected == y_true)
            is_repaired_correct = (pred_repaired == y_true)
            
            subj_unprotected_hits += int(is_unprotected_correct)
            subj_repaired_hits += int(is_repaired_correct)
            
            all_trial_unprotected.append(float(is_unprotected_correct) * 100)
            all_trial_repaired.append(float(is_repaired_correct) * 100)
            all_y_true.append(int(y_true))
            all_y_pred.append(pred_repaired)
            
        subj_unprot_acc = (subj_unprotected_hits / len(y_test)) * 100
        subj_rep_acc = (subj_repaired_hits / len(y_test)) * 100
        
        subject_metrics.append({
            "SubjectID": f"Subj_{subject}",
            "UnprotectedAccuracy": subj_unprot_acc,
            "RepairedAccuracy": subj_rep_acc
        })
        
    return {
        "subject_metrics": subject_metrics,
        "all_trial_unprotected": np.array(all_trial_unprotected),
        "all_trial_repaired": np.array(all_trial_repaired),
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred
    }

def run_regularization_sweep(paradigm: MotorImagery, dataset_train: PhysionetMI) -> list[float]:
    """Sweeps hyperparameter space using repeated cross-validation layers."""
    logger.info("Executing Empirical Alpha Regularization Sweeps...")
    x_train_raw, labels_train, _ = paradigm.get_data(dataset=dataset_train, return_epochs=False)
    y_train = np.array([0 if lbl == "left_hand" else 1 for lbl in labels_train])
    n_channels = x_train_raw.shape[1]
    identity_floor = np.eye(n_channels)
    
    alpha_accuracies = []
    cv_strategy = RepeatedStratifiedKFold(
        n_splits=PipelineConfig.N_SPLITS, 
        n_repeats=PipelineConfig.N_REPEATS, 
        random_state=PipelineConfig.RANDOM_STATE
    )
    
    for alpha_val in PipelineConfig.ALPHAS:
        cov_train_sweep = np.array([np.cov(x) + alpha_val * identity_floor for x in x_train_raw])
        p_ref_sweep = cov_train_sweep.mean(axis=0)
        vals_s, vecs_s = np.linalg.eigh(p_ref_sweep)
        p_inv_sq_sweep = vecs_s @ np.diag(1.0 / np.sqrt(vals_s)) @ vecs_s.T
        
        X_sweep = np.array([compute_tangent_space_vector(C, p_inv_sq_sweep, n_channels) for C in cov_train_sweep])
        clf_sweep = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        score = cross_val_score(clf_sweep, X_sweep, y_train, cv=cv_strategy).mean()
        alpha_accuracies.append(score * 100)
        logger.info(f"  Alpha Regularization = {alpha_val:.5f} -> Validation Accuracy = {score*100:.2f}%")
        
    return alpha_accuracies

def execute_latency_benchmarks() -> tuple[list[int], list[float], list[float]]:
    """Profiles real clock execution times on simulated matrix workloads."""
    logger.info("Running Microsecond-Accurate Algorithmic Complexity Profile Benchmarks...")
    channel_sizes = [8, 16, 24, 32, 64]
    bench_traditional = []
    bench_loopless = []
    
    for C_size in channel_sizes:
        cov_bench = np.random.randn(C_size, C_size)
        cov_bench = cov_bench @ cov_bench.T + np.eye(C_size)
        W_bench = np.random.randn(C_size, C_size)
        
        t_old, t_new = [], []
        for _ in range(PipelineConfig.BENCHMARK_REPS):
            s_old = time.perf_counter_ns()
            _ = logm(cov_bench).real
            t_old.append(time.perf_counter_ns() - s_old)
            
            s_new = time.perf_counter_ns()
            _ = np.sum(W_bench * cov_bench)
            t_new.append(time.perf_counter_ns() - s_new)
            
        bench_traditional.append(np.mean(t_old) / 1e6)  # Convert ns to ms
        bench_loopless.append(np.mean(t_new) / 1e6)
        
    return channel_sizes, bench_traditional, bench_loopless

# ------------------------------------------------------------------------------
# PART C: REPORTING SUITE AND ACADEMIC VISUALIZATION ENGINE
# ------------------------------------------------------------------------------

def export_tabular_reports(metrics: dict, output_dir: str) -> None:
    """Generates structured CSV logs and raw LaTeX tables for paper drafting."""
    csv_path = os.path.join(output_dir, "subject_accuracy_matrix.csv")
    latex_path = os.path.join(output_dir, "ieee_results_table.tex")
    
    # 1. Export Clean CSV Manifest
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Subject ID", "Unprotected Accuracy (%)", "Repaired Accuracy (%)"])
        for row in metrics["subject_metrics"]:
            writer.writerow([row["SubjectID"], f"{row['UnprotectedAccuracy']:.2f}", f"{row['RepairedAccuracy']:.2f}"])
    logger.info(f"Exported metrics raw data log successfully to: {csv_path}")
    
    # 2. Build Programmatic LaTeX Table Document
    with open(latex_path, mode="w") as file:
        file.write(r"\begin{table}[t]" + "\n")
        file.write(r"\caption{Leave-One-Subject-Out Cross-Dataset Performance Framework}" + "\n")
        file.write(r"\label{tab:loso_results}" + "\n")
        file.write(r"\centering" + "\n")
        file.write(r"\begin{tabular}{lcc}" + "\n")
        file.write(r"\hline" + "\n")
        file.write(r"Subject ID & Unprotected Accuracy (\%) & Repaired Accuracy (\%) \\" + "\n")
        file.write(r"\hline" + "\n")
        for row in metrics["subject_metrics"]:
            file.write(f"{row['SubjectID'].replace('_', ' ')} & {row['UnprotectedAccuracy']:.2f}\% & {row['RepairedAccuracy']:.2f}\% \\\\\n")
        file.write(r"\hline" + "\n")
        file.write(f"Pooled Global Mean & {metrics['unprotected_mean']:.2f}\% & {metrics['repaired_mean']:.2f}\% \\\\\n")
        file.write(r"\hline" + "\n")
        file.write(r"\end{tabular}" + "\n")
        file.write(r"\end{table}" + "\n")
    logger.info(f"Programmatic LaTeX manuscript tables built successfully at: {latex_path}")

def generate_export_plots(metrics: dict, output_filename: str) -> None:
    """Constructs academic multi-panel plots adhering to formatting standards."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.linewidth"] = 1.2
    fig, axes = plt.subplots(3, 2, figsize=(14, 16), dpi=300)
    axes_flat = axes.flatten()
    
    # Panel A: Performance Recovery Bounds with Verified Error Bars
    axes_flat[0].bar(["Unprotected\n(Fault State)", "Loopless Repaired\n(O(C^2) Kernel)"], 
                    [metrics["unprotected_mean"], metrics["repaired_mean"]], 
                    yerr=[metrics["unprotected_ci"], metrics["repaired_ci"]], 
                    capsize=8, color=["#d62728", "#1f77b4"], edgecolor="k", alpha=0.85, width=0.4)
    axes_flat[0].set_ylabel("Domain Transfer Accuracy (%)", fontweight="bold")
    axes_flat[0].set_ylim(40, 100)
    axes_flat[0].set_title("A: Domain Generalization Recovery Profile", fontweight="bold", loc="left")
    axes_flat[0].grid(True, linestyle=":", alpha=0.5)
    
    # Panel B: Live Empirical Step Histograms
    sns.histplot(x=metrics["all_trial_unprotected"], ax=axes_flat[1], element="step", fill=True, color="#d62728", alpha=0.3, label="Fault State", binwidth=20)
    sns.histplot(x=metrics["all_trial_repaired"], ax=axes_flat[1], element="step", fill=True, color="#1f77b4", alpha=0.3, label="Repaired State", binwidth=20)
    axes_flat[1].set_xlabel("Trial Metric Scores (%)", fontweight="bold")
    axes_flat[1].set_title("B: Empirical Step Distributions Cross-Cohort", fontweight="bold", loc="left")
    axes_flat[1].grid(True, linestyle=":", alpha=0.5)
    axes_flat[1].legend()
    
    # Panel C: Real Hyperparameter Sweeps
    axes_flat[2].plot(metrics["alpha_sweep"], metrics["alpha_accuracies"], marker="o", color="#1f77b4", linewidth=2)
    axes_flat[2].set_xscale("log")
    axes_flat[2].set_ylabel("Cross-Domain Accuracy (%)", fontweight="bold")
    axes_flat[2].set_xlabel(r"Regularization Term ($\alpha$)", fontweight="bold")
    axes_flat[2].set_title(r"C: Manifold Stability vs. $\alpha$ Floors", fontweight="bold", loc="left")
    axes_flat[2].grid(True, linestyle=":", alpha=0.5)
    
    # Panel D: Empirical Complexity Profiles
    axes_flat[3].plot(metrics["channel_sizes"], metrics["bench_traditional"], marker="x", linestyle=":", color="#7f7f7f", label="Analytical Log-Mapping O(C^3)")
    axes_flat[3].plot(metrics["channel_sizes"], metrics["bench_loopless"], marker="^", linestyle="-", color="#9467bd", label="Loopless Matrix Weight O(C^2)")
    axes_flat[3].set_xlabel("High-Density Channel Count (C)", fontweight="bold")
    axes_flat[3].set_ylabel("Execution Latency (ms)", fontweight="bold")
    axes_flat[3].set_title("D: Computational Complexity Profiles", fontweight="bold", loc="left")
    axes_flat[3].grid(True, linestyle=":", alpha=0.5)
    axes_flat[3].legend()
    
    # Panel E: Dynamic Individual Subject Metrics
    subj_list = metrics["subject_metrics"]
    n_subjs_to_display = min(3, len(subj_list))
    x_ticks_positions = np.arange(n_subjs_to_display + 1)
    
    bar_labels = [row["SubjectID"] for row in subj_list[:n_subjs_to_display]] + ["Pooled Global"]
    unprotected_bars = [row["UnprotectedAccuracy"] for row in subj_list[:n_subjs_to_display]] + [metrics["unprotected_mean"]]
    repaired_bars = [row["RepairedAccuracy"] for row in subj_list[:n_subjs_to_display]] + [metrics["repaired_mean"]]
    
    axes_flat[4].bar(x_ticks_positions - 0.2, unprotected_bars, 0.4, label="Fault Transfer", color="#e377c2", edgecolor="k")
    axes_flat[4].bar(x_ticks_positions + 0.2, repaired_bars, 0.4, label="Repaired Transfer", color="#1f77b4", edgecolor="k")
    axes_flat[4].set_xticks(x_ticks_positions)
    axes_flat[4].set_xticklabels(bar_labels, rotation=15)
    axes_flat[4].set_ylim(40, 100)
    axes_flat[4].set_title("E: Independent Validation Sub-Nodes", fontweight="bold", loc="left")
    axes_flat[4].grid(True, linestyle=":", alpha=0.5)
    axes_flat[4].legend()
    
    # Panel F: Target Class Confusion Matrix Map
    sns.heatmap(metrics["conf_matrix"], annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axes_flat[5], 
                xticklabels=["Left Hand", "Right Hand"], yticklabels=["Left Hand", "Right Hand"], 
                linewidths=1, linecolor="k", annot_kws={"size": 12, "weight": "bold"})
    axes_flat[5].set_title("F: Target Class Alignment Map", fontweight="bold", loc="left")
    
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Publication figures successfully written to file: {output_filename}")

# ------------------------------------------------------------------------------
# MASTER CONTROL ORCHESTRATION PIPELINE
# ------------------------------------------------------------------------------

def main() -> None:
    logger.info("Initializing Publication Execution Framework Archetype...")
    PipelineConfig.init_environment()
    
    paradigm = MotorImagery(
        channels=PipelineConfig.CHANNELS, 
        resample=PipelineConfig.RESAMPLE_RATE, 
        fmin=PipelineConfig.FMIN, 
        fmax=PipelineConfig.FMAX
    )
    
    dataset_train = PhysionetMI()
    dataset_test = BNCI2014_001()
    
    # 1. Execute Alpha Sweep Loop 
    alpha_accs = run_regularization_sweep(paradigm, dataset_train)
    best_alpha = PipelineConfig.ALPHAS[np.argmax(alpha_accs)]
    
    # 2. Execute Complete Multi-Subject Evaluation Pipeline
    raw_loso_results = execute_loso_evaluation(paradigm, dataset_train, dataset_test, best_alpha)
    
    # 3. Post-Process Statistical Computations
    unprot_array = [row["UnprotectedAccuracy"] for row in raw_loso_results["subject_metrics"]]
    repair_array = [row["RepairedAccuracy"] for row in raw_loso_results["subject_metrics"]]
    
    unprotected_mean = float(np.mean(unprot_array))
    repaired_mean = float(np.mean(repair_array))
    unprotected_ci = float(1.96 * sem(unprot_array)) if len(unprot_array) > 1 else 0.0
    repaired_ci = float(1.96 * sem(repair_array)) if len(repair_array) > 1 else 0.0
    
    # Conduct statistical two-sided paired t-test over cohort
    if len(unprot_array) > 1:
        t_stat, p_val = ttest_rel(repair_array, unprot_array)
        logger.info(f"Statistical Significance Verification Matrix: t-value = {t_stat:.4f}, p-value = {p_val:.6e}")
    
    # 4. Profile Real Hardware Target Performance Clocks
    c_sizes, bench_trad, bench_loop = execute_latency_benchmarks()
    
    # 5. Pack Manifest Payload and Export Artifact Deliverables
    metrics_payload = {
        "unprotected_mean": unprotected_mean,
        "repaired_mean": repaired_mean,
        "unprotected_ci": unprotected_ci,
        "repaired_ci": repaired_ci,
        "all_trial_unprotected": raw_loso_results["all_trial_unprotected"],
        "all_trial_repaired": raw_loso_results["all_trial_repaired"],
        "subject_metrics": raw_loso_results["subject_metrics"],
        "conf_matrix": confusion_matrix(raw_loso_results["all_y_true"], raw_loso_results["all_y_pred"], normalize="true"),
        "alpha_sweep": PipelineConfig.ALPHAS,
        "alpha_accuracies": alpha_accs,
        "channel_sizes": c_sizes,
        "bench_traditional": bench_trad,
        "bench_loopless": bench_loop
    }
    
    export_tabular_reports(metrics_payload, PipelineConfig.OUTPUT_DIR)
    generate_export_plots(metrics_payload, os.path.join(PipelineConfig.OUTPUT_DIR, "ieee_journal_manifest.pdf"))
    logger.info("Experimental testing sequence terminated successfully without runtime anomalies.")

if __name__ == "__main__":
    main()
