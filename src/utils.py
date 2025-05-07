"""
utils.py
----------------
Utility functions for evaluation and analysis of spike detection.

Authors : Daria Kleeva
Email: dkleeva@gmail.com
"""

from __future__ import annotations
import math
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import mne

def compute_performance(
    true_latencies_s: "list[float] | np.ndarray",
    results: dict,
    raw: mne.io.BaseRaw,
    *,
    tolerance_ms: float = 20.0,
    verbose: bool = True,
):
    """
    Compare automatically detected spike latencies with ground-truth labels.

    This function evaluates the quality of spike detection by matching
    the detected latencies (`results['peaks_samples']`) with true spike 
    times (in seconds) within a fixed temporal tolerance window.

    Parameters
    ----------
    true_latencies_s : list of float or np.ndarray
        Ground-truth spike latencies in **seconds** from the start of the Raw object.
    results : dict
        Dictionary returned by `detect_spikes_fpcm`; must contain `'peaks_samples'`.
    raw : mne.io.Raw
        Raw object used for detection; needed for sampling rate and alignment.
    tolerance_ms : float
        Maximum allowed deviation (± in milliseconds) for a detected spike to be 
        considered a match with a ground-truth spike.
    verbose : bool
        If True, prints a table of performance metrics.

    Returns
    -------
    metrics : dict
        Dictionary containing performance metrics:
            - TP (true positives)
            - FP (false positives)
            - FN (false negatives)
            - precision
            - recall
            - f1 (F1 score)
            - FPR_per_min (false positive rate per minute)
    matched_pairs : list of tuple(int, int)
        List of matched spike pairs, where each tuple contains the true and 
        detected spike latencies in **samples** relative to raw data.
    """
    import numpy as np

    sfreq = raw.info["sfreq"]
    first = raw.first_samp
    tol_samp = int(round(tolerance_ms / 1000 * sfreq))

    true_samp = (np.asarray(true_latencies_s) * sfreq).astype(int)
    true_samp -= first                   
    det_samp  = np.asarray(results["peaks_samples"], int)

    true_used = np.zeros_like(true_samp, bool)
    det_used  = np.zeros_like(det_samp,  bool)
    pairs     = []


    for d_i, d in enumerate(det_samp):
        diffs = np.abs(true_samp - d)
        idx   = np.where((diffs <= tol_samp) & (~true_used))[0]
        if idx.size:
            t_i = idx[np.argmin(diffs[idx])]  
            true_used[t_i] = True
            det_used[d_i]  = True
            pairs.append((true_samp[t_i], d))

    TP = np.sum(det_used)
    FP = np.sum(~det_used)
    FN = np.sum(~true_used)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0

    duration_min = raw.n_times / sfreq / 60.0
    FPR_per_min  = FP / duration_min

    metrics = dict(TP=TP, FP=FP, FN=FN,
                   precision=precision, recall=recall,
                   f1=f1, FPR_per_min=FPR_per_min)

    if verbose:
        print("Spike detection performance")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"{k:12s}: {v:.4f}")
            else:
                print(f"{k:12s}: {v}")
        print(f"Matched within ±{tolerance_ms} ms (total matches {len(pairs)})")
    return metrics, pairs