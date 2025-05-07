"""
summary.py
----------------
Convenience utilities that operate on the results of spike detection

Authors : Daria Kleeva
Email: dkleeva@gmail.com
"""

from __future__ import annotations
import math
from typing import Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap

# ------------------------------------------------------------------
# 1. Epoch construction
# ------------------------------------------------------------------

def make_epochs(
    raw: mne.io.BaseRaw,
    results: dict,
    *,
    tmin: float = -0.05,
    tmax: float = 0.1,
    baseline: Tuple[float, float] | None = None,
    preload: bool = True,
    event_id: int = 1,
) -> mne.Epochs:
    """
    Create `mne.Epochs` object around detected spike events.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Input raw data (should be restricted to relevant channels).
    results : dict
        Dictionary returned by `detect_spikes_fpcm`, must contain 'events'.
    tmin, tmax : float
        Epoch start and end relative to spike peak (in seconds).
    baseline : tuple or None
        Baseline correction window. Passed to `mne.Epochs`.
    preload : bool
        Whether to preload the data into memory.
    event_id : int
        Event code to assign to all spikes (default: 1).

    Returns
    -------
    epochs : mne.Epochs
        Extracted epochs around each detected spike.
    """
    events = results["events"].copy()
    event_dict = {"spike": event_id}
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=preload,
        picks="data",
    )
    return epochs

# ------------------------------------------------------------------
# 2. Average joint plots
# ------------------------------------------------------------------

def plot_average_joint(
    epochs: mne.Epochs,
    *,
    title: str | None = "Average spike",
    **kwargs,
):
    """
    Plot joint topographic and time-series view of the average spike.

    Wrapper for `epochs.average().plot_joint` with simplified interface.

    Parameters
    ----------
    epochs : mne.Epochs
        Spike-centered epochs.
    title : str or None
    Title to display above the figure.
    **kwargs :
        Passed to `Evoked.plot_joint`.

    Returns
    -------
    evoked : mne.Evoked
        Averaged evoked response for the spikes.
    """
    evoked = epochs.average()
    evoked.plot_joint(title=title, **kwargs)
    return evoked

# ------------------------------------------------------------------
# 3. Grid of topographies with hit‑sensor markers
# ------------------------------------------------------------------

def plot_spike_topographies(
    raw: mne.io.BaseRaw,
    results: dict,
    *,    
    n_cols: int = 4,
    cmap: str = "RdBu_r",
    marker_size: int = 20,      # in points (≈ size/2 of scatter earlier)
    marker_color: str = "yellow",
):
    """
    Plot 2D topographic map for each detected spike at its peak.

    For each spike, this function plots the spatial distribution of the 
    field at time zero and overlays markers on channels that contributed 
    to detection ("hits").

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object used in detection.
    results : dict
        Output of `detect_spikes_fpcm` (must contain 'peaks_samples' and 'hits').
    n_cols : int
        Number of topomaps per row in the grid.
    cmap : str
        Colormap for the topomaps.
    marker_size : int
        Size of the sensor hit marker.
    marker_color : str
        Fill color for hit sensor markers.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing all spike topographies.
    """

    peaks = results["peaks_samples"]
    hit_mask = results["hits"]                

    info_picked = raw.info

    n_peaks = len(peaks)
    n_rows = math.ceil(n_peaks / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        squeeze=False,
    )

    mask_params = dict(
        marker='o',
        markerfacecolor=marker_color,
        markeredgecolor='black',
        markeredgewidth=2.,
        markersize=marker_size,
    )

    for k, peak in enumerate(peaks):
        r, c = divmod(k, n_cols)
        ax = axes[r][c]

        ev_data = raw._data[:, peak : peak + 1]
        evoked = mne.EvokedArray(ev_data, info_picked, tmin=0.0)

        mask_this = hit_mask[:,k][:, None]  
        ch_type= raw.get_channel_types(picks='data', unique=True)[0]

        evoked.plot_topomap(
            times=0.0,
            ch_type=ch_type,
            cmap=cmap,
            axes=ax,
            show=False,
            mask=mask_this,
            mask_params=mask_params,
            time_format="",   
            colorbar=False
        )
        ax.set_title(f"Spike {k+1}", fontsize=9)

    for ax in axes.flat[n_peaks:]:
        ax.axis("off")

    fig.tight_layout()
    return fig

# ------------------------------------------------------------------
# 4. Spline overlay
# ------------------------------------------------------------------

def overlay_spline_fit_grid(
    raw: mne.io.BaseRaw,
    results: dict,
    *,
    n_cols: int = 3,
    unit: str = "uV",
    scale: float | None = None,
    figsize: tuple[int, int] | None = None,
    data_kw: dict | None = None,
    spline_kw: dict | None = None,
):
    """
    Overlay spline reconstructions on raw traces for all spikes.

    For each detected spike, plots all contributing ("hit") channels with 
    the original signal and its corresponding spline fit, aligned to spike 
    peak. Useful for visually inspecting model quality.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object used in detection.
    results : dict
        Output of `detect_spikes_fpcm` (must contain Bp, coeffs, hits, etc).
    n_cols : int
        Number of subplots per row.
    unit : str
        Units for y-axis label ("uV", "fT", etc).
    scale : float or None
        Rescale data (e.g., 1e6 for µV). If None, guessed from unit.
    figsize : (width, height) or None
        Size of the overall matplotlib figure.
    data_kw : dict or None
        Styling for raw traces (passed to `plot`).
    spline_kw : dict or None
        Styling for spline traces (passed to `plot`).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with one subplot per spike showing signal vs fit.
    """
    if scale is None:
        scale = dict(uV=1e6, fT=1e15, T=1.0).get(unit, 1.0)
    if data_kw   is None:
        data_kw   = dict(color="C0", lw=0.7, alpha=0.8)
    if spline_kw is None:
        spline_kw = dict(color="C3", lw=1.2, ls="--")

    peaks  = results["peaks_samples"]
    hits    = results["hits"].astype(bool)
    Bp      = results["bp_matrix"]
    coeffs  = results["coeffs"]
    T       = results["window_len"]
    sfreq   = raw.info["sfreq"]
    peak_hw_s = results["peak_hw_s"]

    t_vec = (np.arange(-peak_hw_s, T-peak_hw_s) / sfreq) * 1e3   # ms


    n_spk   = len(peaks)
    n_rows  = int(np.ceil(n_spk / n_cols))
    if figsize is None:
        figsize = (n_cols * 3.2, n_rows * 2.4)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=False)
    axes = axes.flatten()

    for idx, (ax, peak) in enumerate(zip(axes, peaks)):
        hit_ch = np.where(hits[:, idx])[0]
        if hit_ch.size == 0:
            ax.axis("off")
            continue
        
        rng = slice(peak - peak_hw_s, peak + T - peak_hw_s)
        if rng.start < 0 or rng.stop > raw.n_times:
            ax.axis("off")
            continue  
            

        for ch in hit_ch:
            x = raw._data[ch, rng] * scale
            ax.plot(t_vec, x, **data_kw)
            spline = (Bp @ coeffs[ch][:, peak+T-peak_hw_s]) * scale  # (T,)
            ax.plot(t_vec, spline, **spline_kw)

        ax.set_title(f"Spike {idx+1},{len(hit_ch)} hits", fontsize=8)
        ax.set_xlabel('ms')
        ax.axvline(0, color="k", lw=.6)
        ax.set_yticks([])
        ax.set_xlim(t_vec[0], t_vec[-1])

    
    for ax in axes[n_spk:]:
        ax.axis("off")

    axes[0].set_ylabel(unit)
    fig.tight_layout()
    return fig
