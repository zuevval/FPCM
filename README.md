# FPCM (Fast Parametric Curve Matching)
Toolbox for automatic detection and localization of interictal events in MEG/EEG

## Features:
- Automatic spike detection based on mimetic approach and the constrained parametric morphological model
- Robust against noisy high-amplitude transients
- Adaptable to different scales of the patterns

## Modules:
- `src/fpcm_detector.py` — Core algorithm for spike detection using FPCM
- `src/summary.py` — Convenient visualization tools of the results (e.g., topographies of the spikes, overlays with the splines, etc.)
- `src/source_loc.py` — Dipole fitting of the detected spikes
- `src/utils.py` — Auxiliary tools for evaluating performance, etc.

## Example usage
An example script demonstrating the use of the FPCM algorithm is available in `scripts/FPCM demo.ipynb`

You can download the simulated MEG dataset with interictal spikes used in this script from [Google Drive](https://drive.google.com/file/d/1MHGGDDmAxTF1qWi7Og86pMRp1SrorUzq/view?usp=share_link). 

## Citation
If you use this code, please cite the following paper:

*Kleeva, D., Soghoyan, G., Komoltsev, I., Sinkin, M., & Ossadtchi, A. (2022). Fast parametric curve matching (FPCM) for automatic spike detection. Journal of Neural Engineering, 19(3), 036003.* 
