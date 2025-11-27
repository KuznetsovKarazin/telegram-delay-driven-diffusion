# Telegram Delay-Driven Diffusion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the code and analysis pipeline for a study of information diffusion in Telegram. The project combines large-scale data processing, cascade reconstruction, delay distribution modeling, and competing diffusion analysis. 
The code is designed to work with the Pushshift Telegram dataset (`accounts.ndjson.zst`, `channels.ndjson.zst`, `messages.ndjson.zst`) and reproduces the main empirical results used in our article.

---

## 1. Project goals


- Extract and summarize large-scale statistics of Telegram channels, accounts, and messages.

- Reconstruct forwarding cascades and study their sizes, temporal spans, and depth.

- Model forwarding delays using parametric distributions (Weibull, Lognormal).

- Validate a delay-driven star diffusion model at per-cascade and pooled levels.

- Analyze (the near absence of) competing diffusion between cascades.

The code focuses on transparency and reproducibility: each step produces both figures and CSV tables that can be directly used in a research article.


---


## 2. Repository structure

- `README.md` - this file.

- `LICENSE` - open-source license for the code.

- `requirements.txt` - Python dependencies.

- `.gitignore` - ignore rules for local data and large files.



Main scripts (run sequentially):

1. `analysis\_telegram\_pushshift.py` Global statistics and basic dataset description.

2. `extract\_forwarding\_cascades.py` Extraction of largest forwarding cascades and basic cascade-level summaries.

3. `prepare\_delays\_for\_model.py` Preparation of delay curves for parametric modeling.

4. `fit\_delay\_distribution.py` Fitting Weibull and Lognormal models to cascading delays.

5. `simulate\_star\_delay\_model.py` Validation of the delay-driven star diffusion model (per-cascade and pooled).

6. `simulate\_competing\_diffusion.py` Analysis of competing diffusion between overlapping cascades.


Folders created by the pipeline:


- `Data/` - raw Pushshift Telegram dataset (not tracked by git).

- `Results/` - all generated tables and figures (not tracked by git).

* `Results/figures/`

* `Results/tables/`

* `Results/cascades/`

* `Results/model\_inputs/`

* `Results/delay\_fit/`

* `Results/star\_model\_sim/`

* `Results/competing\_diffusion/`


---


## 3. Requirements



- Python 3.10+ (tested on 64-bit Windows).

- CPU: multi-core is recommended.  

- RAM: at least 32 GB; 64 GB or more is strongly recommended.

- Disk: at least 150 GB free space for:

* original Pushshift files,

* intermediate compressed CSV/NDJSON,

* generated results.



Install Python dependencies into a virtual environment:



```bash

python -m venv env

source env/bin/activate          # Linux / macOS

# or

env\Scripts\activate             # Windows PowerShell / CMD


pip install -r requirements.txt
```

## 4. Data

Place the original Pushshift Telegram files in the Data/ directory:

- `Data/accounts.ndjson.zst`

- `Data/channels.ndjson.zst`

- `Data/messages.ndjson.zst`

The scripts expect this directory layout by default. You can override
paths with --data_dir and other CLI arguments.

## 5. Step-by-step pipeline

Below we show the recommended sequence of scripts and their main outputs.

### 5.1 Global dataset analysis

Script: `analysis_telegram_pushshift.py`

This step computes global statistics and high-level plots for the
Telegram dataset: number of channels, accounts, messages, share of
forwarded messages, temporal activity, and message distribution per
channel.

Example:

```bash
python analysis_telegram_pushshift.py \
  --data_dir Data \
  --out_dir Results
```

Key outputs:

`Results/tables/global_summary.csv`

`Results/tables/messages_per_channel.csv`

`Results/tables/messages_by_month.csv`

`Results/tables/channels_created_by_month.csv`

`Results/figures/*` (histograms and time-series plots)

These tables and figures can be used directly in the dataset description
section of the article.

### 5.2 Forwarding cascades extraction

Script: `extract_forwarding_cascades.py`

This step reconstructs forwarding cascades from the messages dataset.
It identifies original messages and all their forwards, builds edges
between source and forwarded messages, and selects the largest cascades.

Example:
```bash
python extract_forwarding_cascades.py \
  --data_dir Data \
  --out_dir Results/cascades \
  --top_k 5000 \
  --min_size 50
```

Key outputs:

`Results/cascades/edges_topK.csv.gz` - edges in selected cascades.

`Results/cascades/nodes_topK.csv.gz` - nodes in selected cascades.

`Results/cascades/cascades_summary_topK.csv`- per-cascade summary.

`Results/cascades/trees_summary_topK.csv` - depth and branching stats.

`Results/cascades/figures/*` - histograms of sizes, delays, depths.

These files describe cascade sizes, temporal spans, delay distributions,
and confirm that most cascades are shallow (star-shaped).

### 5.3 Preparing delays for modeling

Script: `prepare_delays_for_model.py`

This step converts raw edge-level delays into regular time series suitable for parametric modeling: cumulative forwards C(t) and forwarding rates λ(t). Cascades are binned in fixed time intervals.

Example:
```bash
python prepare_delays_for_model.py \
  --edges Results/cascades/edges_topK.csv.gz \
  --out_dir Results/model_inputs \
  --bin_hours 6 \
  --plot_n 48
```

Key outputs:

Results/model_inputs/model_inputs_summary.csv - core statistics per cascade (size, temporal span, mean/median delays, T10/T50/T90, peak speed).

`Results/model_inputs/curves/<cascade_id>_C.csv` - C(t) curves.

`Results/model_inputs/curves/<cascade_id>_lambda.csv` - λ(t) curves.

`Results/model_inputs/figures/*` - example plots of C(t) and λ(t).

These curves are the empirical basis for fitting delay distributions.

### 5.4 Fitting delay distributions

Script: `fit_delay_distribution.py`

For each cascade, this step fits Weibull and Lognormal distributions to the observed delays and performs model selection via AIC. It also fits a pooled distribution across all cascades.

Example:
```bash
python fit_delay_distribution.py \
  --edges Results/cascades/edges_topK.csv.gz \
  --model_inputs Results/model_inputs/model_inputs_summary.csv \
  --out_dir Results/delay_fit
```

Key outputs:

`Results/delay_fit/per_cascade_fit.csv` - parameters and AIC scores
per cascade; best family per cascade.

`Results/delay_fit/pooled_fit.json` - pooled Weibull and Lognormal
parameters; global AIC comparison.

`Results/delay_fit/figures/*` - CDFs and histograms with fitted curves.

These results show that a sub-Weibull distribution (shape k < 1) captures global delay patterns in Telegram.

### 5.5 Star diffusion model validation

Script: `simulate_star_delay_model.py`

This step validates the delay-driven star diffusion model. For each cascade, it predicts C(t) and λ(t) using the fitted parameters and compares them to the empirical curves.

Example:
```bash
python simulate_star_delay_model.py \
  --curves_dir Results/model_inputs/curves \
  --fit_csv Results/delay_fit/per_cascade_fit.csv \
  --pooled_json Results/delay_fit/pooled_fit.json \
  --out_dir Results/star_model_sim \
  --mode per_cascade_best \
  --plot_n 48
```

Key outputs:

`Results/star_model_sim/sim_results.csv` - per-cascade errors
(RMSE/MAE for C(t) and λ(t)) and parameters actually used.

`Results/star_model_sim/figures/`:

`<cascade_id>_C_pred_vs_obs.png`

`<cascade_id>_lambda_pred_vs_obs.png`

`pooled_C_pred_vs_obs.png`

`pooled_lambda_pred_vs_obs.png`

These figures demonstrate that the Weibull-based star model fits organic cascades very well, while highlighting deviations due to administrative bursts.

### 5.6 Competing diffusion analysis

Script: `simulate_competing_diffusion.py`

Finally, we explore whether cascades compete for attention. The script selects pairs of overlapping cascades, builds competing intensities based on delay-driven λ(t), and compares predicted and observed competing C(t) curves. 

Example:
```bash
python simulate_competing_diffusion.py \
  --curves_dir Results/model_inputs/curves \
  --fit_csv Results/delay_fit/per_cascade_fit.csv \
  --out_dir Results/competing_diffusion \
  --n_pairs 30 \
  --overlap_days 120 \
  --min_size 80 \
  --grid_points 300 \
  --alpha 1.0
```

Key outputs:

`Results/competing_diffusion/competing_pairs.csv` - list of analyzed
cascade pairs, overlap, winners, and intersection times.

`Results/competing_diffusion/figures/`:

`pair_<A>__<B>_pred.png`

`pair_<A>__<B>_obs.png`

`pair_<A>__<B>_pred_vs_obs.png`

The empirical results show that direct competition is rare: one cascade typically dominates while the other receives little attention.

## 6. Reproducibility and usage notes

All scripts are pure Python and operate on CSV/NDJSON/ZST files.

The code was designed to run on a single powerful workstation (e.g., AMD Ryzen 7, 64 GB RAM). Most steps are CPU-bound but can take several hours due to dataset size.

Data directories (Data, Results) are excluded from version control to keep the repository lightweight.

## 7. Citation

If you use this code or analysis in your research, please cite the corresponding article (reference to be added when available).

### 8. Acknowledgments

- Baumgartner, J., Zannettou, S., Squire, M. and Blackburn, J. for providing the OThe Pushshift Telegram Dataset (https://zenodo.org/records/3607497)
- All contributors and users of this system

### 9. Contact

For questions or collaboration opportunities:
- **Email**: oleksandr.o.kuznetsov@gmail.com

---

