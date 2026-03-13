# A Human-in-the-loop Confidence-Aware Failure Recovery Framework for Modular Robot Policies

![workflow](https://github.com/tomsilver/modular-query/actions/workflows/ci.yml/badge.svg)

Simulation framework implementation for the paper [**A Human-in-the-loop Confidence-Aware Failure Recovery Framework for Modular Robot Policies**](https://emprise.cs.cornell.edu/modularhil/).

[Project Website](https://emprise.cs.cornell.edu/modularhil/) | [Paper](https://arxiv.org/abs/2602.10289)

## Requirements

- Python 3.11+
- Tested on MacOS Catalina

## Installation

1. Recommended: create and source a virtualenv.
2. `pip install -e ".[develop]"`
3. `python -m amplpy.modules install coin -q`


## 🚀 Quick Start

To generate the main paper plot (Fig. 4):

**1. Running simulation experiments:**
```bash
python run_experiment.py --variant all_variants
```

**2. Create data directory:**
```bash
mkdir -p experiments/results/[data_dir]
mv experiments/results/* experiments/results/[data_dir]
```

**3. Create dataframes:**
```bash
python pickles_to_df.py --data_dir experiments/results/[data_dir]
```

Note - you will need to run steps 1-3 _4_ times in total, one for each of the configurations
in the `main()` method of `experiments/run_experiment.py`, where the required `[data_dir]` values
are written in the comments.

**Generate main paper plot (Fig. 4):**
```bash
python plot_unified_grid.py --output_dir [output_dir]
```

**Generate appendix plots**

```bash
python plot_appendix_ivs.py --output_dir [output_dir]
python plot_appendix_module_heterogeneity.py --output_dir [output_dir]
```


## 📚 Citation
```bibtex
@inproceedings{banerjee2026modularhil,
      author    = {Banerjee, Rohan and Palempalli, Krishna and Yang, Bohan and Fang, Jiaying and Abdullah, Alif and Silver, Tom and Dean, Sarah and Bhattacharjee, Tapomayukh},
      title     = {A Human-in-the-Loop Confidence-Aware Failure Recovery Framework for Modular Robot Policies},
      booktitle = {Proceedings of the ACM/IEEE International Conference on Human-Robot Interaction (HRI)},
      year      = {2026},
}
```