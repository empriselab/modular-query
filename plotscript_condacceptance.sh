
## Upper plots.
# python experiments/plot_results_grid.py --results_dir experiments/results/20251208_hricondaccept/ --plot_variable 'num_modules' --output_dir experiments/results/20251208_hricondaccept/01_nummodules --pkl_file results_variant_balanced-2_run_02.pkl --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'
# python experiments/plot_results_grid.py --results_dir experiments/results/20251208_hricondaccept/ --plot_variable 'graph_structures' --output_dir experiments/results/20251208_hricondaccept/02_structures --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'
# python experiments/plot_results_grid.py --results_dir experiments/results/20251208_hricondaccept/ --plot_variable 'confidences' --output_dir experiments/results/20251208_hricondaccept/03_confidences --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'
# python experiments/plot_results_grid.py --results_dir experiments/results/20251208_hricondaccept/ --plot_variable 'query_costs' --output_dir experiments/results/20251208_hricondaccept/04_querycosts --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'

## Lower plots.
# python experiments/plot_results_grid.py --results_dir experiments/results/20250929_fixbruteforce/ --plot_variable 'num_modules' --output_dir experiments/results/20250929_fixbruteforce/01_nummodules --pkl_file results_variant_balanced-2_run_02.pkl --run_id '02' --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'
# python experiments/plot_results_grid.py --results_dir experiments/results/20250929_fixbruteforce/ --plot_variable 'graph_structures' --output_dir experiments/results/20250929_fixbruteforce/02_structures --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'
# python experiments/plot_results_grid.py --results_dir experiments/results/20250929_fixbruteforce_varyconfidences/ --plot_variable 'confidences' --output_dir experiments/results/20250929_fixbruteforce_varyconfidences/03_confidences --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'
# python experiments/plot_results_grid.py --results_dir experiments/results/20250929_fixbruteforce/ --plot_variable 'query_costs' --output_dir experiments/results/20250929_fixbruteforce/04_querycosts --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'


## Plots for confidence query/graph query ablation.
# python experiments/plot_results_grid.py --results_dir experiments/results/20251110_exp1/ --plot_variable 'confidences' --output_dir experiments/results/20251110_exp1 --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'
# python experiments/plot_results_grid.py --results_dir experiments/results/20251110_exp2/ --plot_variable 'confidences' --output_dir experiments/results/20251110_exp2 --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'
# python experiments/plot_results_grid.py --results_dir experiments/results/20251110_exp3/ --plot_variable 'confidences' --output_dir experiments/results/20251110_exp3 --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'
# python experiments/plot_results_grid.py --results_dir experiments/results/20251110_finerconfidences_exp1/ --plot_variable 'confidences' --output_dir experiments/results/20251110_finerconfidences_exp1 --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'
# python experiments/plot_results_grid.py --results_dir experiments/results/20251110_finerconfidences_exp2/ --plot_variable 'confidences' --output_dir experiments/results/20251110_finerconfidences_exp2 --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'
# python experiments/plot_results_grid.py --results_dir experiments/results/20251110_finerconfidences_exp3/ --plot_variable 'confidences' --output_dir experiments/results/20251110_finerconfidences_exp3 --fixed_variant balanced-2 --fixed_module_selector 'Graph Query'