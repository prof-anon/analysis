import numpy as np
from utils import *
from sim import *

def scenario_paper():
    experiment = Experiment(param_type=max_demand_type, params=[0.25, .5, .75, 1, 2, 4, 8, 16], users_range=list(range(20,101)), num_iters=1000, file_dir="data/paper")

    experiment.run_all(all_stats=True, save_state=True)
    experiment.print_stats(stat_names=[user_util_stat])
    experiment.plot_stats_bar(stat_names=[user_util_stat], by="param", merge=True)   

if __name__ == "__main__":
    # scenario_paper()
    make_paper_plots_bar(file_location='data/paper/', output_file="figure/paper_plot.png")