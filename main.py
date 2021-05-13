"""Main script to run on the server"""
import experiments
import macro_experiments
from train.mathqa_graph_generation_train import train

if __name__ == "__main__":
    train()
    # macro_experiments.high_complexity_macro()
    # macro_experiments.different_avg_len_macros(0)
    # macro_experiments.different_avg_len_macros(1)
