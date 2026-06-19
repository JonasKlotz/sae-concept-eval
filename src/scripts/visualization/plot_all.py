"""Run all main paper figures and the appendix figures actually used in the paper."""
from scripts.visualization.fig_combined_sota_matching_tapas import fig_combined_sota_matching_tapas
from scripts.visualization.fig_matching_over_dictsizes import fig_matching_over_dictsizes
from scripts.visualization.fig_tapas_over_dictsizes import fig_tapas_over_dictsizes
from scripts.visualization.fig_matching_tapas_correlation import fig_matching_tapas_correlation

from scripts.visualization.appendix.appendix_unnormalized_matching_dict_size import fig_unnormalized_matching_dict_size
from scripts.visualization.appendix.appendix_coalition_size_sweep import fig_coalition_size_sweep
from scripts.visualization.appendix.appendix_topk_sparsity_sweep import fig_sparsity_sweep



def main():
    # --- Main paper ---
    fig_combined_sota_matching_tapas()
    fig_matching_over_dictsizes()
    fig_tapas_over_dictsizes()
    fig_matching_tapas_correlation()

    # --- Appendix ---
    fig_unnormalized_matching_dict_size()
    fig_coalition_size_sweep()
    fig_sparsity_sweep()
    # fig_sae_training_stats()


if __name__ == "__main__":
    main()
