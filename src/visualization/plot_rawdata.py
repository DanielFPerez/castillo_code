import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def plot_histogram_and_cdf(data: List[int], in_size=(10, 4), in_title="Conversations' Length", 
                           in_xlabel = str(),
                           x_forcdf = 0, cdf_forx=list(), n_bins=50, logx=False):
    """
    Plots a histogram (left) and an empirical CDF (right) of the given list of integers.

    Parameters:
    data (list of int): The input list of integers.
    """
    import matplotlib.ticker as ticker
    data = np.array(data)
    
    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=in_size)

    xlabel = in_xlabel if in_xlabel else "Number of Tokens"

    # Histogram
    axes[0].hist(data, bins=n_bins, edgecolor='black', alpha=0.7)
    axes[0].set_title(f"Histogram - {in_title}")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

    if logx:
        axes[0].set_xscale('log')
        axes[0].xaxis.set_major_formatter(ticker.ScalarFormatter())
        axes[0].xaxis.set_minor_formatter(ticker.NullFormatter())

    # Empirical CDF
    
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axes[1].plot(sorted_data, cdf, linestyle='-', color='b', alpha=0.7)
    
    axes[1].set_title(f"ECDF - {in_title}")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Cumulative Probability")

    if len(cdf_forx):
        for elem in cdf_forx:
            cdf_threshold = elem
            if np.any(cdf >= cdf_threshold):
                x_value_at_elem = sorted_data[np.searchsorted(cdf, cdf_threshold)]
                axes[1].axvline(x_value_at_elem, color='green', linestyle='--', label=f"{elem} CDF at {x_value_at_elem}")
                axes[1].legend()

     # Find CDF value at x = 1
    if x_forcdf > 0:
        cdf_at_1_index = np.searchsorted(sorted_data, 2, side='right')  # Find how many values are ≤ 1
        cdf_at_1 = cdf_at_1_index / len(sorted_data) if cdf_at_1_index < len(sorted_data) else 1.0
    
        axes[1].axvline(2, color='red', linestyle='--', label=f"CDF at x=1: {cdf_at_1:.2f}")
        axes[1].annotate(f"{cdf_at_1:.2f}", xy=(1, cdf_at_1), xytext=(1.5, cdf_at_1 - 0.05),
                         arrowprops=dict(arrowstyle="->", color="red"), color="red")

    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    if logx:
        axes[1].set_xscale('log')
        axes[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
        axes[1].xaxis.set_minor_formatter(ticker.NullFormatter())

    plt.tight_layout()
    return fig


def plot_dataframe_ecdf(df: pd.DataFrame, percentiles=[0.25, 0.5, 0.75, 0.95, 0.99], figsize=(14, 6), title_suffix=""):
    """
    Plots two empirical CDFs:
    - Left: ECDF of 'n_tokens_input' with vertical lines at given percentiles.
    - Right: ECDF of 'n_tokens_input' per category.
    
    Parameters:
    - df: pandas DataFrame containing 'n_tokens_input' and 'category'.
    - percentiles: List of percentiles (between 0 and 1) for vertical lines in the first plot.
    - figsize: Tuple indicating the figure size.
    - title_suffix: String to append to the titles for customization.
    
    Returns:
    - fig: Matplotlib figure object.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Left plot: ECDF of all data
    sorted_data = np.sort(df['n_tokens_input'])
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axs[0].plot(sorted_data, ecdf, label='ECDF')
    
    # Percentile lines
    for p in percentiles:
        val = np.percentile(sorted_data, p * 100)
        axs[0].axvline(val, color='green', linestyle='--', label=f'{p:.2f} Perc. at {val:.1f}')
    
    axs[0].set_title(f'ECDF # Input Tokens {title_suffix}')
    axs[0].set_xlabel('n_tokens_input')
    axs[0].set_ylabel('CDF')
    axs[0].legend()
    axs[0].grid(True)

    # Right plot: ECDF per category
    categories = df['category'].unique()
    for cat in sorted(categories):
        data_cat = np.sort(df[df['category'] == cat]['n_tokens_input'])
        ecdf_cat = np.arange(1, len(data_cat) + 1) / len(data_cat)
        axs[1].plot(data_cat, ecdf_cat, label=cat)
    
    axs[1].set_xscale('log')
    axs[1].set_title(f'ECDF per Category {title_suffix}')
    axs[1].set_xlabel('n_tokens_input (log-scale)')
    axs[1].xaxis.set_major_formatter(ScalarFormatter())
    axs[1].xaxis.set_minor_formatter(NullFormatter())
    axs[1].set_ylabel('CDF')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    return fig


def plot_dollydataframe_ecdf_threeplot(df: pd.DataFrame, percentiles=[0.25, 0.5, 0.75, 0.95, 0.99], figsize=(21, 6), title_suffix=""):
    """
    Plots three visualizations:
    - Left: ECDF of 'n_tokens_input' with vertical lines at given percentiles.
    - Middle: ECDF of 'n_tokens_input' per category.
    - Right: One vertical stacked bar showing category proportions.
    
    Parameters:
    - df: pandas DataFrame containing 'n_tokens_input' and 'category'.
    - percentiles: List of percentiles (between 0 and 1) for vertical lines in the first plot.
    - figsize: Tuple indicating the figure size.
    - title_suffix: String to append to the titles for customization.
    
    Returns:
    - fig: Matplotlib figure object.
    """
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D

    fig, axs = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [2.5, 3, 1]})

    # --- Left plot: Global ECDF ---
    sorted_data = np.sort(df['n_tokens_input'])
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axs[0].plot(sorted_data, ecdf, label='CDF')

    for p in percentiles:
        val = np.percentile(sorted_data, p * 100)
        axs[0].axvline(val, color="green", linestyle='--', label=f'{p:.2f} Perc. at {val:.1f}')
    
    axs[0].set_title(f'CDF # Input Tokens {title_suffix}')
    axs[0].set_xlabel('Input tokens')
    axs[0].set_ylabel('CDF')
    axs[0].legend()
    axs[0].grid(True)

    # --- Middle plot: ECDF per category ---
    category_colors = {}
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    categories = sorted(df['category'].unique())

    for i, cat in enumerate(categories):
        data_cat = np.sort(df[df['category'] == cat]['n_tokens_input'])
        ecdf_cat = np.arange(1, len(data_cat) + 1) / len(data_cat)
        color = color_cycle[i % len(color_cycle)]
        category_colors[cat] = color
        axs[1].plot(data_cat, ecdf_cat, label=cat, color=color)

    axs[1].set_xscale('log')
    axs[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
    axs[1].xaxis.set_minor_formatter(ticker.NullFormatter())
    axs[1].set_title(f'CDF per Category {title_suffix}')
    axs[1].set_xlabel('Input tokens (log-scale)')
    axs[1].set_ylabel('CDF')
    
    # Custom legend with thicker lines
    custom_lines = [
        Line2D([0], [0], color=category_colors[cat], lw=3)  # thicker legend line
        for cat in categories
    ]
    axs[1].legend(custom_lines, categories, title="Category", fontsize=9, title_fontsize=8)
    
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Right plot: Stacked bar for category distribution ---
    category_counts = df['category'].value_counts().reindex(categories)
    heights = category_counts.values
    bottoms = np.cumsum([0] + list(heights[:-1]))

    for i, cat in enumerate(categories):
        height = heights[i]
        bottom = bottoms[i]
        color = category_colors[cat]
        axs[2].bar(0, height, bottom=bottom, color=color, label=cat)
        # Add count text
        axs[2].text(0, bottom + height / 2, f'{height}', ha='center', va='center', fontsize=8, color='white' if height > 20 else 'black')

    axs[2].set_xlim(-0.5, 0.5)
    axs[2].set_xticks([])
    axs[2].set_title(f'Sample Counts')
    axs[2].set_ylabel('Num. of Samples')
    axs[2].grid(False)

    plt.tight_layout()
    return fig


def plot_appsdataframe_ecdf_threeplot(df: pd.DataFrame, percentiles=[0.25, 0.5, 0.75, 0.95, 0.99], figsize=(21, 6), title_suffix=""):
    """
    Plots three visualizations:
    - Left: ECDF of 'n_tokens_input' with vertical lines at given percentiles.
    - Middle: ECDF of 'n_tokens_input' per difficulty
    - Right: One vertical stacked bar showing difficulty proportions.
    
    Parameters:
    - df: pandas DataFrame containing 'n_tokens_input' and 'difficulty'.
    - percentiles: List of percentiles (between 0 and 1) for vertical lines in the first plot.
    - figsize: Tuple indicating the figure size.
    - title_suffix: String to append to the titles for customization.
    
    Returns:
    - fig: Matplotlib figure object.
    """
    import matplotlib.ticker as ticker
    from matplotlib.lines import Line2D

    fig, axs = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [2.5, 3, 1]})

    # --- Left plot: Global ECDF ---
    sorted_data = np.sort(df['n_tokens_input'])
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    axs[0].plot(sorted_data, ecdf, label='CDF')

    for p in percentiles:
        val = np.percentile(sorted_data, p * 100)
        axs[0].axvline(val, color="green", linestyle='--', label=f'{p:.2f} Perc. at {val:.1f}')
    
    axs[0].set_title(f'CDF # Input Tokens {title_suffix}')
    axs[0].set_xlabel('Number of tokens')
    axs[0].set_ylabel('CDF')
    axs[0].legend()
    axs[0].grid(True)

    # --- Middle plot: ECDF per category ---
    difficulty_colors = {}
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    difficulties = sorted(df['difficulty'].unique())

    for i, diffic in enumerate(difficulties):
        data_diffic = np.sort(df[df['difficulty'] == diffic]['n_tokens_input'])
        ecdf_diffic = np.arange(1, len(data_diffic) + 1) / len(data_diffic)
        color = color_cycle[i % len(color_cycle)]
        difficulty_colors[diffic] = color
        axs[1].plot(data_diffic, ecdf_diffic, label=diffic, color=color)

    axs[1].set_xscale('log')
    axs[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
    axs[1].xaxis.set_minor_formatter(ticker.NullFormatter())
    axs[1].set_title(f'CDF per difficulty {title_suffix}')
    axs[1].set_xlabel('Number of tokens(log-scale)')
    axs[1].set_ylabel('CDF')
    
    # Custom legend with thicker lines
    custom_lines = [
        Line2D([0], [0], color=difficulty_colors[diffic], lw=3)  # thicker legend line
        for diffic in difficulties
    ]
    axs[1].legend(custom_lines, difficulties, title="Difficulty", fontsize=9, title_fontsize=9)
    
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    # --- Right plot: Stacked bar for category distribution ---
    difficulty_counts = df['difficulty'].value_counts().reindex(difficulties)
    heights = difficulty_counts.values
    bottoms = np.cumsum([0] + list(heights[:-1]))

    for i, diffic in enumerate(difficulties):
        height = heights[i]
        bottom = bottoms[i]
        color = difficulty_colors[diffic]
        axs[2].bar(0, height, bottom=bottom, color=color, label=diffic)
        # Add count text
        axs[2].text(0, bottom + height / 2, f'{height}', ha='center', va='center', fontsize=9, color='white' if height > 20 else 'black')

    axs[2].set_xlim(-0.5, 0.5)
    axs[2].set_xticks([])
    axs[2].set_title(f'Sample Counts')
    axs[2].set_ylabel('Num. of Samples')
    axs[2].grid(False)

    plt.tight_layout()
    return fig