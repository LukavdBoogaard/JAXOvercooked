from typing import List
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import json
import glob
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def split_into_chunks(arr: np.ndarray, n_chunks: int) -> List[np.ndarray]:
    """
    Evenly split *arr* into n_chunks (the last chunk gets the remainder).
    """
    base = len(arr) // n_chunks
    chunks = [arr[i*base:(i+1)*base] for i in range(n_chunks-1)]
    chunks.append(arr[(n_chunks-1)*base:])            # remainder → last chunk
    return chunks

# 1. Set your new base directory
base_dir = "/home/luka/repo/JAXOvercooked/results/ablation_data_different_structure/ippo/Online EWC/random_5"

# 2. Glob all reward files
pattern = os.path.join(base_dir, "**", "training_reward.json")
filepaths = glob.glob(pattern, recursive=True)

records = []

# 3. Loop through each file
for fp in filepaths:
    parts = fp.split(os.sep)

    # Determine condition from parent directory of seed folders
    idx = parts.index("random_5")
    condition = parts[idx + 1]
    seed_str = parts[idx + 2]  # "seed_1"
    seed = int(seed_str.split("_")[-1])

    # Parse factor flags from condition name
    flags = {
        "cnn": False,
        "shared": True,
        "multihead": True,
        "task_id": True,
        "layer_norm": True,
    }

    if condition != "baseline":
        if "use_cnn" in condition:
            flags["cnn"] = True
        if "shared_backbone" in condition:
            flags["shared"] = False
        if "multihead" in condition:
            flags["multihead"] = False
        if "task_id" in condition:
            flags["task_id"] = False
        if "layer_norm" in condition:
            flags["layer_norm"] = False

    # Load and scale rewards
    with open(fp, "r") as f:
        rewards = json.load(f)
        # rewards = [r / 340.0 if r > 2.0 else r for r in rewards]
        avg_reward = sum(rewards) / len(rewards)

        # chunks = split_into_chunks(rewards, 5)
        # task_means = [np.mean(chunk[-10:]) for chunk in chunks]
        # avg_reward = np.mean(task_means)

    # Save record
    records.append({
        "condition": condition,
        "seed": seed,
        "avg_reward": avg_reward,
        **flags
    })

# 4. Create DataFrame
df = pd.DataFrame(records)

# 5. Convert to categorical where appropriate
for col in ["cnn", "shared", "multihead", "task_id", "layer_norm", "condition"]:
    df[col] = df[col].astype("category")

# 6. Show a preview
print(df.head(25))
print(df['condition'].value_counts())

#######################
# Remove seed 4 from the dataframe
df = df[df["seed"] != 4]



# Assuming your DataFrame is named `df`
baseline_rewards = df[df['condition'] == 'baseline']['avg_reward']

# Store results
results = []

for condition in df['condition'].unique():
    if condition == 'baseline':
        continue
    ablation_rewards = df[df['condition'] == condition]['avg_reward']
    
    # Welch's t-test
    t_stat, p_val = ttest_ind(baseline_rewards, ablation_rewards, equal_var=False)
    
    results.append({
        'ablation': condition,
        'mean_baseline': baseline_rewards.mean(),
        'mean_ablation': ablation_rewards.mean(),
        'mean_diff': ablation_rewards.mean() - baseline_rewards.mean(),
        't_stat': t_stat,
        'p_val_uncorrected': p_val
    })

results_df = pd.DataFrame(results)


from statsmodels.stats.multitest import multipletests

pvals = results_df['p_val_uncorrected'].values
_, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')
results_df['p_val_corrected'] = pvals_corrected
results_df['significant'] = results_df['p_val_corrected'] < 0.05

print(results_df)

exit(0)






















































factors = ['cnn', 'shared', 'multihead', 'task_id', 'layer_norm']

# Construct formula: avg_reward ~ cnn + shared + ...
formula = 'avg_reward ~ ' + ' + '.join([f'C({f})' for f in factors])

# Fit linear model
model = ols(formula, data=df).fit()

# Run ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()

print("\n==== Main Effects ANOVA Table ====")
print(anova_table)

from scipy.stats import ttest_ind

baseline_scores = df[(df['condition'] == 'baseline')]['avg_reward']
no_shared_scores = df[(df['shared'] == False) & (df['condition'] != 'baseline')]['avg_reward']

t, p = ttest_ind(baseline_scores, no_shared_scores, equal_var=False)
print(f"Shared Backbone: t = {t:.3f}, p = {p:.4f}")


# Define the baseline group
baseline_scores = df[df["condition"] == "baseline"]["avg_reward"]

# Get all unique conditions except baseline
conditions = df["condition"].unique()
ablations = [cond for cond in conditions if cond != "baseline"]

# Store t-test results
results = []

for ablation in ablations:
    ablation_scores = df[df["condition"] == ablation]["avg_reward"]
    t_stat, p_val = ttest_ind(baseline_scores, ablation_scores, equal_var=False)  # Welch's t-test
    mean_diff = ablation_scores.mean() - baseline_scores.mean()
    results.append({
        "ablation": ablation,
        "mean_baseline": baseline_scores.mean(),
        "mean_ablation": ablation_scores.mean(),
        "mean_diff": mean_diff,
        "t_stat": t_stat,
        "p_val_uncorrected": p_val
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Apply Bonferroni or Holm correction
corrected = multipletests(results_df["p_val_uncorrected"], method="holm")
results_df["p_val_corrected"] = corrected[1]
results_df["significant"] = corrected[0]

# Display results
print("Pairwise T-Test Results \n", results_df)


# Function to run a main-effect test for a given factor
def test_main_effect(df, factor):
    # Baseline group = factor is True
    # Ablation group = factor is False
    baseline = df[df['condition'] == 'baseline']
    ablation = df[(df[factor] == False) & (df['condition'] != 'baseline')]

    if len(ablation) == 0:
        print(f"⚠️  No ablation found for {factor}, skipping.")
        return

    # Combine and relabel for modeling
    data = pd.concat([baseline, ablation], ignore_index=True)
    data['group'] = data[factor].apply(lambda x: 'on' if x else 'off')
    # print(data)

    print(f"\n==== Effect of {factor.upper()} ====")
    
    # T-test
    t, p = ttest_ind(
        data[data['group'] == 'on']['avg_reward'],
        data[data['group'] == 'off']['avg_reward'],
        equal_var=False
    )
    delta = data[data['group'] == 'off']['avg_reward'].mean() - data[data['group'] == 'on']['avg_reward'].mean()
    print(f"t={t:.3f}, p={p:.4f}, Δ={delta:.3f}")

    # Optional: Linear model (useful for η²)
    model = ols('avg_reward ~ C(group)', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
    print(anova_table)

# Run for each factor
for factor in ['cnn', 'shared', 'multihead', 'task_id', 'layer_norm']:
    test_main_effect(df, factor)
