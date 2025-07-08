import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import json
import glob

# 1. Load your results into a DataFrame.
#    For example, if you saved them to CSV:

# 1. Set your base directory where all experiment folders live
base_dir = "/home/luka/repo/JAXOvercooked/results/data/ippo/Online EWC/random_5"

# 2. Recursively find all `training_reward.json` files
pattern = os.path.join(base_dir, "**", "training_reward.json")
filepaths = glob.glob(pattern, recursive=True)

# 3. Iterate and parse each run
records = []
for fp in filepaths:
    parts = fp.split(os.sep)
    # find the index of 'random_5' to locate architecture parts consistently
    idx = parts.index("random_5")
    cnn_part       = parts[idx + 1].lower()
    shared_part    = parts[idx + 2]
    multihead_part = parts[idx + 3]
    taskid_part    = parts[idx + 4]
    seed_part      = parts[idx + 5]  # e.g. 'seed_1'
    
    # parse boolean flags: assume parts prefixed with 'no-' indicate False
    cnn_flag       = (cnn_part == "cnn")
    shared_flag    = not shared_part.lower().startswith("no")
    multihead_flag = not multihead_part.lower().startswith("no")
    taskid_flag    = not taskid_part.lower().startswith("no")
    seed = int(seed_part.split("_")[-1])
    
    # load the JSON array of rewards
    with open(fp, "r") as f:
        rewards = json.load(f)
        # ensure rewards are scaled if they are higher than 2.0
        rewards = [r / 340.0 if r > 2.0 else r for r in rewards]
    
    # optionally aggregate (mean) or keep the full list
    avg_reward = sum(rewards) / len(rewards)
    
    records.append({
        "cnn": cnn_flag,
        "shared": shared_flag,
        "multihead": multihead_flag,
        "task_id": taskid_flag,
        "seed": seed,
        "avg_reward": avg_reward
        # if you want the full time-series, store `rewards` instead
    })

# 4. Convert into a DataFrame
df = pd.DataFrame(records)

# 5. Display the resulting DataFrame
df

print(df.head())
exit(0)

print(df['avg_reward'].describe())   # sanity check on your metric
for col in ['cnn','shared','multihead','task_id']:
    print(col, df[col].value_counts())


means = df.groupby('cnn')['avg_reward'].agg(['mean','std','count'])
print(means)

# ---------------------------------------------------------------------------
# Now, perform the full-factorial ANOVA analysis:

# 2. Ensure your factors are categorical.
for col in ['cnn', 'shared', 'multihead', 'task_id', 'seed']:
    df[col] = df[col].astype('category')

# 3. Specify and fit the full-factorial ANOVA model.
#    performance is your metric column (e.g., average return).
model = ols('avg_reward ~ C(cnn)*C(shared)*C(multihead)*C(task_id)', data=df).fit()

# 4. Generate the ANOVA table (Type II sums of squares).
anova_table = sm.stats.anova_lm(model, typ=2)

# 5. Compute effect sizes.
anova_table['eta_sq'] = anova_table['sum_sq'] / anova_table['sum_sq'].sum()
anova_table['partial_eta_sq'] = anova_table['sum_sq'] / (anova_table['sum_sq'] + anova_table.loc['Residual','sum_sq'])

# 6. Display results
print(anova_table)

print("\n__________________________________________________\n")
means = df.groupby(['cnn','multihead'])['avg_reward'].mean()
print(means.unstack())

# Key rows:
# - 'C(cnn)'                  → main effect of the CNN encoder
# - 'C(shared)'               → main effect of shared backbone
# - 'C(cnn):C(shared)'        → two-way interaction between CNN and shared backbone
# - Higher-order interactions follow similarly (e.g., 'C(cnn):C(shared):C(multihead)', etc.)


