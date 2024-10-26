import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Import the data
df = pd.read_csv('medical_examination.csv')

# Step 2: Add an overweight column
# BMI = weight(kg) / (height(m)^2), overweight if BMI > 25
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# Step 3: Normalize cholesterol and glucose columns
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Step 4: Draw Categorical Plot
def draw_cat_plot():
    # Step 4a: Melt the data for categorical plotting
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Step 4b: Group and format the data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    
    # Step 4c: Draw the catplot
    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio", data=df_cat, kind="bar").fig
    
    return fig

# Step 5: Draw Heat Map
def draw_heat_map():
    # Step 5a: Clean the data based on provided filters
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]
    
    # Step 5b: Calculate the correlation matrix
    corr = df_heat.corr()
    
    # Step 5c: Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Step 5d: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Step 5e: Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5}, ax=ax)
    
    return fig
