import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
import pickle

# Ensure images folder exists at project root
images_folder = 'images'
os.makedirs(images_folder, exist_ok=True)

# Load dataset
df = pd.read_csv('transformed_dataset.csv')

# Shared feature list
feature_columns = ['urgency_level', 'mood_before', 'social_pressure', 'cost_implication']

# -------------------------
# PART A — Logistic Regression (with feature interactions)
# -------------------------
print('\n== Logistic Regression analyses (with interactions) ==')
# Build interaction features
X_lr = df[feature_columns].copy()
interaction_names = []
for f1, f2 in combinations(feature_columns, 2):
    name = f"{f1}_x_{f2}"
    X_lr[name] = X_lr[f1] * X_lr[f2]
    interaction_names.append(name)
all_features_lr = feature_columns + interaction_names

# Split for regret and satisfaction using same X_lr split
X_train_lr, X_test_lr, y_regret_train_lr, y_regret_test_lr = train_test_split(
    X_lr, df['regret_label'], test_size=0.2, random_state=42, stratify=df['regret_label'] if 'regret_label' in df.columns else None
)
_, _, y_sat_train_lr, y_sat_test_lr = train_test_split(
    X_lr, df['outcome_satisfaction'], test_size=0.2, random_state=42, stratify=df['outcome_satisfaction'] if 'outcome_satisfaction' in df.columns else None
)

# Train LR models
lr_regret = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)
lr_regret.fit(X_train_lr, y_regret_train_lr)
y_pred_lr_regret = lr_regret.predict(X_test_lr)

lr_sat = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)
lr_sat.fit(X_train_lr, y_sat_train_lr)
y_pred_lr_sat = lr_sat.predict(X_test_lr)

# Save LR models
with open(os.path.join(images_folder, 'lr_model_regret.pkl'), 'wb') as f:
    pickle.dump(lr_regret, f)
with open(os.path.join(images_folder, 'lr_model_satisfaction.pkl'), 'wb') as f:
    pickle.dump(lr_sat, f)

# Visualization function for LR (regret / satisfaction)
def plot_lr_analysis(target_col, y_test, y_pred, corr_df, rf_importances=None, filename=None, ylabel=None, title_prefix=''):
    n_cols = len(feature_columns) + len(interaction_names)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    fig = plt.figure(figsize=(max(28, 4*n_cols), 14))
    gs = fig.add_gridspec(3, n_cols, hspace=0.30, wspace=0.30)

    # Row 1: main features
    for idx, (feature, color) in enumerate(zip(feature_columns, colors)):
        ax = fig.add_subplot(gs[0, idx])
        means = df.groupby(feature)[target_col].mean()
        ax.plot(means.index, means.values, marker='o', linewidth=3, markersize=10, color=color)
        ax.set_xlabel(feature.replace('_',' ').title())
        ax.set_ylabel(ylabel if ylabel else f'Avg {target_col}')
        ax.set_title(f'{title_prefix} {target_col} vs {feature.replace("_"," ").title()}')
        ax.grid(alpha=0.3)

    # Row 1 continued: interactions
    for idx, inter in enumerate(interaction_names):
        ax = fig.add_subplot(gs[0, idx+len(feature_columns)])
        vals = X_lr[inter]
        uniq = np.unique(vals)
        if len(uniq) > 10:
            bins = np.linspace(vals.min(), vals.max(), 10)
            binned = pd.cut(vals, bins=bins, include_lowest=True)
            means = df[target_col].groupby(binned, observed=False).mean()
            centers = [interval.mid for interval in means.index]
            ax.plot(centers, means.values, marker='o', color='#6f42c1')
            ax.set_xlabel(f'{inter} (binned)')
        else:
            means = pd.DataFrame({'inter': vals, target_col: df[target_col]}).groupby('inter')[target_col].mean()
            ax.plot(means.index, means.values, marker='o', color='#6f42c1')
            ax.set_xlabel(inter)
        ax.set_ylabel(ylabel if ylabel else f'Avg {target_col}')
        ax.set_title(f'Interaction: {inter}')
        ax.grid(alpha=0.2)

    # Row 2: correlation heatmap and  RF importances
    ax1 = fig.add_subplot(gs[1, :int(n_cols/2)])
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, ax=ax1)
    ax1.set_title(f'Feature Correlation Heatmap ({target_col})')

    if rf_importances is not None:
        ax2 = fig.add_subplot(gs[1, int(n_cols/2):])
        ax2.barh(all_features_lr, rf_importances, color='steelblue')
        ax2.set_xlabel('Importance')
        ax2.set_title('Random Forest Feature Importance')

    # Row 3: Actual vs Predicted scatter, confusion matrix, combined features
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax3.set_xlabel('Actual')
    ax3.set_ylabel('Predicted')
    ax3.set_title('Actual vs Predicted')

    ax4 = fig.add_subplot(gs[2, 2:4])
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('Actual')
    ax4.set_title('Confusion Matrix')

    ax5 = fig.add_subplot(gs[2, 4:])
    for feature, color in zip(feature_columns, colors):
        means = df.groupby(feature)[target_col].mean()
        ax5.plot(means.index, means.values, marker='o', linewidth=2, markersize=6, label=feature.replace('_',' ').title(), color=color)
    ax5.set_title('All Main Features Combined')
    ax5.legend()
    ax5.grid(alpha=0.35)

    plt.tight_layout()
    fig.savefig(os.path.join(images_folder, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")

# For LR regret: compute correlation and RF for importances
corr_lr_regret = pd.concat([X_lr, df['regret_label']], axis=1)[all_features_lr + ['regret_label']].corr()
rf_lr_regret = RandomForestRegressor(n_estimators=100, random_state=42)
rf_lr_regret.fit(X_train_lr, y_regret_train_lr)
rf_imp_lr_regret = rf_lr_regret.feature_importances_

plot_lr_analysis('regret_label', y_regret_test_lr, y_pred_lr_regret, corr_lr_regret,
                 rf_importances=rf_imp_lr_regret, filename='lr_regret_analysis.png', ylabel='Avg Regret Label', title_prefix='LR')

# For LR satisfaction
corr_lr_sat = pd.concat([X_lr, df['outcome_satisfaction']], axis=1)[all_features_lr + ['outcome_satisfaction']].corr()
rf_lr_sat = RandomForestRegressor(n_estimators=100, random_state=42)
rf_lr_sat.fit(X_train_lr, y_sat_train_lr)
rf_imp_lr_sat = rf_lr_sat.feature_importances_

plot_lr_analysis('outcome_satisfaction', y_sat_test_lr, y_pred_lr_sat, corr_lr_sat,
                 rf_importances=rf_imp_lr_sat, filename='lr_satisfaction_analysis.png', ylabel='Avg Outcome Satisfaction', title_prefix='LR')

# -------------------------
# PART B — Naive Bayes analyses (no feature interactions)
# -------------------------
print('\n== Naive Bayes analyses (no interactions) ==')
X_nb = df[feature_columns].copy()

# Regret NB
X_train_nb_reg, X_test_nb_reg, y_train_nb_reg, y_test_nb_reg = train_test_split(
    X_nb, df['regret_label'], test_size=0.2, random_state=42, stratify=df['regret_label'] if 'regret_label' in df.columns else None
)
nb_reg = GaussianNB()
nb_reg.fit(X_train_nb_reg, y_train_nb_reg)
y_pred_nb_reg = nb_reg.predict(X_test_nb_reg)

# RF for regret 
rf_nb_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_nb_reg.fit(X_train_nb_reg, y_train_nb_reg)
rf_imp_nb_reg = rf_nb_reg.feature_importances_

# Plot NB regret
corr_nb_reg = df[feature_columns + ['regret_label']].corr()
fig = plt.figure(figsize=(20,12))
gs = fig.add_gridspec(3,4,hspace=0.3,wspace=0.3)
colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728']
for idx,(feature,color) in enumerate(zip(feature_columns,colors)):
    ax = fig.add_subplot(gs[0,idx])
    means = df.groupby(feature)['regret_label'].mean()
    ax.plot(means.index, means.values, marker='o', linewidth=2.5, markersize=8, color=color)
    ax.set_xlabel(feature.replace('_',' ').title())
    ax.set_ylabel('Avg Regret Label')
    ax.set_title(f'{feature} vs Regret')
    ax.grid(True,alpha=0.3)
ax1 = fig.add_subplot(gs[1,0:2])
sns.heatmap(corr_nb_reg, annot=True, cmap='coolwarm', center=0, ax=ax1)
ax1.set_title('Feature Correlation Heatmap (Regret)')
ax2 = fig.add_subplot(gs[1,2:4])
ax2.barh(feature_columns, rf_imp_nb_reg, color='steelblue')
ax2.set_title('Random Forest Importances (Regret)')
ax3 = fig.add_subplot(gs[2,0:2])
ax3.scatter(y_test_nb_reg, rf_nb_reg.predict(X_test_nb_reg), alpha=0.5, color='steelblue')
ax3.plot([y_test_nb_reg.min(), y_test_nb_reg.max()],[y_test_nb_reg.min(), y_test_nb_reg.max()],'r--')
ax3.set_title('RF Actual vs Predicted (Regret)')
ax4 = fig.add_subplot(gs[2,2])
sns.heatmap(confusion_matrix(y_test_nb_reg, y_pred_nb_reg), annot=True, fmt='d', cmap='Blues', ax=ax4)
ax4.set_title('Naive Bayes Confusion Matrix (Regret)')
ax5 = fig.add_subplot(gs[2,3])
for feature,color in zip(feature_columns,colors):
    means = df.groupby(feature)['regret_label'].mean()
    ax5.plot(means.index, means.values, marker='o', linewidth=2, markersize=6, label=feature.replace('_',' ').title(), color=color)
ax5.set_title('All Features Combined (Regret)')
ax5.legend()
plt.tight_layout()
nb_regret_path = os.path.join(images_folder,'nb_regret_analysis.png')
fig.savefig(nb_regret_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {nb_regret_path}')

# Satisfaction NB
X_train_nb_sat, X_test_nb_sat, y_train_nb_sat, y_test_nb_sat = train_test_split(
    X_nb, df['outcome_satisfaction'], test_size=0.2, random_state=42, stratify=df['outcome_satisfaction'] if 'outcome_satisfaction' in df.columns else None
)
nb_sat = GaussianNB()
nb_sat.fit(X_train_nb_sat, y_train_nb_sat)
y_pred_nb_sat = nb_sat.predict(X_test_nb_sat)

rf_nb_sat = RandomForestRegressor(n_estimators=100, random_state=42)
rf_nb_sat.fit(X_train_nb_sat, y_train_nb_sat)
rf_imp_nb_sat = rf_nb_sat.feature_importances_

corr_nb_sat = df[feature_columns + ['outcome_satisfaction']].corr()
fig = plt.figure(figsize=(20,12))
gs = fig.add_gridspec(3,4,hspace=0.3,wspace=0.3)
for idx,(feature,color) in enumerate(zip(feature_columns,colors)):
    ax = fig.add_subplot(gs[0,idx])
    means = df.groupby(feature)['outcome_satisfaction'].mean()
    ax.plot(means.index, means.values, marker='o', linewidth=2.5, markersize=8, color=color)
    ax.set_xlabel(feature.replace('_',' ').title())
    ax.set_ylabel('Avg Outcome Satisfaction')
    ax.set_title(f'{feature} vs Outcome Satisfaction')
    ax.grid(True,alpha=0.3)
ax1 = fig.add_subplot(gs[1,0:2])
sns.heatmap(corr_nb_sat, annot=True, cmap='coolwarm', center=0, ax=ax1)
ax1.set_title('Feature Correlation Heatmap (Satisfaction)')
ax2 = fig.add_subplot(gs[1,2:4])
ax2.barh(feature_columns, rf_imp_nb_sat, color='steelblue')
ax2.set_title('Random Forest Importances (Satisfaction)')
ax3 = fig.add_subplot(gs[2,0:2])
ax3.scatter(y_test_nb_sat, rf_nb_sat.predict(X_test_nb_sat), alpha=0.5, color='steelblue')
ax3.plot([y_test_nb_sat.min(), y_test_nb_sat.max()],[y_test_nb_sat.min(), y_test_nb_sat.max()],'r--')
ax3.set_title('RF Actual vs Predicted (Satisfaction)')
ax4 = fig.add_subplot(gs[2,2])
sns.heatmap(confusion_matrix(y_test_nb_sat, y_pred_nb_sat), annot=True, fmt='d', cmap='Blues', ax=ax4)
ax4.set_title('Naive Bayes Confusion Matrix (Satisfaction)')
ax5 = fig.add_subplot(gs[2,3])
for feature,color in zip(feature_columns,colors):
    means = df.groupby(feature)['outcome_satisfaction'].mean()
    ax5.plot(means.index, means.values, marker='o', linewidth=2, markersize=6, label=feature.replace('_',' ').title(), color=color)
ax5.set_title('All Features Combined (Satisfaction)')
ax5.legend()
plt.tight_layout()
nb_sat_path = os.path.join(images_folder,'nb_satisfaction_analysis.png')
fig.savefig(nb_sat_path, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {nb_sat_path}')

# Save NB models
with open(os.path.join(images_folder,'nb_model_regret.pkl'),'wb') as f:
    pickle.dump(nb_reg, f)
with open(os.path.join(images_folder,'nb_model_satisfaction.pkl'),'wb') as f:
    pickle.dump(nb_sat, f)

print('\nAll analyses complete. Four images saved in images/:')
print(' - lr_regret_analysis.png')
print(' - lr_satisfaction_analysis.png')
print(' - nb_regret_analysis.png')
print(' - nb_satisfaction_analysis.png')
