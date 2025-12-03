import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from itertools import combinations
import os

# ============================================================
# SETUP
# ============================================================
# Create images folder if it doesn't exist
images_folder = '../images'
os.makedirs(images_folder, exist_ok=True)

# Load the transformed dataset
df = pd.read_csv('../transformed_dataset.csv')

# Define features (X) - same for both analyses
feature_columns = ['urgency_level', 'mood_before', 'social_pressure', 'cost_implication']
X = df[feature_columns].copy()

# Add all 2-way feature interactions
interaction_names = []
for feat1, feat2 in combinations(feature_columns, 2):
    inter_name = f"{feat1}_x_{feat2}"
    X[inter_name] = X[feat1] * X[feat2]
    interaction_names.append(inter_name)

all_feature_columns = feature_columns + interaction_names

# Split the data (80% train, 20% test) - same split for both analyses
X_train, X_test, y_regret_train, y_regret_test = train_test_split(
    X, df['regret_label'], test_size=0.2, random_state=42
)
_, _, y_satisfaction_train, y_satisfaction_test = train_test_split(
    X, df['outcome_satisfaction'], test_size=0.2, random_state=42
)

print("="*80)
print("MERGED ANALYSIS: REGRET LABEL & OUTCOME SATISFACTION")
print("="*80)
print(f"\nTotal samples: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"\nFeatures used: {all_feature_columns}")

# ============================================================
# PART 1: REGRET LABEL ANALYSIS
# ============================================================
print("\n" + "="*80)
print("PART 1: REGRET LABEL ANALYSIS")
print("="*80)

print(f"\nRegret Label Distribution:")
print(df['regret_label'].value_counts().sort_index())

# ============================================================
# 1.1 LOGISTIC REGRESSION - REGRET LABEL
# ============================================================
print("\n" + "-"*80)
print("1.1 LOGISTIC REGRESSION CLASSIFICATION (REGRET LABEL)")
print("-"*80)

lr_model_regret = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)
lr_model_regret.fit(X_train, y_regret_train)
y_pred_lr_regret = lr_model_regret.predict(X_test)

print("\nClassification Report (Regret Label):")
print(classification_report(y_regret_test, y_pred_lr_regret))

print("\nConfusion Matrix (Regret Label):")
cm_regret = confusion_matrix(y_regret_test, y_pred_lr_regret)
print(cm_regret)

cv_scores_regret = cross_val_score(lr_model_regret, X, df['regret_label'], cv=5)
print(f"\n5-Fold Cross-Validation Accuracy (Regret): {cv_scores_regret.mean():.4f} (+/- {cv_scores_regret.std():.4f})")

print("\nFeature Statistics by Regret Label:")
for feature in feature_columns:
    print(f"\n{feature}:")
    print(df.groupby('regret_label')[feature].agg(['mean', 'std']))

# ============================================================
# 1.2 RANDOM FOREST REGRESSOR - REGRET LABEL
# ============================================================
print("\n" + "-"*80)
print("1.2 RANDOM FOREST REGRESSOR (REGRET LABEL)")
print("-"*80)

rf_model_regret = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_regret.fit(X_train, y_regret_train)
y_pred_rf_regret = rf_model_regret.predict(X_test)

print("\nRandom Forest Regression Results (Regret Label):")
print(f"R² Score: {r2_score(y_regret_test, y_pred_rf_regret):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_regret_test, y_pred_rf_regret)):.4f}")
print(f"MAE: {mean_absolute_error(y_regret_test, y_pred_rf_regret):.4f}")

print("\nFeature Importances (Regret Label):")
for feature, importance in zip(all_feature_columns, rf_model_regret.feature_importances_):
    print(f"  {feature}: {importance:.4f}")

# ============================================================
# 1.3 CORRELATION ANALYSIS - REGRET LABEL
# ============================================================
print("\n" + "-"*80)
print("1.3 CORRELATION ANALYSIS (REGRET LABEL)")
print("-"*80)

corr_features_regret = all_feature_columns + ['regret_label']
correlation_df_regret = pd.concat([X, df['regret_label']], axis=1)[corr_features_regret].corr()
print("\nCorrelation with Regret Label:")
print(correlation_df_regret['regret_label'].sort_values(ascending=False))

# ============================================================
# PART 2: OUTCOME SATISFACTION ANALYSIS
# ============================================================
print("\n" + "="*80)
print("PART 2: OUTCOME SATISFACTION ANALYSIS")
print("="*80)

print(f"\nOutcome Satisfaction Distribution:")
print(df['outcome_satisfaction'].value_counts().sort_index())

# ============================================================
# 2.1 LOGISTIC REGRESSION - OUTCOME SATISFACTION
# ============================================================
print("\n" + "-"*80)
print("2.1 LOGISTIC REGRESSION CLASSIFICATION (OUTCOME SATISFACTION)")
print("-"*80)

lr_model_satisfaction = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', random_state=42)
lr_model_satisfaction.fit(X_train, y_satisfaction_train)
y_pred_lr_satisfaction = lr_model_satisfaction.predict(X_test)

print("\nClassification Report (Outcome Satisfaction):")
print(classification_report(y_satisfaction_test, y_pred_lr_satisfaction))

print("\nConfusion Matrix (Outcome Satisfaction):")
cm_satisfaction = confusion_matrix(y_satisfaction_test, y_pred_lr_satisfaction)
print(cm_satisfaction)

cv_scores_satisfaction = cross_val_score(lr_model_satisfaction, X, df['outcome_satisfaction'], cv=5)
print(f"\n5-Fold Cross-Validation Accuracy (Satisfaction): {cv_scores_satisfaction.mean():.4f} (+/- {cv_scores_satisfaction.std():.4f})")

print("\nFeature Statistics by Outcome Satisfaction:")
for feature in feature_columns:
    print(f"\n{feature}:")
    print(df.groupby('outcome_satisfaction')[feature].agg(['mean', 'std']))

# ============================================================
# 2.2 RANDOM FOREST REGRESSOR - OUTCOME SATISFACTION
# ============================================================
print("\n" + "-"*80)
print("2.2 RANDOM FOREST REGRESSOR (OUTCOME SATISFACTION)")
print("-"*80)

rf_model_satisfaction = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_satisfaction.fit(X_train, y_satisfaction_train)
y_pred_rf_satisfaction = rf_model_satisfaction.predict(X_test)

print("\nRandom Forest Regression Results (Outcome Satisfaction):")
print(f"R² Score: {r2_score(y_satisfaction_test, y_pred_rf_satisfaction):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_satisfaction_test, y_pred_rf_satisfaction)):.4f}")
print(f"MAE: {mean_absolute_error(y_satisfaction_test, y_pred_rf_satisfaction):.4f}")

print("\nFeature Importances (Outcome Satisfaction):")
for feature, importance in zip(all_feature_columns, rf_model_satisfaction.feature_importances_):
    print(f"  {feature}: {importance:.4f}")

# ============================================================
# 2.3 CORRELATION ANALYSIS - OUTCOME SATISFACTION
# ============================================================
print("\n" + "-"*80)
print("2.3 CORRELATION ANALYSIS (OUTCOME SATISFACTION)")
print("-"*80)

corr_features_satisfaction = all_feature_columns + ['outcome_satisfaction']
correlation_df_satisfaction = pd.concat([X, df['outcome_satisfaction']], axis=1)[corr_features_satisfaction].corr()
print("\nCorrelation with Outcome Satisfaction:")
print(correlation_df_satisfaction['outcome_satisfaction'].sort_values(ascending=False))

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
n_cols = len(feature_columns) + len(interaction_names)

# ============================================================
# VISUALIZATION 1: REGRET LABEL ANALYSIS
# ============================================================
print("\nCreating visualization for Regret Label Analysis...")

fig1 = plt.figure(figsize=(max(28, 4*n_cols), 14))
gs1 = fig1.add_gridspec(3, n_cols, hspace=0.30, wspace=0.30)

# Row 1: Individual main feature line plots
for idx, (feature, color) in enumerate(zip(feature_columns, colors)):
    ax = fig1.add_subplot(gs1[0, idx])
    feature_means = df.groupby(feature)['regret_label'].mean()
    ax.plot(feature_means.index, feature_means.values, marker='o', linewidth=3, markersize=10, color=color)
    ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Avg Regret Label', fontsize=12)
    ax.set_title(f'Regret Label vs {feature.replace("_", " ").title()}', fontsize=13)
    ax.grid(True, alpha=0.3)

# Row 1 continued: All 2-feature interaction line plots
for idx, inter in enumerate(interaction_names):
    ax = fig1.add_subplot(gs1[0, idx+len(feature_columns)])
    inter_vals = X[inter]
    unique_vals = np.unique(inter_vals)
    if len(unique_vals) > 10:
        bins = np.linspace(np.min(inter_vals), np.max(inter_vals), 10)
        inter_binned = pd.cut(inter_vals, bins=bins, include_lowest=True)
        means = df['regret_label'].groupby(inter_binned, observed=False).mean()
        bin_centers = [interval.mid for interval in means.index]
        ax.plot(bin_centers, means.values, marker='o', color='#6f42c1', linewidth=3, markersize=10)
        ax.set_xlabel(f'{inter}(Binned)', fontsize=11)
    else:
        means = pd.DataFrame({'inter': inter_vals, 'regret': df['regret_label']}).groupby('inter')['regret'].mean()
        ax.plot(means.index, means.values, marker='o', color='#6f42c1', linewidth=3, markersize=10)
        ax.set_xlabel(inter, fontsize=11)
    ax.set_ylabel('Avg Regret Label', fontsize=12)
    ax.set_title(f'Interaction: {inter}', fontsize=13)
    ax.grid(True, alpha=0.2)

# Row 2: Correlation heatmap, Feature importance
ax1 = fig1.add_subplot(gs1[1, :int(n_cols/2)])
sns.heatmap(correlation_df_regret, annot=True, cmap='coolwarm', center=0, ax=ax1)
ax1.set_title('Feature Correlation Heatmap (Regret Label)', fontsize=14)

ax2 = fig1.add_subplot(gs1[1, int(n_cols/2):])
ax2.barh(all_feature_columns, rf_model_regret.feature_importances_, color='steelblue')
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title('Random Forest Feature Importance (Regret Label)', fontsize=14)

# Row 3: Actual vs Predicted, Confusion Matrix, All features line plot
ax3 = fig1.add_subplot(gs1[2, :2])
ax3.scatter(y_regret_test, y_pred_rf_regret, alpha=0.5, color='steelblue')
ax3.plot([y_regret_test.min(), y_regret_test.max()], [y_regret_test.min(), y_regret_test.max()], 'r--', lw=2)
ax3.set_xlabel('Actual Regret Label', fontsize=12)
ax3.set_ylabel('Predicted Regret Label', fontsize=12)
ax3.set_title('Random Forest: Actual vs Predicted (Regret)', fontsize=14)

ax4 = fig1.add_subplot(gs1[2, 2:4])
sns.heatmap(cm_regret, annot=True, fmt='d', cmap='Blues', ax=ax4)
ax4.set_xlabel('Predicted', fontsize=12)
ax4.set_ylabel('Actual', fontsize=12)
ax4.set_title('Logistic Regression Confusion Matrix (Regret)', fontsize=14)

ax5 = fig1.add_subplot(gs1[2, 4:])
for feature, color in zip(feature_columns, colors):
    feature_means = df.groupby(feature)['regret_label'].mean()
    ax5.plot(feature_means.index, feature_means.values, marker='o', linewidth=2, markersize=8, label=feature.replace('_', ' ').title(), color=color)
ax5.set_xlabel('Feature Value', fontsize=12)
ax5.set_ylabel('Avg Regret Label', fontsize=12)
ax5.set_title('All Main Features Combined (Regret)', fontsize=13)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.35)

plt.tight_layout()
regret_image_path = os.path.join(images_folder, 'model_analysis_regret_merged.png')
fig1.savefig(regret_image_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved as '{regret_image_path}'")
plt.close(fig1)

# ============================================================
# VISUALIZATION 2: OUTCOME SATISFACTION ANALYSIS
# ============================================================
print("Creating visualization for Outcome Satisfaction Analysis...")

fig2 = plt.figure(figsize=(max(28, 4*n_cols), 14))
gs2 = fig2.add_gridspec(3, n_cols, hspace=0.30, wspace=0.30)

# Row 1: Individual main feature line plots
for idx, (feature, color) in enumerate(zip(feature_columns, colors)):
    ax = fig2.add_subplot(gs2[0, idx])
    feature_means = df.groupby(feature)['outcome_satisfaction'].mean()
    ax.plot(feature_means.index, feature_means.values, marker='o', linewidth=3, markersize=10, color=color)
    ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Avg Outcome Satisfaction', fontsize=12)
    ax.set_title(f'Outcome Satisfaction vs {feature.replace("_", " ").title()}', fontsize=13)
    ax.grid(True, alpha=0.3)

# Row 1 continued: All 2-feature interaction line plots
for idx, inter in enumerate(interaction_names):
    ax = fig2.add_subplot(gs2[0, idx+len(feature_columns)])
    inter_vals = X[inter]
    unique_vals = np.unique(inter_vals)
    if len(unique_vals) > 10:
        bins = np.linspace(np.min(inter_vals), np.max(inter_vals), 10)
        inter_binned = pd.cut(inter_vals, bins=bins, include_lowest=True)
        means = df['outcome_satisfaction'].groupby(inter_binned, observed=False).mean()
        bin_centers = [interval.mid for interval in means.index]
        ax.plot(bin_centers, means.values, marker='o', color='#6f42c1', linewidth=3, markersize=10)
        ax.set_xlabel(f'{inter}(Binned)', fontsize=11)
    else:
        means = pd.DataFrame({'inter': inter_vals, 'outcome_satisfaction': df['outcome_satisfaction']}).groupby('inter')['outcome_satisfaction'].mean()
        ax.plot(means.index, means.values, marker='o', color='#6f42c1', linewidth=3, markersize=10)
        ax.set_xlabel(inter, fontsize=11)
    ax.set_ylabel('Avg Outcome Satisfaction', fontsize=12)
    ax.set_title(f'Interaction: {inter}', fontsize=13)
    ax.grid(True, alpha=0.2)

# Row 2: Correlation heatmap, Feature importance
ax1 = fig2.add_subplot(gs2[1, :int(n_cols/2)])
sns.heatmap(correlation_df_satisfaction, annot=True, cmap='coolwarm', center=0, ax=ax1)
ax1.set_title('Feature Correlation Heatmap (Outcome Satisfaction)', fontsize=14)

ax2 = fig2.add_subplot(gs2[1, int(n_cols/2):])
ax2.barh(all_feature_columns, rf_model_satisfaction.feature_importances_, color='steelblue')
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title('Random Forest Feature Importance (Outcome Satisfaction)', fontsize=14)

# Row 3: Actual vs Predicted, Confusion Matrix, All features line plot
ax3 = fig2.add_subplot(gs2[2, :2])
ax3.scatter(y_satisfaction_test, y_pred_rf_satisfaction, alpha=0.5, color='steelblue')
ax3.plot([y_satisfaction_test.min(), y_satisfaction_test.max()], [y_satisfaction_test.min(), y_satisfaction_test.max()], 'r--', lw=2)
ax3.set_xlabel('Actual Outcome Satisfaction', fontsize=12)
ax3.set_ylabel('Predicted Outcome Satisfaction', fontsize=12)
ax3.set_title('Random Forest: Actual vs Predicted (Satisfaction)', fontsize=14)

ax4 = fig2.add_subplot(gs2[2, 2:4])
sns.heatmap(cm_satisfaction, annot=True, fmt='d', cmap='Blues', ax=ax4)
ax4.set_xlabel('Predicted', fontsize=12)
ax4.set_ylabel('Actual', fontsize=12)
ax4.set_title('Logistic Regression Confusion Matrix (Satisfaction)', fontsize=14)

ax5 = fig2.add_subplot(gs2[2, 4:])
for feature, color in zip(feature_columns, colors):
    feature_means = df.groupby(feature)['outcome_satisfaction'].mean()
    ax5.plot(feature_means.index, feature_means.values, marker='o', linewidth=2, markersize=8, label=feature.replace('_', ' ').title(), color=color)
ax5.set_xlabel('Feature Value', fontsize=12)
ax5.set_ylabel('Avg Outcome Satisfaction', fontsize=12)
ax5.set_title('All Main Features Combined (Satisfaction)', fontsize=13)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.35)

plt.tight_layout()
satisfaction_image_path = os.path.join(images_folder, 'model_analysis_satisfaction_merged.png')
fig2.savefig(satisfaction_image_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved as '{satisfaction_image_path}'")
plt.close(fig2)

# ============================================================
# SAVE MODELS
# ============================================================
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

models_folder = '.'
with open(os.path.join(models_folder, 'logistic_regression_model_regret.pkl'), 'wb') as f:
    pickle.dump(lr_model_regret, f)

with open(os.path.join(models_folder, 'random_forest_model_regret.pkl'), 'wb') as f:
    pickle.dump(rf_model_regret, f)

with open(os.path.join(models_folder, 'logistic_regression_model_satisfaction.pkl'), 'wb') as f:
    pickle.dump(lr_model_satisfaction, f)

with open(os.path.join(models_folder, 'random_forest_model_satisfaction.pkl'), 'wb') as f:
    pickle.dump(rf_model_satisfaction, f)

print("\n✓ Models saved:")
print("  - logistic_regression_model_regret.pkl")
print("  - random_forest_model_regret.pkl")
print("  - logistic_regression_model_satisfaction.pkl")
print("  - random_forest_model_satisfaction.pkl")

# ============================================================
# EXAMPLE PREDICTIONS
# ============================================================
print("\n" + "="*80)
print("EXAMPLE PREDICTIONS (WITH FEATURE INTERACTIONS)")
print("="*80)

# Create interaction columns for example input
example_input = pd.DataFrame({
    'urgency_level': [4],
    'mood_before': [5],
    'social_pressure': [1],
    'cost_implication': [1]
})
for feat1, feat2 in combinations(feature_columns, 2):
    inter_name = f"{feat1}_x_{feat2}"
    example_input[inter_name] = example_input[feat1] * example_input[feat2]

print(f"\nExample Input: Urgency=4, Mood=5, Social Pressure=1, Cost=1")

print("\n--- REGRET LABEL PREDICTIONS ---")
lr_pred_regret = lr_model_regret.predict(example_input)[0]
rf_pred_regret = rf_model_regret.predict(example_input)[0]
print(f"Logistic Regression Prediction: {lr_pred_regret}")
print(f"Random Forest Prediction: {rf_pred_regret:.2f}")

print("\n--- OUTCOME SATISFACTION PREDICTIONS ---")
lr_pred_satisfaction = lr_model_satisfaction.predict(example_input)[0]
rf_pred_satisfaction = rf_model_satisfaction.predict(example_input)[0]
print(f"Logistic Regression Prediction: {lr_pred_satisfaction}")
print(f"Random Forest Prediction: {rf_pred_satisfaction:.2f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\n✓ All visualizations saved to: {images_folder}/")
print(f"✓ Check the images folder for:")
print(f"  - model_analysis_regret_merged.png")
print(f"  - model_analysis_satisfaction_merged.png")
