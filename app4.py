

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base              import BaseEstimator, TransformerMixin
from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import (VarianceThreshold, f_classif,
                                        mutual_info_classif, SelectKBest)
from sklearn.ensemble          import RandomForestClassifier, StackingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.pipeline          import Pipeline
from sklearn.metrics           import (confusion_matrix, roc_curve, roc_auc_score,
                                        accuracy_score, precision_score,
                                        recall_score, f1_score)
import xgboost  as xgb
import lightgbm as lgb
import shap

# =============================================================================
#  KONFIGURASI
# =============================================================================

CSV_PATH    = 'Dataset/SOMLAP DATASET.csv'   # sesuaikan path CSV Anda
OUTPUT_DIR  = '.'
RANDOM_SEED = 42
TEST_SIZE   = 0.20
K_FEATURES  = 40
SHAP_SAMPLE = 300

# =============================================================================
#  NOVELTY 1 — HybridFeatureSelector (sklearn-compatible, no leakage)
# =============================================================================

class HybridFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Hybrid Feature Selection:
      ANOVA F-test + Mutual Information + XGBoost Importance.

    Setiap skor dinormalisasi ke [0,1] lalu dirata-rata → Combined Score.
    fit() HANYA menggunakan data training → bebas data leakage.
    """

    def __init__(self, k=40, random_state=42):
        self.k            = k
        self.random_state = random_state

    def fit(self, X, y):
        X_df  = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        cols  = X_df.columns.tolist()
        X_arr = X_df.values
        y_arr = np.array(y)

        # ── Metode A: ANOVA F-test ───────────────────────────────────────────
        anova_s = f_classif(X_arr, y_arr)[0]
        ar      = pd.Series(anova_s, index=cols)
        ar_n    = (ar - ar.min()) / (ar.max() - ar.min() + 1e-9)

        # ── Metode B: Mutual Information (subsample 5000) ────────────────────
        rng   = np.random.RandomState(self.random_state)
        idx_s = rng.choice(len(X_arr), min(5000, len(X_arr)), replace=False)
        mi_s  = mutual_info_classif(X_arr[idx_s], y_arr[idx_s],
                                     random_state=self.random_state)
        mr    = pd.Series(mi_s, index=cols)
        mr_n  = (mr - mr.min()) / (mr.max() - mr.min() + 1e-9)

        # ── Metode C: XGBoost Feature Importance ────────────────────────────
        xpre = xgb.XGBClassifier(n_estimators=50, random_state=self.random_state,
                                  eval_metric='logloss', n_jobs=-1)
        xpre.fit(X_arr, y_arr)
        xi   = pd.Series(xpre.feature_importances_, index=cols)
        xi_n = (xi - xi.min()) / (xi.max() - xi.min() + 1e-9)

        # ── Combined Score ───────────────────────────────────────────────────
        self.combined_score_     = (ar_n + mr_n + xi_n) / 3
        self.selected_features_  = self.combined_score_.nlargest(self.k).index.tolist()
        self.feature_names_in_   = cols

        # Simpan skor individual (untuk visualisasi)
        self.ar_n_ = ar_n
        self.mr_n_ = mr_n
        self.xi_n_ = xi_n

        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names_in_) \
               if not isinstance(X, pd.DataFrame) else X
        return X_df[self.selected_features_]

    def get_feature_names_out(self):
        return np.array(self.selected_features_)


# =============================================================================
#  HELPER — plot & evaluasi
# =============================================================================

PANEL  = '#ffffff'
BORDER = '#cccccc'
PAL    = ['#aaaaaa', '#5bc8f5', '#6bcb77', '#ffd93d', '#ff6b6b']

def base_fig(w=10, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.spines[:].set_color(BORDER)
    ax.tick_params(colors='black')
    return fig, ax

def save_fig(fig, name):
    path = f'{OUTPUT_DIR}/{name}'
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'    ✓ {path}')

def train_eval(model, X_tr, X_ts, y_tr, y_ts):
    model.fit(X_tr, y_tr)
    yp  = model.predict(X_ts)
    ypr = model.predict_proba(X_ts)[:, 1]
    return dict(
        y_pred    = yp,
        y_prob    = ypr,
        accuracy  = accuracy_score(y_ts, yp),
        precision = precision_score(y_ts, yp),
        recall    = recall_score(y_ts, yp),
        f1        = f1_score(y_ts, yp),
        auc       = roc_auc_score(y_ts, ypr),
    )


# =============================================================================
#  TAHAP 1 — LOAD DATA
# =============================================================================

print("=" * 65)
print("  SOMLAP — Pipeline Terkoreksi (Bebas Data Leakage)")
print("=" * 65)

print("\n[1/8] Load Data...")

df = pd.read_csv(CSV_PATH)
df = df.drop_duplicates()
X  = df.drop('class', axis=1)
X.columns = X.columns.str.strip()
y  = df['class'].reset_index(drop=True)
X  = X.reset_index(drop=True)

print(f"  Shape awal       : {df.shape}")
print(f"  Distribusi kelas :\n{y.value_counts().to_string()}")

# =============================================================================
#  TAHAP 2 — TRAIN/TEST SPLIT (WAJIB PERTAMA)
# =============================================================================

print("\n[2/8] Train/Test Split (PERTAMA — sebelum preprocessing apapun)...")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_SEED,
    stratify     = y
)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

print(f"  Training : {X_train_raw.shape[0]} sampel")
print(f"  Test     : {X_test_raw.shape[0]} sampel")

# =============================================================================
#  TAHAP 3 — VARIANCE THRESHOLD (fit di train saja) ✅ PERBAIKAN
# =============================================================================

print("\n[3/8] Variance Threshold (fit di training set saja)...")

# ✅ fit() hanya di training set → transform() ke test set
vt = VarianceThreshold(threshold=0)
vt.fit(X_train_raw)

X_train_vt = pd.DataFrame(
    vt.transform(X_train_raw),
    columns = X_train_raw.columns[vt.get_support()]
).reset_index(drop=True)

X_test_vt = pd.DataFrame(
    vt.transform(X_test_raw),
    columns = X_train_raw.columns[vt.get_support()]
).reset_index(drop=True)

print(f"  Fitur setelah VT : {X_train_vt.shape[1]}")

# =============================================================================
#  TAHAP 4 — NOVELTY 1: HYBRID FEATURE SELECTION (fit di train saja)
# =============================================================================

print("\n[4/8] Novelty 1 — Hybrid Feature Selection...")

hfs = HybridFeatureSelector(k=K_FEATURES, random_state=RANDOM_SEED)
hfs.fit(X_train_vt, y_train)       # ✅ fit hanya di training

X_train_h = hfs.transform(X_train_vt)
X_test_h  = hfs.transform(X_test_vt)

hybrid_features = hfs.selected_features_

print(f"  Fitur terpilih (Hybrid FS) : {len(hybrid_features)}")
print(f"  Top-5                      : {hybrid_features[:5]}")

# Baseline ANOVA-only (fit di train saja, jumlah fitur sama = K_FEATURES)
sel_base = SelectKBest(f_classif, k=K_FEATURES)
sel_base.fit(X_train_vt, y_train)  # ✅ fit hanya di training

base_cols = X_train_vt.columns[sel_base.get_support()].tolist()
X_train_b = pd.DataFrame(sel_base.transform(X_train_vt), columns=base_cols)
X_test_b  = pd.DataFrame(sel_base.transform(X_test_vt),  columns=base_cols)

print(f"  Fitur terpilih (ANOVA-only): {X_train_b.shape[1]}")

# =============================================================================
#  TAHAP 5 — TRAINING MODEL
# =============================================================================

print("\n[5/8] Training Model...")

# ── Baseline ──────────────────────────────────────────────────────────────────
print("  [a] Baseline: XGBoost + ANOVA")
r_base = train_eval(
    xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_SEED,
                       eval_metric='logloss', n_jobs=-1),
    X_train_b, X_test_b, y_train, y_test)

# ── Individual + Hybrid FS ────────────────────────────────────────────────────
print("  [b] XGBoost + Hybrid FS")
r_xgb = train_eval(
    xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_SEED,
                       eval_metric='logloss', n_jobs=-1),
    X_train_h, X_test_h, y_train, y_test)

print("  [c] Random Forest + Hybrid FS")
r_rf = train_eval(
    RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
    X_train_h, X_test_h, y_train, y_test)

print("  [d] LightGBM + Hybrid FS")
r_lgb = train_eval(
    lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_SEED,
                        n_jobs=-1, verbose=-1),
    X_train_h, X_test_h, y_train, y_test)

# ── Novelty 2: Stacking Ensemble ──────────────────────────────────────────────
print("  [e] Stacking Ensemble + Hybrid FS (Novelty 2)")

base_learners = [
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_SEED,
                                eval_metric='logloss', n_jobs=-1)),
    ('rf',  RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)),
    ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_SEED,
                                 n_jobs=-1, verbose=-1)),
]
meta_learner = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
stacking     = StackingClassifier(
    estimators      = base_learners,
    final_estimator = meta_learner,
    cv              = 5,
    n_jobs          = -1
)
r_stack = train_eval(stacking, X_train_h, X_test_h, y_train, y_test)

# =============================================================================
#  TAHAP 6 — CROSS-VALIDATION END-TO-END (✅ PERBAIKAN: hanya di train set)
# =============================================================================

print("\n[6/8] Cross-Validation End-to-End (di training set saja)...")

# ✅ CV di X_train_vt + y_train (bukan X_vt + y semua data)
# Pipeline memastikan HybridFS di-fit ulang tiap fold → bebas leakage
base_learners_cv = [
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_SEED,
                                eval_metric='logloss', n_jobs=-1)),
    ('rf',  RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)),
    ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=RANDOM_SEED,
                                 n_jobs=-1, verbose=-1)),
]

pipeline_cv = Pipeline([
    ('vt',    VarianceThreshold(threshold=0)),                         # ✅ dalam pipeline
    ('fs',    HybridFeatureSelector(k=K_FEATURES, random_state=RANDOM_SEED)),
    ('model', StackingClassifier(
        estimators      = base_learners_cv,
        final_estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        cv              = 5,
        n_jobs          = -1
    )),
])

cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scores = cross_val_score(
    pipeline_cv, X_train_raw, y_train,   # ✅ hanya training data
    cv      = cv,
    scoring = 'f1',
    n_jobs  = -1
)

print(f"  CV F1 per fold : {[f'{s:.4f}' for s in cv_scores]}")
print(f"  CV F1 mean±std : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# =============================================================================
#  TAHAP 7 — NOVELTY 3: SHAP EXPLAINABILITY
# =============================================================================

print("\n[7/8] Novelty 3 — SHAP Explainability...")

xgb_shap = xgb.XGBClassifier(n_estimators=100, random_state=RANDOM_SEED,
                               eval_metric='logloss', n_jobs=-1)
xgb_shap.fit(X_train_h, y_train)

explainer     = shap.TreeExplainer(xgb_shap)
shap_sample   = X_test_h.iloc[:SHAP_SAMPLE]
shap_values   = explainer.shap_values(shap_sample)
mean_abs_shap = np.abs(shap_values).mean(axis=0)

print(f"  SHAP dihitung untuk {SHAP_SAMPLE} sampel test")

# =============================================================================
#  RINGKASAN METRIK
# =============================================================================

print("\n" + "=" * 65)
print("  RINGKASAN METRIK EVALUASI")
print("=" * 65)

model_labels = [
    'XGB+ANOVA (Baseline)',
    'XGBoost + Hybrid FS',
    'Random Forest + Hybrid FS',
    'LightGBM + Hybrid FS',
    'Stacking + Hybrid FS (Proposed ★)',
]
results_list = [r_base, r_xgb, r_rf, r_lgb, r_stack]

print(f"\n  {'Model':<34} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>8}")
print("  " + "-" * 60)
for lbl, r in zip(model_labels, results_list):
    mark = ' ★' if 'Proposed' in lbl else '  '
    print(f"  {lbl:<34}{mark} {r['accuracy']:>6.4f} {r['precision']:>7.4f} "
          f"{r['recall']:>7.4f} {r['f1']:>7.4f} {r['auc']:>8.4f}")

print(f"\n  5-Fold CV F1 (training only): "
      f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# =============================================================================
#  TAHAP 8 — 9 VISUALISASI INDIVIDUAL
# =============================================================================

print("\n[8/8] Membuat 9 Visualisasi...")

ml_plot = [
    'XGB+ANOVA\n(Baseline)',
    'XGBoost\n+Hybrid FS',
    'Random Forest\n+Hybrid FS',
    'LightGBM\n+Hybrid FS',
    'Stacking+Hybrid\n(Proposed ★)',
]

f1_vals   = [r['f1']        for r in results_list]
auc_vals  = [r['auc']       for r in results_list]
acc_vals  = [r['accuracy']  for r in results_list]
prec_vals = [r['precision'] for r in results_list]
rec_vals  = [r['recall']    for r in results_list]

# ── Plot 1 — Distribusi Kelas ────────────────────────────────────────────────
print("  [1/9] Distribusi Kelas")
fig, ax = base_fig(8, 5)
counts       = y.value_counts().sort_index()
class_labels = ['Benign (0)', 'Malware (1)']
bars = ax.bar(class_labels, counts.values,
              color=['#6bcb77', '#ff6b6b'], width=0.45, edgecolor='none')
for b, c in zip(bars, counts.values):
    pct = c / counts.sum() * 100
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 400,
            f'{c:,}\n({pct:.1f}%)', ha='center', color='black',
            fontsize=12, fontweight='bold')
ax.set_ylim(0, counts.max() * 1.22)
ax.set_ylabel('Jumlah Sampel', fontsize=11, color='#555')
ax.set_title('Distribusi Kelas Dataset SOMLAP',
             fontsize=14, fontweight='bold', color='black', pad=15)
ax.tick_params(labelsize=12)
fig.tight_layout()
save_fig(fig, '01_distribusi_kelas.png')

# ── Plot 2 — Hybrid FS: Skor 3 Metode (Top-15) ───────────────────────────────
print("  [2/9] Hybrid FS — Skor 3 Metode")
fig, ax = base_fig(14, 6)
top_15 = hybrid_features[:15]
x      = np.arange(len(top_15))
w      = 0.25
ax.bar(x - w, [hfs.ar_n_[f] for f in top_15], w,
       label='ANOVA F-test',       color='#5bc8f5', alpha=0.9)
ax.bar(x,     [hfs.mr_n_[f] for f in top_15], w,
       label='Mutual Information', color='#ffd93d', alpha=0.9)
ax.bar(x + w, [hfs.xi_n_[f] for f in top_15], w,
       label='XGBoost Importance', color='#ff6b6b', alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([f[:9] for f in top_15], rotation=40, ha='right',
                    fontsize=9, color='black')
ax.set_ylabel('Normalized Score [0–1]', fontsize=11, color='#555')
ax.set_ylim(0, 1.22)
ax.set_title('Hybrid Feature Selection: Skor per Metode (Top-15 Fitur)',
             fontsize=13, fontweight='bold', color='black', pad=15)
leg = ax.legend(fontsize=10)
fig.tight_layout()
save_fig(fig, '02_hybrid_fs_skor.png')

# ── Plot 3 — Hybrid FS: Combined Ranking Score (Top-20) ──────────────────────
print("  [3/9] Combined Ranking Score")
fig, ax = base_fig(12, 7)
top_20     = hfs.combined_score_.nlargest(20)
vals_top20 = top_20.values
colors_r   = plt.cm.plasma(np.linspace(0.1, 0.9, 20))
ax.barh(range(20), vals_top20[::-1], color=colors_r, height=0.65)
ax.set_yticks(range(20))
ax.set_yticklabels(top_20.index[::-1], fontsize=9.5)
ax.set_xlabel('Combined Ranking Score  (ANOVA + MI + XGB) / 3',
              fontsize=11, color='#555')
ax.set_title('Hybrid FS: Combined Ranking Score (Top-20)',
             fontsize=13, fontweight='bold', color='black', pad=15)
for i, v in enumerate(vals_top20[::-1]):
    ax.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=8.5, color='black')
ax.axvline(vals_top20.mean(), color='#ffd93d', linestyle='--', lw=1.5,
           label=f'Rata-rata = {vals_top20.mean():.3f}')
ax.legend(fontsize=9)
fig.tight_layout()
save_fig(fig, '03_hybrid_fs_ranking.png')

# ── Plot 4 — Perbandingan F1-Score ───────────────────────────────────────────
print("  [4/9] Perbandingan F1-Score")
fig, ax = base_fig(12, 6)
bars = ax.bar(ml_plot, [v * 100 for v in f1_vals],
              color=PAL, width=0.50, edgecolor='none')
bars[-1].set_edgecolor('#ff6b6b'); bars[-1].set_linewidth(2.5)
for b, v in zip(bars, f1_vals):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
            f'{v * 100:.3f}%', ha='center', color='black',
            fontsize=9.5, fontweight='bold')
ax.set_ylim(min(f1_vals) * 100 - 1, 100.6)
ax.set_ylabel('F1-Score (%)', fontsize=11, color='#555')
ax.set_title('Perbandingan F1-Score Semua Model',
             fontsize=13, fontweight='bold', color='black', pad=15)
ax.axhline(f1_vals[-1] * 100, color='#ff6b6b',
           linestyle='--', lw=1.2, alpha=0.5, label='Stacking (Proposed)')
ax.tick_params(labelsize=9)
ax.legend(fontsize=9)
fig.tight_layout()
save_fig(fig, '04_perbandingan_f1.png')

# ── Plot 5 — Perbandingan Semua Metrik (Grouped Bar) ─────────────────────────
print("  [5/9] Perbandingan Semua Metrik")
fig, ax = base_fig(15, 6)
metrics_dict = {
    'Accuracy' : acc_vals,
    'Precision': prec_vals,
    'Recall'   : rec_vals,
    'F1'       : f1_vals,
    'AUC'      : auc_vals,
}
met_colors = ['#5bc8f5', '#ffd93d', '#6bcb77', '#ff6b6b', '#c77dff']
x = np.arange(len(ml_plot)); w = 0.15
for i, (met_name, vals) in enumerate(metrics_dict.items()):
    ax.bar(x + (i - 2) * w, [v * 100 for v in vals], w,
           label=met_name, color=met_colors[i], alpha=0.88)
ax.set_xticks(x); ax.set_xticklabels(ml_plot, fontsize=8.5)
ax.set_ylabel('Score (%)', fontsize=11, color='#555')
ymin = min(min(v) for v in metrics_dict.values()) * 100 - 2
ax.set_ylim(max(ymin, 85), 101)
ax.set_title('Perbandingan Semua Metrik Evaluasi',
             fontsize=13, fontweight='bold', color='black', pad=15)
ax.legend(fontsize=9, loc='lower right')
ax.tick_params(labelsize=8.5)
fig.tight_layout()
save_fig(fig, '05_perbandingan_semua_metrik.png')

# ── Plot 6 — ROC Curves ──────────────────────────────────────────────────────
print("  [6/9] ROC Curves")
fig, ax = base_fig(9, 7)
roc_data = [
    ('Baseline (XGB+ANOVA)',   r_base['y_prob'],  '#aaaaaa'),
    ('XGBoost + Hybrid FS',    r_xgb['y_prob'],   '#5bc8f5'),
    ('Random Forest + Hybrid', r_rf['y_prob'],    '#6bcb77'),
    ('LightGBM + Hybrid',      r_lgb['y_prob'],   '#ffd93d'),
    ('Stacking + Hybrid ★',    r_stack['y_prob'], '#ff6b6b'),
]
for name, prob, c in roc_data:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_v = roc_auc_score(y_test, prob)
    lw    = 2.8 if '★' in name else 1.6
    ax.plot(fpr, tpr, color=c, lw=lw, label=f'{name}  (AUC={auc_v:.6f})')
ax.plot([0, 1], [0, 1], '--', color='#aaa', lw=1.2, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=11, color='#555')
ax.set_ylabel('True Positive Rate', fontsize=11, color='#555')
ax.set_title('ROC Curves Semua Model',
             fontsize=13, fontweight='bold', color='black', pad=15)
ax.legend(fontsize=8.5, loc='lower right')
fig.tight_layout()
save_fig(fig, '06_roc_curves.png')

# ── Plot 7 — Confusion Matrix ────────────────────────────────────────────────
print("  [7/9] Confusion Matrix")
fig, ax = base_fig(7, 6)
cm = confusion_matrix(y_test, r_stack['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
            xticklabels=['Benign', 'Malware'],
            yticklabels=['Benign', 'Malware'],
            linewidths=1.5, annot_kws={'size': 18, 'weight': 'bold'},
            cbar_kws={'shrink': 0.8})
ax.set_xlabel('Predicted Label', fontsize=11, color='#555')
ax.set_ylabel('Actual Label',    fontsize=11, color='#555')
ax.set_title('Confusion Matrix\nStacking Ensemble + Hybrid FS (Proposed)',
             fontsize=12, fontweight='bold', color='black', pad=15)
ax.tick_params(colors='black', labelsize=11)
tn, fp, fn, tp = cm.ravel()
ax.text(0.5, -0.14, f'TP={tp:,}   TN={tn:,}   FP={fp:,}   FN={fn:,}',
        transform=ax.transAxes, ha='center', color='#888', fontsize=9)
fig.tight_layout()
save_fig(fig, '07_confusion_matrix.png')

# ── Plot 8 — SHAP Feature Importance ─────────────────────────────────────────
print("  [8/9] SHAP Feature Importance")
fig, ax = base_fig(12, 8)
n_top       = 20
top_idx_s   = np.argsort(mean_abs_shap)[-n_top:]
top_names_s = [hybrid_features[i] for i in top_idx_s]
top_vals_s  = mean_abs_shap[top_idx_s]
colors_s    = plt.cm.plasma(np.linspace(0.1, 0.9, n_top))
ax.barh(range(n_top), top_vals_s, color=colors_s, height=0.65)
ax.set_yticks(range(n_top))
ax.set_yticklabels(top_names_s, fontsize=9.5)
ax.set_xlabel('Mean |SHAP value|', fontsize=11, color='#555')
ax.set_title('SHAP: Top-20 Fitur Paling Berpengaruh\n(XGBoost + Hybrid FS)',
             fontsize=13, fontweight='bold', color='black', pad=15)
mean_s = top_vals_s.mean()
ax.axvline(mean_s, color='#ffd93d', linestyle='--', lw=1.5,
           label=f'Rata-rata = {mean_s:.4f}')
for i, v in enumerate(top_vals_s):
    ax.text(v + top_vals_s.max() * 0.01, i,
            f'{v:.4f}', va='center', fontsize=8.5, color='black')
ax.legend(fontsize=9)
fig.tight_layout()
save_fig(fig, '08_shap_feature_importance.png')

# ── Plot 9 — Heatmap Semua Metrik ────────────────────────────────────────────
print("  [9/9] Heatmap Semua Metrik")
fig, ax = base_fig(10, 5)
heat_data = [
    [r['f1'], r['auc'], r['accuracy'], r['precision'], r['recall']]
    for r in results_list
]
row_labels = ['Baseline', 'XGB+Hybrid', 'RF+Hybrid', 'LGB+Hybrid', 'Stacking ★']
col_labels = ['F1', 'AUC', 'Accuracy', 'Precision', 'Recall']
df_heat    = pd.DataFrame(heat_data, index=row_labels, columns=col_labels)
sns.heatmap(df_heat, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax,
            linewidths=0.8, annot_kws={'size': 10, 'weight': 'bold'},
            cbar_kws={'shrink': 0.85})
ax.set_title('Heatmap Perbandingan Semua Metrik Evaluasi',
             fontsize=13, fontweight='bold', color='black', pad=15)
ax.tick_params(colors='black', labelsize=10)
fig.tight_layout()
save_fig(fig, '09_metrics_heatmap.png')

# =============================================================================
#  SELESAI
# =============================================================================

print("\n" + "=" * 65)
print("  PIPELINE SELESAI — Bebas Data Leakage")
print("=" * 65)
print(f"""
  Ringkasan Pipeline (urutan yang benar):
    1. Load data & drop duplicates
    2. Train/Test Split PERTAMA          ← kunci bebas leakage
    3. VarianceThreshold.fit(X_train)    ← fit di train saja
    4. HybridFeatureSelector.fit(train)  ← fit di train saja
    5. Training 5 model
    6. Cross-Validation di X_train saja  ← tidak pakai X_test
    7. SHAP pada model yang dilatih
    8. 9 Visualisasi individual

  Hasil terbaik: Stacking + Hybrid FS
    F1       : {r_stack['f1']:.4f}
    AUC      : {r_stack['auc']:.6f}
    Accuracy : {r_stack['accuracy']:.4f}
    CV F1    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}
""")