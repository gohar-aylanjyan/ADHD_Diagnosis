import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("🧠 ASRS RANDOM FOREST ԴԱՍԱԿԱՐԳԻՉ - OVERFITTING ԿԱՆԽԱՐԳԵԼՈՒՄ")
print("=" * 80)

# =============================================
# 1. ԲԵՌՆԵԼ ՏՎՅԱԼՆԵՐԸ
# =============================================

print("\n📂 Քայլ 1: Տվյալների բեռնում")
print("-" * 80)

try:
    df = pd.read_csv('ASRS_dataset_1000_rows.csv')
    print(f"✅ Բազան հաջողությամբ բեռնված է: {df.shape[0]} տող × {df.shape[1]} սյունակ")
except FileNotFoundError:
    print("❌ Ֆայլը չի գտնվել։ Խնդրում ենք նախ գործարկել Step 1 ծրագիրը։")
    exit()


# =============================================
# 2. ԿԱՏԵԳՈՐԻԱՆԵՐԻ ՍՏԵՂԾՈՒՄ
# =============================================

def classify_partA(score):
    if 0 <= score <= 9:
        return 'Low Negative'
    elif 10 <= score <= 13:
        return 'High Negative'
    elif 14 <= score <= 17:
        return 'Low Positive'
    elif 18 <= score <= 24:
        return 'High Positive'
    else:
        return 'Unknown'


df['Category'] = df['PartA'].apply(classify_partA)

print("\nԿատեգորիաների բաշխում:")
category_dist = df['Category'].value_counts().sort_index()
for cat, count in category_dist.items():
    print(f"  {cat:20s}: {count:4d} ({count / len(df) * 100:5.1f}%)")

# =============================================
# 3. ՀԱՏԿԱՆԻՇՆԵՐԻ ՊԱՏՐԱՍՏՈՒՄ
# =============================================

feature_columns = ['Part_A_Q1', 'Part_A_Q2', 'Part_A_Q3', 'Part_A_Q4', 'Part_A_Q5', 'Part_A_Q6']
X = df[feature_columns]
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Ուսուցման բազմություն: {len(X_train)} | Թեստի բազմություն: {len(X_test)}")

# =============================================
# 4. OVERFITTING ԿԱՆԽԱՐԳԵԼՈՒՄ
# =============================================

print("\n" + "=" * 80)
print("🛡️ Քայլ 2: Overfitting-ի կանխարգելում")
print("=" * 80)

print("\n🔍 Թեստավորում ենք տարբեր կոնֆիգուրացիաներ...\n")

# Թեստային կոնֆիգուրացիաներ
configs = [
    {
        'name': 'Baseline (հիմնական)',
        'params': {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'class_weight': 'balanced'
        }
    },
    {
        'name': 'Light Regularization',
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'max_features': 'sqrt',
            'class_weight': 'balanced'
        }
    },
    {
        'name': 'Medium Regularization',
        'params': {
            'n_estimators': 150,
            'max_depth': 8,
            'min_samples_split': 15,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'class_weight': 'balanced'
        }
    },
    {
        'name': 'Strong Regularization',
        'params': {
            'n_estimators': 200,
            'max_depth': 6,
            'min_samples_split': 20,
            'min_samples_leaf': 8,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'max_leaf_nodes': 20
        }
    }
]

results = []

for config in configs:
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **config['params'])
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    gap = train_acc - test_acc

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=10, n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    results.append({
        'name': config['name'],
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': gap,
        'cv_mean': cv_mean,
        'cv_std': cv_std
    })

    status = '✅' if gap <= 0.03 else '⚠️' if gap <= 0.05 else '❌'
    print(f"{status} {config['name']:25s}")
    print(f"     Train: {train_acc:.3f} | Test: {test_acc:.3f} | Gap: {gap:.3f} ({gap * 100:.1f}%)")
    print(f"     CV: {cv_mean:.3f} ± {cv_std:.3f}")
    print()

# Ընտրել լավագույն մոդելը (ամենափոքր gap-ով + բարձր test accuracy)
best_result = min(results, key=lambda x: abs(x['gap']) + (1 - x['test_acc']))
print("=" * 80)
print(f"🏆 ԸՆՏՐՎԱԾ ՄՈԴԵԼ: {best_result['name']}")
print("=" * 80)
print(f"   Train Accuracy: {best_result['train_acc']:.3f} ({best_result['train_acc'] * 100:.1f}%)")
print(f"   Test Accuracy:  {best_result['test_acc']:.3f} ({best_result['test_acc'] * 100:.1f}%)")
print(f"   Gap (Overfitting): {best_result['gap']:.3f} ({best_result['gap'] * 100:.1f}%)")
print(f"   CV Mean: {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}")

rf_model = best_result['model']

# =============================================
# 5. ՄԱՆՐԱՄԱՍՆ ԳՆԱՀԱՏՈՒՄ
# =============================================

print("\n" + "=" * 80)
print("📊 Քայլ 3: Մանրամասն գնահատում")
print("=" * 80)

y_pred = rf_model.predict(X_test)

print("\nԴասակարգման հաշվետվություն:")
print("-" * 80)
print(classification_report(y_test, y_pred, zero_division=0))

# Շփոթության մատրից
cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
print("\nՇփոթության մատրից:")
cm_df = pd.DataFrame(cm,
                     index=rf_model.classes_,
                     columns=rf_model.classes_)
print(cm_df)

# Հատկանիշների կարևորություն
print("\n" + "=" * 80)
print("🔍 Հատկանիշների կարևորություն")
print("=" * 80)
feature_importance = pd.DataFrame({
    'Հարց': feature_columns,
    'Կարևորություն': rf_model.feature_importances_
}).sort_values('Կարևորություն', ascending=False)

for idx, row in feature_importance.iterrows():
    importance = row['Կարևորություն']
    bar = '█' * int(importance * 80)
    print(f"{row['Հարց']:12s}: {importance:.3f} {bar}")

# =============================================
# 6. ՎԻԶՈՒԱԼԱՑՈՒՄ (3 ԳՐԱՖԻԿ)
# =============================================

print("\n" + "=" * 80)
print("📊 Քայլ 4: Վիզուալացում")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('ASRS Random Forest - Overfitting կանխարգելված',
             fontsize=16, fontweight='bold')

# 1. Overfitting համեմատություն
ax1 = axes[0]
model_names = [r['name'] for r in results]
train_accs = [r['train_acc'] for r in results]
test_accs = [r['test_acc'] for r in results]
gaps = [r['gap'] for r in results]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax1.bar(x - width / 2, train_accs, width, label='Train', color='#3498db', alpha=0.8)
bars2 = ax1.bar(x + width / 2, test_accs, width, label='Test', color='#2ecc71', alpha=0.8)

# Ավելացնել gap-ը որպես տեքստ
for i, (train, test, gap) in enumerate(zip(train_accs, test_accs, gaps)):
    ax1.text(i, max(train, test) + 0.01, f'Δ{gap * 100:.1f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold',
             color='red' if gap > 0.05 else 'orange' if gap > 0.03 else 'green')

ax1.set_xlabel('Մոդել', fontsize=11)
ax1.set_ylabel('Ճշտություն', fontsize=11)
ax1.set_title('Train vs Test Accuracy - Overfitting համեմատություն', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([name.split()[0] for name in model_names], rotation=0, ha='center')
ax1.legend()
ax1.set_ylim([0.75, 1.0])
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(0.95, color='gray', linestyle=':', alpha=0.5, label='Target')

# 2. Շփոթության մատրից
ax2 = axes[1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[c.replace(' ', '\n') for c in rf_model.classes_],
            yticklabels=rf_model.classes_,
            ax=ax2, cbar_kws={'label': 'Քանակ'}, annot_kws={'size': 11, 'weight': 'bold'})
ax2.set_title('Շփոթության մատրից', fontsize=12, fontweight='bold')
ax2.set_ylabel('Իրական', fontsize=11)
ax2.set_xlabel('Կանխատեսված', fontsize=11)

# 3. Հատկանիշների կարևորություն
ax3 = axes[2]
colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
ax3.barh(feature_importance['Հարց'], feature_importance['Կարևորություն'],
         color=colors_feat, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Կարևորություն', fontsize=11)
ax3.set_title('Հատկանիշների կարևորություն', fontsize=12, fontweight='bold')
ax3.invert_yaxis()
ax3.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('asrs_rf_no_overfitting.png', dpi=300, bbox_inches='tight')
print("✅ Գրաֆիկները պահպանված են: asrs_rf_no_overfitting.png")

# =============================================
# 7. ԿԱՆԽԱՏԵՍՈՒՄՆԵՐԻ ՊԱՀՊԱՆՈՒՄ
# =============================================

print("\n" + "=" * 80)
print("💾 Քայլ 5: Կանխատեսումների պահպանում")
print("=" * 80)

results_df = df.copy()
results_df['Predicted_Category'] = rf_model.predict(X)
proba = rf_model.predict_proba(X)

for i, cat in enumerate(rf_model.classes_):
    results_df[f'Prob_{cat.replace(" ", "_")}'] = proba[:, i]

results_df['Prediction_Confidence'] = proba.max(axis=1)
results_df['Correct_Prediction'] = (results_df['Category'] == results_df['Predicted_Category'])

output_file = 'ASRS_predictions_no_overfitting.csv'
results_df.to_csv(output_file, index=False)
print(f"✅ Կանխատեսումները պահպանված են: {output_file}")

correct = results_df['Correct_Prediction'].sum()
print(f"   Ընդհանուր ճշտություն: {correct}/{len(results_df)} ({correct / len(results_df):.2%})")

# =============================================
# 8. ԿԱՆԽԱՏԵՍՄԱՆ API
# =============================================

print("\n" + "=" * 80)
print("🎯 Քայլ 6: Նոր դեպքերի կանխատեսում")
print("=" * 80)


def predict_new_case(q1, q2, q3, q4, q5, q6):
    """Կանխատեսել նոր դեպքի կատեգորիան"""
    new_data = pd.DataFrame([[q1, q2, q3, q4, q5, q6]], columns=feature_columns)
    prediction = rf_model.predict(new_data)[0]
    probabilities = rf_model.predict_proba(new_data)[0]

    print(f"\n{'=' * 70}")
    print(f"📝 Մուտք: Q1={q1}, Q2={q2}, Q3={q3}, Q4={q4}, Q5={q5}, Q6={q6}")
    print(f"   Part A գնահատական: {q1 + q2 + q3 + q4 + q5 + q6}/24")
    print(f"\n🎯 Կանխատեսում: {prediction}")
    print(f"   Հավատարմություն: {probabilities.max():.1%}")
    print(f"\n📊 Հավանականություններ:")
    for cat, prob in zip(rf_model.classes_, probabilities):
        bar = '█' * int(prob * 40)
        print(f"   {cat:20s}: {prob:6.1%} {bar}")
    print('=' * 70)

    return prediction


# Օրինակներ
print("\n📌 Օրինակ 1: Ցածր ախտանիշներ")
predict_new_case(1, 1, 0, 2, 1, 0)

print("\n📌 Օրինակ 2: Միջին ախտանիշներ")
predict_new_case(2, 3, 2, 3, 2, 1)

print("\n📌 Օրինակ 3: Բարձր ախտանիշներ")
predict_new_case(4, 4, 4, 3, 4, 2)

# =============================================
# 9. ՎԵՐՋՆԱԿԱՆ ԱՄՓՈՓՈՒՄ
# =============================================

print("\n" + "=" * 80)
print("✅ ԾՐԱԳԻՐԸ ԱՎԱՐՏՎԱԾ Է!")
print("=" * 80)

print(f"\n📊 ՎԵՐՋՆԱԿԱՆ ՄԵՏՐԻԿՆԵՐ")
print("-" * 80)
print(f"🎓 Train Accuracy:     {best_result['train_acc']:.3f} ({best_result['train_acc'] * 100:.1f}%)")
print(f"🧪 Test Accuracy:      {best_result['test_acc']:.3f} ({best_result['test_acc'] * 100:.1f}%)")
print(f"📉 Overfitting Gap:    {best_result['gap']:.3f} ({best_result['gap'] * 100:.1f}%)")
print(f"📊 CV Mean:            {best_result['cv_mean']:.3f} ± {best_result['cv_std']:.3f}")

if best_result['gap'] <= 0.03:
    status = "✅ ԳԵՐԱԶԱՆՑ - Overfitting չկա"
elif best_result['gap'] <= 0.05:
    status = "✅ ԼԱՎ - Թույլ overfitting"
else:
    status = "⚠️ Միջին overfitting"

print(f"\n🏆 Վիճակ: {status}")

print("\n📂 ՍՏԵՂԾՎԱԾ ՖԱՅԼԵՐ:")
print("-" * 80)
print(f"✅ asrs_rf_no_overfitting.png (3 գրաֆիկ)")
print(f"✅ {output_file}")

print("\n💡 ՄՈԴԵԼԻ ՊԱՐԱՄԵՏՐԵՐ:")
print("-" * 80)
for key, value in rf_model.get_params().items():
    if key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'max_leaf_nodes']:
        print(f"   {key:20s}: {value}")

print("\n🎯 Մոդելը պատրաստ է օգտագործման համար!")