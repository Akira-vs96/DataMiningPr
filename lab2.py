import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Датасетті жүктеу және тексеру
df = pd.read_csv("processed_advertising.csv")
print("--- 1. Датасет туралы ақпарат ---")
print("Бағандар:", df.columns.tolist())
print(df.info())

# 2. Айнымалылар: TV және Sales (медиа инвестициялар мен сатылым)
x = df['TV']
y = df['Sales']

# 3. Бос мәндер мен outlier тексеру
print("\n--- 3. Бос мәндер саны ---")
print(df[['TV', 'Radio', 'Newspaper', 'Sales']].isnull().sum())

# 4. Графикалық талдау: TV vs Sales
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='TV', y='Sales', s=80, color='royalblue')
plt.title('TV spending vs Sales')
plt.xlabel('TV реклаmasına кеткен шығын (TV)')
plt.ylabel('Сатылым (Sales)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('advertising_scatterplot.png')
plt.show()

# 5, 6. Pearson, Spearman және Kendall коэффициенттерін есептеу
r, p_r = pearsonr(x, y)
rho, p_rho = spearmanr(x, y)
tau, p_tau = kendalltau(x, y)

results_df = pd.DataFrame({
    'Method': ['Pearson r', 'Spearman rho', 'Kendall tau'],
    'Coefficient': [r, rho, tau],
    'p_value': [p_r, p_rho, p_tau]
})
print("\n--- Корреляция нәтижелері (TV vs Sales) ---")
print(results_df)

# 7. Correlation Heatmap (Барлық сандық бағандар үшін)
plt.figure(figsize=(8, 6))
num_cols = ['TV', 'Radio', 'Newspaper', 'Sales']
corr_matrix = df[num_cols].corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar=True)
plt.title('Advertising Dataset - Pearson Correlation Heatmap')
plt.savefig('advertising_heatmap.png')
plt.show()

# Арнайы тапсырма: әрбір медиа-арна мен Sales арасындағы Pearson корреляциясы
print("\n--- Әрбір медиа-арна бойынша Pearson корреляциясы ---")
for col in ['TV', 'Radio', 'Newspaper']:
    r_group, _ = pearsonr(df[col], df['Sales'])
    print(f"{col} vs Sales: Pearson r = {r_group:.3f}")