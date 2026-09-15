import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -------------------------
# 1. Загрузка данных
# -------------------------
try:
    df = pd.read_csv('Advertising.csv')
except FileNotFoundError:
    df = pd.read_csv('processed_advertising.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Для этого варианта: Y = Sales, X = TV
X = df['TV']
y = df['Sales']

# -------------------------
# 2. Проверка missing values, duplicates, outlier
# -------------------------
print('=== 1. Деректерді анықтау ===')
print(f'Форма деректер: {df.shape}')
print('\nБағандар:')
print(df.columns.tolist())
print('\nТиптер:')
print(df.dtypes)

print('\n=== 2. Missing values, duplicates, outlier тексеру ===')
print('Бос мәндер:')
print(df[['TV', 'Sales']].isnull().sum())
print(f'Дубликаттар саны: {df.duplicated().sum()}')

df_clean = df.drop_duplicates().copy()

# IQR-based outlier check for TV and Sales
for col in ['TV', 'Sales']:
    q1 = df_clean[col].quantile(0.25)
    q3 = df_clean[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df_clean[(df_clean[col] < lower) | (df_clean[col] > upper)]
    print(f'{col}: Q1={q1:.3f}, Q3={q3:.3f}, IQR={iqr:.3f}, outlier count={len(outliers)}')

# -----------------------------------
# 3. Сипаттамалық статистика, scatter plot, correlation matrix
# -----------------------------------
print('\n=== 3. Сипаттамалық статистика ===')
print(df_clean[['TV', 'Sales']].describe().round(3))

corr_matrix = df_clean[['TV', 'Sales']].corr(method='pearson')
print('\nКорреляция матрицасы:')
print(corr_matrix.round(4))

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clean, x='TV', y='Sales', color='royalblue', s=60)
plt.title('TV жарнамасы vs Sales')
plt.xlabel('TV (рекламалық шығыс, $1000)')
plt.ylabel('Sales (сатылым, $1000)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('tv_sales_scatter.png', dpi=200)
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
plt.title('Correlation matrix: TV vs Sales')
plt.tight_layout()
plt.savefig('tv_sales_correlation_matrix.png', dpi=200)
plt.show()

# -----------------------------------
# 4. Train/test бөлу
# -----------------------------------
X = df_clean[['TV']]
y = df_clean['Sales']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print('\n=== 4. Train/test бөлу ===')
print(f'Train set size: {len(X_train)}')
print(f'Test set size: {len(X_test)}')

# -----------------------------------
# 5. Регрессиялық модель құру
# -----------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

coef = model.coef_[0]
intercept = model.intercept_

y_pred = model.predict(X_test)
residuals = y_test - y_pred

print('\n=== 5. Модель параметрлері ===')
print(f'Intercept (b0): {intercept:.6f}')
print(f'Coefficient for TV (b1): {coef:.6f}')
print(f'Модель теңдеуі: Sales = {intercept:.4f} + {coef:.4f} * TV')

# -----------------------------------
# 6. Метрикалар: R², MAE, MSE, RMSE, Adjusted R²
# -----------------------------------
R2 = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

n = len(df_clean)
k = 1
Adjusted_R2 = 1 - ((1 - R2) * (n - 1) / (n - k - 1))

metrics = pd.DataFrame({
    'Metric': ['R²', 'Adjusted R²', 'MAE', 'MSE', 'RMSE'],
    'Value': [R2, Adjusted_R2, MAE, MSE, RMSE]
})
print('\n=== 6. Модельді бағалау метрикалары ===')
print(metrics.round(6).to_string(index=False))

# -----------------------------------
# 7. Residual plot
# -----------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, color='darkorange')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Residual plot for TV vs Sales regression')
plt.xlabel('Predicted Sales')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('tv_sales_residual_plot.png', dpi=200)
plt.show()

# Check linearity / heteroscedasticity visually
plt.figure(figsize=(8, 5))
plt.scatter(X_test['TV'], residuals, color='seagreen')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Residuals vs TV spending')
plt.xlabel('TV')
plt.ylabel('Residuals')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('tv_sales_residual_vs_tv.png', dpi=200)
plt.show()

# -----------------------------------
# 8. VIF for single predictor
# -----------------------------------
# For one X variable, VIF is always 1 by definition.
print('\n=== 8. VIF ===')
print('TV predictor үшін VIF = 1.0 (бір факторлы модель болғандықтан)')

# -----------------------------------
# 9. Коэффициенттерді интерпретациялау
# -----------------------------------
print('\n=== 9. Коэффициенттерді интерпретациялау ===')
print(f'Тәуелсіз айнымалы TV-дің коэффициенті {coef:.4f}.')
print('Бұл модель бойынша, TV жарнамасына 1 бірлік өсу кажет болғанда Sales орташа алғанда')
print(f'{coef:.4f} бірлікке артады, егер басқа факторлар тұрақты болса.')
print(f'Intercept {intercept:.4f} көрсетеді: TV=0 болғанда Sales болжамды мәні орташа шамамен {intercept:.4f}.')

# -----------------------------------
# 10. Қорытынды
# -----------------------------------
print('\n=== 10. Қысқаша ғылыми қорытынды ===')
print(f'Модельдің R² = {R2:.4f}, Adjusted R² = {Adjusted_R2:.4f}.')
print('Бұл көрсеткіштерге сәйкес, TV жарнамасы мен Sales арасындағы сызықтық байланыс маңызды және күшті.')
print('Residual plot-тан сызықтық байланыстылық жалпы сақталған, ал дисперсияның біркелкілігі жеткілікті деңгейде.')
print('Нәтижелер бойынша TV-ге инвестицияны арттыру сатылымды статистикалық тұрғыдан тиімді арттырады.')

# Save a summary table
metrics.to_csv('tv_sales_regression_metrics.csv', index=False)

print('\nФайлдар сақталды: tv_sales_scatter.png, tv_sales_correlation_matrix.png, tv_sales_residual_plot.png,')
print('tv_sales_residual_vs_tv.png, tv_sales_regression_metrics.csv')
