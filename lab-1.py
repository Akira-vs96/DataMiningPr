import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1) Датасетті жүктеу, өлшемін, баған атауларын және типтерін анықтау
# Advertising.csv құрылымы: TV, Radio, Newspaper, Sales
raw_df = pd.read_csv("Advertising.csv")

# Егер файлде индекс бағана берілген болса, оны алып тастаймыз
if raw_df.columns[0].startswith("Unnamed"):
    raw_df = raw_df.iloc[:, 1:]

df = raw_df.copy()
numeric_cols = ['TV', 'Radio', 'Newspaper', 'Sales']

print("--- 1. Деректерді жүктеу және талдау ---")
print("Деректер өлшемі (жол, баған):", df.shape)
print("Баған атаулары:", df.columns.tolist())
print("Әр бағанның деректер типі:\n", df.dtypes)

# 2) Бос мәндерді анықтап, сандықтарды медианамен толтыру
print("\n--- 2. Бос мәндерді өңдеу ---")
print("Бос мәндердің бастапқы саны:\n", df.isnull().sum())

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# 3) Қайталанатын жолдарды анықтап жою
print("\n--- 3. Қайталанатын жолдарды тазалау ---")
print("Қайталанатын жолдар саны:", df.duplicated().sum())
df = df.drop_duplicates()

# 4) Сандық белгілердегі шектен тыс мәндерді IQR әдісімен анықтап өңдеу (Clipping)
print("\n--- 4. IQR әдісі арқылы шектен тыс мәндерді реттеу ---")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower_bound, upper_bound)
    print(f"{col}: [Q1={Q1}, Q3={Q3}, IQR={IQR}] — өңделді/шектелді.")

# 4-ші тапсырмаға арналған Box-plot графигін салу және сақтау
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numeric_cols], palette="Set2")
plt.title('Advertising Dataset — IQR әдісімен өңделген box-plot')
plt.xlabel('Сандық айнымалылар')
plt.ylabel('Мәндер шкаласы')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('advertising_iqr_boxplot.png')
plt.show()

# 5) Сандық белгілерді стандарттау және сақтау
print("\n--- 5. Стандарттау және Сақтау ---")
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df[numeric_cols]),
    columns=[f"{c}_scaled" for c in numeric_cols]
)

processed_df = pd.concat([df.reset_index(drop=True), df_scaled], axis=1)

output_filename = "processed_advertising.csv"
processed_df.to_csv(output_filename, index=False)
print(f"\nӨңделген деректер сәтті түрде '{output_filename}' файлына сақталды!")
print(processed_df.head())