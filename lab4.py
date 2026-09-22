from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats


DATA_FILE = Path(__file__).with_name('processed_iris.csv')
OUTPUT_DIR = Path(__file__).parent


# 1. Деректерді жүктеу және қажетті бағандарды таңдау
df = pd.read_csv(DATA_FILE)
required_columns = {'species', 'sepal_length'}
missing_columns = required_columns - set(df.columns)
if missing_columns:
	raise ValueError(f'CSV файлында қажетті бағандар жоқ: {sorted(missing_columns)}')

data = df[['species', 'sepal_length']].copy()
data['sepal_length'] = pd.to_numeric(data['sepal_length'], errors='coerce')
missing_values = data.isna().sum()
print('=== 1. Деректерді жүктеу және missing values ===')
print(f'Форма деректер: {data.shape}')
print(f'Бағандар: {data.columns.tolist()}')
print('\nMissing values:')
print(missing_values.to_string())

if missing_values.any():
	data = data.dropna().copy()
	print(f'Пропустары бар жолдар алынып тасталды. Қалған жолдар: {len(data)}')

# 2. Әр species үшін сипаттамалық статистика
summary = (
	data.groupby('species')['sepal_length']
	.agg(count='count', mean='mean', median='median', std='std')
	.reset_index()
)
print('\n=== 2. Сипаттамалық статистика ===')
print(summary.round(4).to_string(index=False))
summary.to_csv(OUTPUT_DIR / 'iris_sepal_length_summary.csv', index=False)

# 3. Boxplot және stripplot
sns.set_theme(style='whitegrid')
plt.figure(figsize=(9, 6))
sns.boxplot(data=data, x='species', y='sepal_length', color='#9ecae1', width=0.55)
plt.title('Species × Sepal Length: boxplot')
plt.xlabel('Species')
plt.ylabel('Sepal Length')
plt.tight_layout()
boxplot_file = OUTPUT_DIR / 'iris_sepal_length_boxplot.png'
plt.savefig(boxplot_file, dpi=200)
plt.close()

plt.figure(figsize=(9, 6))
sns.stripplot(
	data=data,
	x='species',
	y='sepal_length',
	color='#173f5f',
	alpha=0.75,
	jitter=0.18,
	size=4,
)
plt.title('Species × Sepal Length: stripplot')
plt.xlabel('Species')
plt.ylabel('Sepal Length')
plt.tight_layout()
stripplot_file = OUTPUT_DIR / 'iris_sepal_length_stripplot.png'
plt.savefig(stripplot_file, dpi=200)
plt.close()

# 4. Гипотезалар
print('\n=== 4. Гипотезалар ===')
print('H0: барлық Iris түрлерінің Sepal Length орташа мәндері тең.')
print('H1: кемінде бір Iris түрінің орташа мәні өзгеше.')

groups = [group['sepal_length'].to_numpy() for _, group in data.groupby('species')]
species = data['species'].drop_duplicates().tolist()

# 5. Қалыптылық және дисперсиялардың теңдігі
print('\n=== 5. Shapiro-Wilk және Levene тесттері ===')
shapiro_results = []
for name, group in data.groupby('species'):
	statistic, p_value = stats.shapiro(group['sepal_length'])
	shapiro_results.append({'species': name, 'W': statistic, 'p_value': p_value})
shapiro_table = pd.DataFrame(shapiro_results)
print(shapiro_table.round(4).to_string(index=False))
shapiro_table.to_csv(OUTPUT_DIR / 'iris_sepal_length_shapiro.csv', index=False)

levene_statistic, levene_p_value = stats.levene(*groups, center='median')
print(f'\nLevene: W={levene_statistic:.4f}, p={levene_p_value:.4f}')
if levene_p_value < 0.05:
	print('Ескерту: дисперсиялар біртекті емес, сондықтан Welch ANOVA нәтижесі де беріледі.')

# 6. Бір факторлы ANOVA
anova_statistic, anova_p_value = stats.f_oneway(*groups)
anova_table = pd.DataFrame({
	'test': ['One-way ANOVA'],
	'F': [anova_statistic],
	'p_value': [anova_p_value],
})
print('\n=== 6. One-way ANOVA ===')
print(f'F = {anova_statistic:.4f}, p = {anova_p_value:.4f}')
anova_table.to_csv(OUTPUT_DIR / 'iris_sepal_length_anova.csv', index=False)

# Лекциядағы effect size және дисперсиялар тең болмаған жағдайдағы Welch ANOVA
grand_mean = data['sepal_length'].mean()
between_sum_squares = sum(
	len(group) * (group['sepal_length'].mean() - grand_mean) ** 2
	for _, group in data.groupby('species')
)
total_sum_squares = ((data['sepal_length'] - grand_mean) ** 2).sum()
eta_squared = between_sum_squares / total_sum_squares
welch_statistic, welch_p_value = stats.f_oneway(*groups, equal_var=False)
print(f'Effect size (η²) = {eta_squared:.4f}')
print(f'Welch ANOVA: F = {welch_statistic:.4f}, p = {welch_p_value:.4f}')
welch_table = pd.DataFrame({
	'test': ['Welch ANOVA'],
	'F': [welch_statistic],
	'p_value': [welch_p_value],
	'eta_squared': [eta_squared],
})
welch_table.to_csv(OUTPUT_DIR / 'iris_sepal_length_welch_anova.csv', index=False)

# 7. Tukey HSD post-hoc test, only when ANOVA is significant
print('\n=== 7. Tukey HSD ===')
tukey_table = None
if anova_p_value < 0.05:
	tukey_result = stats.tukey_hsd(*groups)
	tukey_table = pd.DataFrame(
		tukey_result.pvalue,
		index=species,
		columns=species,
	)
	print('Мәнді жұптар (p < 0.05):')
	significant_pairs = []
	for first_index in range(len(species)):
		for second_index in range(first_index + 1, len(species)):
			pair_p_value = tukey_result.pvalue[first_index, second_index]
			if pair_p_value < 0.05:
				pair = (species[first_index], species[second_index], pair_p_value)
				significant_pairs.append(pair)
				print(f'{pair[0]} – {pair[1]}: p={pair[2]:.4f}')
	if not significant_pairs:
		print('Мәнді айырмашылығы бар жұп табылмады.')
	tukey_table.to_csv(OUTPUT_DIR / 'iris_sepal_length_tukey_pvalues.csv')
else:
	print('ANOVA мәнді емес, сондықтан Tukey HSD жүргізілмейді.')

# 8. Биологиялық мазмұндағы қорытынды
means = summary.set_index('species')['mean']
largest_mean_species = means.idxmax()
smallest_mean_species = means.idxmin()
print('\n=== 8. Биологиялық қорытынды ===')
if anova_p_value < 0.05:
	print(
		f'ANOVA нәтижесі Iris түрлерінің Sepal Length орташа мәндері арасында '
		f'статистикалық маңызды айырмашылық бар екенін көрсетті (F={anova_statistic:.4f}, '
		f'p={anova_p_value:.4f}).'
	)
	print(
		f'Орташа сепал ұзындығы ең үлкен түр: {largest_mean_species} '
		f'({means[largest_mean_species]:.2f}), ең кіші түр: {smallest_mean_species} '
		f'({means[smallest_mean_species]:.2f}).'
	)
	print('Tukey HSD нәтижесі айырмашылықтың қай түрлер жұбы арасында орналасқанын нақтылады.')
	if levene_p_value < 0.05:
		print('Levene тесті дисперсиялардың тең еместігін көрсетті; Welch ANOVA бұл нәтижені растады.')
	print(f'η²={eta_squared:.4f}, яғни Sepal Length өзгергіштігінің осы үлесі species факторымен байланысты.')
	print('Бұл белгі Iris түрлерін морфологиялық тұрғыдан ажыратуға пайдалы екенін көрсетеді.')
else:
	print('ANOVA нәтижесі түрлердің Sepal Length орташа мәндері арасында маңызды айырмашылық таппады.')

print(
	f'\nСақталған файлдар: {boxplot_file.name}, {stripplot_file.name}, '
	'iris_sepal_length_summary.csv, iris_sepal_length_anova.csv, '
	'iris_sepal_length_welch_anova.csv'
)
