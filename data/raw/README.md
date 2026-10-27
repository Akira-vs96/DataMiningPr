# Данные практик 10–15, вариант 1

Исходные WDBC и Penguins используются прежними работами. Для новых работ
сохранены следующие реальные открытые данные; повторный запуск не обращается к сети.

| Файл | Практики | Источник и единицы |
| --- | --- | --- |
| [electricity_monthly_eia.json](electricity_monthly_eia.json) | 10 | EIA Electricity Retail Sales, US / ALL, месячные продажи конечным потребителям, млн кВт·ч = GWh |
| [elecdaily.csv](elecdaily.csv) | 12, 15 | fpp2 / AEMO / BOM: Victoria, Australia, 365 дней 2014 года; Demand — исходный дневной агрегат, Temperature — °C, WorkDay — 0/1 |
| [scientific_abstracts_openalex.json](scientific_abstracts_openalex.json) | 14 | OpenAlex: 60 статей 2024 года с аннотациями; фильтр English дополнительно проверяется при обработке |

Загрузка: 2026-10-27. SHA-256 точных файлов и версии библиотек записываются
в `environment.json` каждой работы.

## Электропотребление

Официальный [EIA Data Browser](https://www.eia.gov/electricity/data/browser/).
API: `https://api.eia.gov/v2/electricity/retail-sales/data/`;
параметры: `api_key=DEMO_KEY`, `frequency=monthly`, `data[0]=sales`,
`facets[stateid][]=US`, `facets[sectorid][]=ALL`,
`sort[0][column]=period`, `sort[0][direction]=asc`, `offset=0`, `length=5000`.
Продажи конечным потребителям — операционное измерение потребления;
оно не включает все возможные формы собственного производства за счётчиком.
Данные могут уточняться источником; сохранённый JSON фиксирует использованный снимок.

CSV дневного ряда:
<https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/elecdaily.csv>.
[Описание fpp2](https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/elecdaily.html)
задаёт календарь 2014 года; в CSV даты отсутствуют, поэтому их восстанавливает загрузчик.
Описание обозначает Demand как GW даже для суточного агрегата. В работах сохранена
исходная шкала, пересчёт этого агрегата в кВт·ч не заявляется.
Дневная температура фактическая: для прогноза до начала дня потребуется отдельный погодный прогноз.

В №12 после индекса 200 дневной Demand умножается на 0.75. Это **искусственный
контролируемый сценарий**, а не найденный факт изменения тарифа в исходных данных.
Исходные и изменённые значения сохраняются рядом для проверки.
№15 использует исходные значения без этого вмешательства.

## Научные аннотации

API: <https://api.openalex.org/works>.
Параметры: `filter=has_abstract:true,publication_year:2024,type:article,language:en`,
`sample=60`, `seed=42`, `per-page=60`,
`select=id,title,doi,abstract_inverted_index,primary_topic`.
Аннотации восстановлены по позициям слов `abstract_inverted_index`.
Исключённые неанглийские записи сохраняются в результатах №14;
DOI и OpenAlex ID каждого использованного документа сохраняются в корпусе.
Выборка небольшая и случайная, поэтому не характеризует мировую науку.

Wine и Karate Club входят в scikit-learn и NetworkX соответственно и не требуют загрузки.
