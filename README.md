# Класифікація SMS: спам / не спам

Навчальний проєкт з курсу нейронних мереж (тема 6).

Задача: за текстом SMS-повідомлення визначити, чи це спам. Порівнюються базова лінія
(TF-IDF + логістична регресія) і невелика нейромережа (ембединги слів, усереднення, повнозв'язні шари),
а також LSTM як додатковий експеримент.

## Дані

SMS Spam Collection, UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

5 574 повідомлення англійською, 747 з них спам. Після видалення дублікатів лишається 5 159.
Дані діляться на train / val / test (70 / 15 / 15) зі збереженням частки спаму.

Almeida T. A., Gómez Hidalgo J. M., Yamakami A. Contributions to the Study of SMS Spam Filtering:
New Collection and Results. Proc. ACM DocEng, 2011.

## Запуск

Python 3.12.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m src.data                      # завантаження і поділ даних
python -m src.train --config base --seed 0
python -m scripts.run_experiments base e3_lr_1e-4 e3_lr_1e-2   # серія конфігів x 5 seed
python -m scripts.summarize             # таблиця результатів на val
python -m scripts.final_test            # оцінка на test (один раз)
python -m scripts.make_figures

python demo.py "Free entry in 2 a wkly comp to win FA Cup final tkts"
```

## Структура

- `configs/` - конфігурації запусків
- `src/` - дані, словник, моделі, навчання, метрики
- `scripts/` - запуск експериментів, таблиці, графіки, фінальна оцінка
- `results/` - логи і метрики кожного запуску, `summary.md`, `test_metrics.csv`, `figures/`
- `models/` - фінальна модель для `demo.py`
