"""
Прогноз почасового числа пользователей с отложенной выборкой в последний месяц.

Схема: последние 30 суток отрезаются и НЕ участвуют в обучении.
Модели учатся только на train, делают прогноз на горизонт теста,
затем прогноз сравнивается с реально отрезанным y.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

CSV = "TS (1) (1).csv"
TEST_DAYS = 30
SEASON = 24  # суточная сезонность для почасовых данных


# ---------------------------------------------------------------- загрузка
def load() -> tuple[pd.Series, pd.DatetimeIndex]:
    """Возвращает восстановленный ряд и метки часов, которые были испорчены."""
    # в файле старые mac-переводы строк (\r), поэтому lineterminator задан явно
    df = pd.read_csv(CSV, lineterminator="\r")
    df.columns = [c.strip() for c in df.columns]
    df["Time"] = pd.to_datetime(df["Time"], format="%m/%d/%y %H:%M")
    df["Users"] = pd.to_numeric(df["Users"], errors="coerce")
    s = df.set_index("Time")["Users"].sort_index()

    # нули - это сбои сбора (по 3-6 ночных часов подряд), а не реальный трафик
    zero_idx = s.index[s == 0]
    s = s.replace(0, np.nan)

    # выравниваем сетку на ровный час: один час отсутствует (переход на летнее время)
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    gap_idx = full.difference(s.index)
    s = s.reindex(full)

    # пропуски восстанавливаем по времени, затем сглаживаем края
    s = s.interpolate(method="time", limit_direction="both")
    s.index.freq = "h"

    bad = zero_idx.union(gap_idx)
    print(f"Загружено: {len(s)} часов  {s.index.min()} -> {s.index.max()}")
    print(f"Восстановлено: {len(zero_idx)} нулей + {len(gap_idx)} пропущенных часов")
    return s, bad


# ---------------------------------------------------------------- метрики
def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err / y_true)) * 100
    # sMAPE устойчивее MAPE при близких к нулю значениях
    smape = np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred))) * 100
    r2 = 1 - np.sum(err**2) / np.sum((y_true - y_true.mean()) ** 2)
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape, "sMAPE_%": smape, "R2": r2}


def main() -> None:
    s, bad = load()

    # ------------------------------------------------------------ split
    split = s.index.max() - pd.Timedelta(days=TEST_DAYS)
    train, test = s[s.index <= split], s[s.index > split]
    print(f"\nTrain: {len(train)} ч  {train.index.min()} -> {train.index.max()}")
    print(f"Test:  {len(test)} ч  {test.index.min()} -> {test.index.max()}")
    h = len(test)

    # ------------------------------------------------------------ структура ряда
    dec = seasonal_decompose(train, model="additive", period=SEASON)
    print(f"\nАмплитуда суточной сезонности: {dec.seasonal.max() - dec.seasonal.min():,.0f}")
    print(f"Тренд: {dec.trend.dropna().iloc[0]:,.0f} -> {dec.trend.dropna().iloc[-1]:,.0f}")

    preds: dict[str, pd.Series] = {}

    # ------------------------------------------------------------ baseline
    # seasonal naive: повторяем последние сутки train на весь горизонт
    last_day = train.iloc[-SEASON:].to_numpy()
    preds["Seasonal naive"] = pd.Series(
        np.resize(last_day, h), index=test.index, name="Seasonal naive"
    )

    # ------------------------------------------------------------ SARIMA
    # сезонная разность (D=1, s=24) снимает суточный цикл; лог убирает
    # рост дисперсии вместе с уровнем
    print("\nОбучение SARIMAX(2,1,2)(1,1,1,24) на логарифме...")
    model = SARIMAX(
        np.log(train),
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, SEASON),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=200)
    print(f"AIC={fit.aic:,.1f}  BIC={fit.bic:,.1f}")

    fc = fit.get_forecast(steps=h)
    # обратно из логарифма; для среднего добавлена поправка на дисперсию
    sigma2 = fc.var_pred_mean
    preds["SARIMA"] = pd.Series(
        np.exp(fc.predicted_mean + sigma2 / 2).to_numpy(), index=test.index, name="SARIMA"
    )
    ci = np.exp(fc.conf_int(alpha=0.05).to_numpy())
    lo, hi = ci[:, 0], ci[:, 1]

    # ------------------------------------------------------------ сравнение
    # часы со сбоем сбора внутри теста исключаем: там реальный y был нулём,
    # это не ошибка модели, а дыра в данных
    ok = ~test.index.isin(bad)
    n_bad = int((~ok).sum())
    if n_bad:
        print(f"\nИз оценки исключено {n_bad} испорченных часов теста")

    rows = {name: metrics(test[ok], p[ok]) for name, p in preds.items()}
    table = pd.DataFrame(rows).T.sort_values("MAE")
    print("\n" + "=" * 62)
    print(f"ПРОГНОЗ vs РЕАЛЬНЫЙ ОТРЕЗАННЫЙ Y  ({int(ok.sum())} часов)")
    print("=" * 62)
    print(table.to_string(float_format=lambda v: f"{v:>12,.3f}"))

    cov = float(
        ((test.to_numpy()[ok] >= lo[ok]) & (test.to_numpy()[ok] <= hi[ok])).mean() * 100
    )
    print(f"\nПокрытие 95% интервала SARIMA: {cov:.1f}% (номинально 95%)")

    best = table.index[0]
    print(f"Лучшая модель по MAE: {best}")

    # ошибка по мере удаления от конца train
    err_by_day = (
        (test[ok] - preds[best][ok])
        .abs()
        .groupby((np.arange(h)[ok] // 24) + 1)
        .mean()
    )
    print("\nСредняя абсолютная ошибка по суткам горизонта:")
    for day in [1, 2, 3, 7, 14, 21, 30]:
        if day in err_by_day.index:
            print(f"  сутки {day:>2}: {err_by_day[day]:>10,.0f}")

    # ------------------------------------------------------------ сохранение
    out = pd.DataFrame({"y_true": test})
    for name, p in preds.items():
        out[name] = p
    out["SARIMA_lo95"], out["SARIMA_hi95"] = lo, hi
    out["used_in_metrics"] = ok
    out.index.name = "Time"
    out.to_csv("forecast_vs_actual.csv")
    table.to_csv("metrics.csv")

    # ------------------------------------------------------------ графики
    fig, ax = plt.subplots(2, 1, figsize=(15, 9))

    ax[0].plot(train.index[-14 * 24:], train.iloc[-14 * 24:], color="#4C6EF5",
               lw=1, label="Train (последние 2 недели)")
    ax[0].plot(test.index, test, color="#111827", lw=1.6, label="Реальный y (отрезанный)")
    ax[0].plot(preds["SARIMA"].index, preds["SARIMA"], color="#E8590C",
               lw=1.4, ls="--", label="SARIMA прогноз")
    ax[0].fill_between(test.index, lo, hi, color="#E8590C", alpha=0.15, label="95% интервал")
    ax[0].axvline(split, color="#868E96", ls=":", lw=1.5)
    ax[0].set_title(f"Прогноз на отложенный месяц (MAE={table.loc['SARIMA','MAE']:,.0f})")
    ax[0].set_ylabel("Users")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[0].grid(alpha=0.25)

    ax[1].plot(test.index, test - preds["SARIMA"], color="#C92A2A", lw=0.9)
    ax[1].axhline(0, color="#111827", lw=1)
    ax[1].set_title("Остатки: реальный y − прогноз")
    ax[1].set_ylabel("Ошибка")
    ax[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig("forecast_vs_actual.png", dpi=130)
    print("\nСохранено: forecast_vs_actual.csv, metrics.csv, forecast_vs_actual.png")


if __name__ == "__main__":
    main()
