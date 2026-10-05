"""
Прогноз почасового числа пользователей по схеме train / valid / test.

Три куска вместо двух:
  train  (янв - 15 марта)  - на нём обучаются кандидаты;
  valid  (16-31 марта)     - на нём выбираются параметры;
  test   (апрель)          - трогается один раз в конце, для честной оценки.

После выбора параметров модель переобучается на train+valid и прогнозирует test.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

CSV = "TS (1) (1).csv"
VALID_START = pd.Timestamp("2017-03-16")
TEST_START = pd.Timestamp("2017-04-01")
SEASON = 24

# кандидаты: немного осмысленных вариантов, а не сетка из сотен
CANDIDATES = [
    ((1, 1, 1), (1, 1, 1, SEASON)),
    ((2, 1, 2), (1, 1, 1, SEASON)),
    ((2, 1, 1), (1, 1, 1, SEASON)),
    ((1, 1, 2), (1, 1, 1, SEASON)),
    ((3, 1, 1), (1, 1, 1, SEASON)),
    ((2, 1, 2), (0, 1, 1, SEASON)),
    ((2, 1, 2), (1, 1, 0, SEASON)),
]


# ---------------------------------------------------------------- загрузка
def load() -> tuple[pd.Series, pd.DatetimeIndex]:
    """Возвращает восстановленный ряд и метки испорченных часов."""
    df = pd.read_csv(CSV, lineterminator="\r")
    df.columns = [c.strip() for c in df.columns]
    df["Time"] = pd.to_datetime(df["Time"], format="%m/%d/%y %H:%M")
    df["Users"] = pd.to_numeric(df["Users"], errors="coerce")
    s = df.set_index("Time")["Users"].sort_index()

    # нули - сбои сбора (3-6 ночных часов подряд), а не реальный трафик
    zero_idx = s.index[s == 0]
    s = s.replace(0, np.nan)

    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    gap_idx = full.difference(s.index)
    s = s.reindex(full).interpolate(method="time", limit_direction="both")
    s.index.freq = "h"

    bad = zero_idx.union(gap_idx)
    print(f"Загружено: {len(s)} часов  {s.index.min()} -> {s.index.max()}")
    print(f"Восстановлено: {len(zero_idx)} нулей + {len(gap_idx)} пропущенных часов")
    return s, bad


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err**2)),
        "MAPE_%": np.mean(np.abs(err / y_true)) * 100,
        "sMAPE_%": np.mean(2 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred))) * 100,
        "R2": 1 - np.sum(err**2) / np.sum((y_true - y_true.mean()) ** 2),
    }


def forecast(endog: pd.Series, order, seasonal, steps: int, index: pd.DatetimeIndex):
    """Обучает SARIMA на логарифме и возвращает прогноз с интервалом."""
    fit = SARIMAX(
        np.log(endog),
        order=order,
        seasonal_order=seasonal,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=200)
    fc = fit.get_forecast(steps=steps)
    # обратно из логарифма; +sigma^2/2 - поправка от медианы к среднему
    mean = pd.Series(
        np.exp(fc.predicted_mean + fc.var_pred_mean / 2).to_numpy(), index=index
    )
    ci = np.exp(fc.conf_int(alpha=0.05).to_numpy())
    return fit, mean, ci[:, 0], ci[:, 1]


def main() -> None:
    s, bad = load()

    # ------------------------------------------------------------ три куска
    train = s[s.index < VALID_START]
    valid = s[(s.index >= VALID_START) & (s.index < TEST_START)]
    test = s[s.index >= TEST_START]
    trainval = s[s.index < TEST_START]

    for name, part in [("Train", train), ("Valid", valid), ("Test ", test)]:
        print(f"{name}: {len(part):>4} ч  {part.index.min()} -> {part.index.max()}")

    ok_v = ~valid.index.isin(bad)
    ok_t = ~test.index.isin(bad)

    # ------------------------------------------------------------ подбор на valid
    print("\nПодбор параметров на валидации...")
    rows = []
    for order, seasonal in CANDIDATES:
        fit, pred, _, _ = forecast(train, order, seasonal, len(valid), valid.index)
        rows.append(
            {
                "order": str(order),
                "seasonal": str(seasonal),
                "AIC": fit.aic,
                "valid_MAE": metrics(valid[ok_v], pred[ok_v])["MAE"],
            }
        )
        print(f"  {order} {seasonal}: AIC={fit.aic:>9,.1f}  valid MAE={rows[-1]['valid_MAE']:>8,.0f}")

    grid = pd.DataFrame(rows).sort_values("valid_MAE").reset_index(drop=True)
    print("\n" + "=" * 70)
    print("КАНДИДАТЫ (отсортированы по ошибке на валидации)")
    print("=" * 70)
    print(grid.to_string(index=False, float_format=lambda v: f"{v:>11,.1f}"))

    # AIC и ошибка на валидации могут расходиться: AIC оценивает качество
    # подгонки внутри train, а valid_MAE - реальный прогноз вперёд
    print(f"\nЛучший по AIC:       {grid.loc[grid.AIC.idxmin(), 'order']} "
          f"{grid.loc[grid.AIC.idxmin(), 'seasonal']}")
    print(f"Лучший по valid MAE: {grid.loc[0, 'order']} {grid.loc[0, 'seasonal']}")

    best_order = eval(grid.loc[0, "order"])
    best_seasonal = eval(grid.loc[0, "seasonal"])

    # ------------------------------------------------------------ финал на test
    # переобучаем на train+valid: модель стартует от самых свежих данных
    print(f"\nПереобучение {best_order}{best_seasonal} на train+valid -> прогноз test...")
    h = len(test)
    _, pred, lo, hi = forecast(trainval, best_order, best_seasonal, h, test.index)

    # для сравнения: та же модель, обученная только на train
    _, pred_tr_full, _, _ = forecast(
        train, best_order, best_seasonal, len(valid) + h,
        pd.date_range(valid.index.min(), periods=len(valid) + h, freq="h")
    )
    pred_train_only = pd.Series(pred_tr_full.to_numpy()[-h:], index=test.index)

    # baseline: повторяем последние сутки train+valid
    naive = pd.Series(
        np.resize(trainval.iloc[-SEASON:].to_numpy(), h), index=test.index
    )

    results = {
        "SARIMA (train+valid)": metrics(test[ok_t], pred[ok_t]),
        "SARIMA (только train)": metrics(test[ok_t], pred_train_only[ok_t]),
        "Seasonal naive": metrics(test[ok_t], naive[ok_t]),
    }
    table = pd.DataFrame(results).T.sort_values("MAE")

    n_bad = int((~ok_t).sum())
    print("\n" + "=" * 70)
    print(f"ФИНАЛЬНАЯ ОЦЕНКА НА TEST ({int(ok_t.sum())} часов"
          + (f", исключено {n_bad} испорченных)" if n_bad else ")"))
    print("=" * 70)
    print(table.to_string(float_format=lambda v: f"{v:>12,.3f}"))

    cov = float(
        ((test.to_numpy()[ok_t] >= lo[ok_t]) & (test.to_numpy()[ok_t] <= hi[ok_t])).mean() * 100
    )
    print(f"\nПокрытие 95% интервала: {cov:.1f}% (номинально 95%)")
    print(f"Средний уровень test:   {test[ok_t].mean():,.0f}")
    print(f"Ошибка в процентах:     {table.iloc[0]['MAE'] / test[ok_t].mean() * 100:.1f}%")

    # ------------------------------------------------------------ сохранение
    out = pd.DataFrame(
        {
            "y_true": test,
            "SARIMA_trainval": pred,
            "SARIMA_train_only": pred_train_only,
            "Seasonal_naive": naive,
            "lo95": lo,
            "hi95": hi,
            "used_in_metrics": ok_t,
        }
    )
    out.index.name = "Time"
    out.to_csv("tvt_forecast_vs_actual.csv")
    grid.to_csv("tvt_grid.csv", index=False)
    table.to_csv("tvt_metrics.csv")

    # ------------------------------------------------------------ графики
    fig, ax = plt.subplots(2, 1, figsize=(15, 9))

    ax[0].plot(train.index[-7 * 24:], train.iloc[-7 * 24:], color="#4C6EF5", lw=1, label="Train")
    ax[0].plot(valid.index, valid, color="#2F9E44", lw=1, label="Valid (подбор параметров)")
    ax[0].plot(test.index, test, color="#111827", lw=1.6, label="Test: реальный y")
    ax[0].plot(test.index, pred, color="#E8590C", lw=1.4, ls="--", label="Прогноз (train+valid)")
    ax[0].fill_between(test.index, lo, hi, color="#E8590C", alpha=0.15, label="95% интервал")
    ax[0].axvline(VALID_START, color="#868E96", ls=":", lw=1.5)
    ax[0].axvline(TEST_START, color="#868E96", ls=":", lw=1.5)
    ax[0].set_title(f"train / valid / test (test MAE={table.iloc[0]['MAE']:,.0f})")
    ax[0].set_ylabel("Users")
    ax[0].legend(loc="upper left", fontsize=9)
    ax[0].grid(alpha=0.25)

    daily = pd.DataFrame(
        {"реальность": test, "train+valid": pred, "только train": pred_train_only}
    ).resample("D").mean()
    daily.plot(ax=ax[1], marker="o", ms=3,
               color=["#111827", "#E8590C", "#4C6EF5"], lw=1.3)
    ax[1].set_title("Средние по суткам: куда уходит прогноз")
    ax[1].set_ylabel("Users")
    ax[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig("tvt_forecast_vs_actual.png", dpi=130)
    print("\nСохранено: tvt_forecast_vs_actual.csv, tvt_grid.csv, "
          "tvt_metrics.csv, tvt_forecast_vs_actual.png")


if __name__ == "__main__":
    main()
