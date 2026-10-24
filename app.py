import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Анализ заказов", layout="wide")

# CSV-файлы ищем рядом с app.py, независимо от того, откуда запущено приложение
DATA_DIR = Path(__file__).parent

DAYS_RU = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]


# ---------------------------------------------------------------------------
# 1–2. Загрузка данных
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    users = pd.read_csv(DATA_DIR / "users.csv")
    items = pd.read_csv(DATA_DIR / "items.csv")
    return orders, users, items


# ---------------------------------------------------------------------------
# 3–4. Объединение и очистка данных
# ---------------------------------------------------------------------------
@st.cache_data
def prepare_data():
    orders, users, items = load_data()

    # orders + users по user_id, затем + items по item_id
    df = orders.merge(users, on="user_id", how="left")
    df = df.merge(items, on="item_id", how="left")

    # Типы данных: даты -> datetime, числа -> int/float
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price_per_unit"] = pd.to_numeric(df["price_per_unit"], errors="coerce")
    df["base_price"] = pd.to_numeric(df["base_price"], errors="coerce")

    # Пропуски: категориальные -> "Unknown", строки без даты/количества/цены удаляем
    missing_before = df.isnull().sum()
    for col in ["item_name", "category", "supplier", "city", "user_segment", "user_name"]:
        df[col] = df[col].fillna("Unknown")
    df = df.dropna(subset=["order_date", "quantity", "price_per_unit"])

    # Выручка по строке заказа и день недели
    df["revenue"] = df["quantity"] * df["price_per_unit"]
    df["weekday_num"] = df["order_date"].dt.dayofweek
    df["weekday"] = df["weekday_num"].map(dict(enumerate(DAYS_RU)))

    return df, missing_before


st.title("Дашборд анализа заказов интернет-магазина")

df, missing_before = prepare_data()

# ---------------------------------------------------------------------------
# Фильтры (боковая панель)
# ---------------------------------------------------------------------------
st.sidebar.header("Фильтры")

min_date = df["order_date"].dt.date.min()
max_date = df["order_date"].dt.date.max()
date_range = st.sidebar.date_input(
    "Период заказов", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

all_categories = sorted(df["category"].unique())
selected_categories = st.sidebar.multiselect(
    "Категории товаров", options=all_categories, default=all_categories
)

# date_input возвращает кортеж из 2 дат только когда выбран весь период
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start, end = date_range
else:
    start = end = date_range[0] if isinstance(date_range, (list, tuple)) else date_range

mask = (
    (df["order_date"].dt.date >= start)
    & (df["order_date"].dt.date <= end)
    & (df["category"].isin(selected_categories))
)
fdf = df[mask]

if fdf.empty:
    st.warning("По выбранным фильтрам данных нет. Измените период или категории.")
    st.stop()

# ---------------------------------------------------------------------------
# 5. Вкладки дашборда
# ---------------------------------------------------------------------------
tab_raw, tab_kpi, tab_viz, tab_conclusions = st.tabs(
    ["Сырые данные", "Ключевые показатели", "Визуализация", "Аналитические выводы"]
)

# 5.1 Сырые данные ----------------------------------------------------------
with tab_raw:
    st.subheader("Итоговая таблица (заказы + пользователи + товары)")
    st.caption(f"Строк после фильтрации: {len(fdf)} из {len(df)}")
    st.dataframe(
        fdf.drop(columns=["weekday_num"]).sort_values("order_date"),
        use_container_width=True,
    )

    with st.expander("Проверка качества данных"):
        st.write("Пропуски в объединённой таблице до очистки:")
        st.dataframe(missing_before.rename("Количество пропусков").to_frame())
        if missing_before.sum() == 0:
            st.write("Пропусков не обнаружено, заполнение не потребовалось.")
        st.write("Типы данных после преобразования:")
        st.dataframe(df.dtypes.astype(str).rename("Тип").to_frame())

# 5.2 Ключевые показатели ---------------------------------------------------
total_orders = fdf["order_id"].nunique()
total_revenue = fdf["revenue"].sum()
unique_users = fdf["user_id"].nunique()
avg_check = total_revenue / total_orders

with tab_kpi:
    st.subheader("Ключевые показатели")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Общее количество заказов", f"{total_orders}")
    c2.metric("Общая выручка", f"{total_revenue:,.2f} ₽".replace(",", " "))
    c3.metric("Уникальных пользователей", f"{unique_users}")
    c4.metric("Средний чек", f"{avg_check:,.2f} ₽".replace(",", " "))

# Данные для графиков -------------------------------------------------------
top_items = (
    fdf.groupby("item_name")["revenue"].sum().sort_values(ascending=False).head(10)
)
cat_revenue = fdf.groupby("category")["revenue"].sum().sort_values(ascending=False)
weekday_orders = (
    fdf.groupby("weekday_num")["order_id"].nunique().reindex(range(7), fill_value=0)
)
weekday_orders.index = DAYS_RU

# 5.3 Визуализация ----------------------------------------------------------
with tab_viz:
    st.subheader("Топ-10 товаров по выручке")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    top_sorted = top_items.sort_values()  # самый большой товар окажется сверху
    ax1.barh(top_sorted.index, top_sorted.values, color="#4C78A8")
    ax1.set_xlabel("Выручка, ₽")
    ax1.set_ylabel("Товар")
    for i, v in enumerate(top_sorted.values):
        ax1.text(v, i, f" {v:,.0f}".replace(",", " "), va="center", fontsize=9)
    ax1.set_xlim(0, top_sorted.max() * 1.15)
    plt.tight_layout()
    st.pyplot(fig1)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Выручка по категориям")
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        # Мелкие категории (< 3%) объединяем в «Прочее», чтобы подписи не слипались
        share = cat_revenue / cat_revenue.sum()
        big = cat_revenue[share >= 0.03]
        small_sum = cat_revenue[share < 0.03].sum()
        pie_data = big.copy()
        if small_sum > 0:
            pie_data["Прочее"] = small_sum
        ax2.pie(
            pie_data.values,
            labels=pie_data.index,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.8,
        )
        ax2.axis("equal")
        plt.tight_layout()
        st.pyplot(fig2)

    with col_b:
        st.subheader("Количество заказов по дням недели")
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        ax3.bar(weekday_orders.index, weekday_orders.values, color="#F58518")
        ax3.set_xlabel("День недели")
        ax3.set_ylabel("Количество заказов")
        ax3.tick_params(axis="x", rotation=45)
        for i, v in enumerate(weekday_orders.values):
            ax3.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig3)

# 5.4 Аналитические выводы --------------------------------------------------
# Выводы формируются из данных, поэтому меняются вместе с фильтрами
top_cat = cat_revenue.index[0]
top_cat_share = cat_revenue.iloc[0] / cat_revenue.sum() * 100
top_item = top_items.index[0]
top_item_rev = top_items.iloc[0]
peak_day = weekday_orders.idxmax()
peak_val = weekday_orders.max()
weekend_share = (
    weekday_orders[["Суббота", "Воскресенье"]].sum() / weekday_orders.sum() * 100
)
top_city = fdf.groupby("city")["revenue"].sum().sort_values(ascending=False)
seg_rev = fdf.groupby("user_segment")["revenue"].sum().sort_values(ascending=False)
top3_share = top_items.head(3).sum() / total_revenue * 100

# Сколько раз каждый день недели встречается в выбранном периоде (для оговорки)
period_days = pd.date_range(start, end)
days_count = pd.Series(period_days.dayofweek).value_counts()

with tab_conclusions:
    st.subheader("Аналитические выводы")
    st.markdown(
        f"""
- **Основная выручка приходится на категорию "{top_cat}"**: {cat_revenue.iloc[0]:,.0f} ₽,
  то есть {top_cat_share:.1f}% от общей выручки за выбранный период.
- **Лидер по выручке среди товаров: "{top_item}"**: {top_item_rev:,.0f} ₽.
  Три самых прибыльных товара дают {top3_share:.1f}% всей выручки.
- **Пик заказов приходится на день "{peak_day}"**: {peak_val} заказов.
  На выходные (суббота и воскресенье) приходится {weekend_share:.1f}% заказов.
- **Средний чек** составляет {avg_check:,.0f} ₽ при {total_orders} заказах
  от {unique_users} уникальных пользователей.
- **География:** больше всего выручки приносит город {top_city.index[0]}
  ({top_city.iloc[0]:,.0f} ₽, {top_city.iloc[0] / total_revenue * 100:.1f}%).
- **Сегменты клиентов:** лидирует сегмент "{seg_rev.index[0]}"
  ({seg_rev.iloc[0] / total_revenue * 100:.1f}% выручки).
""".replace(",", " ")
    )

    st.markdown("#### Рекомендации")
    st.markdown(
        f"""
- Развивать и расширять ассортимент в категории "{top_cat}", так как она формирует основную часть выручки.
- Планировать акции и рекламу на пиковые дни ("{peak_day}"), а в слабые дни стимулировать спрос скидками.
- Следить за зависимостью выручки от узкого круга товаров: топ-3 дают {top3_share:.0f}% продаж.
""".replace(",", " ")
    )

    st.info(
        "Ограничение анализа: данные охватывают короткий период "
        f"({start} — {end}), поэтому некоторые дни недели встречаются в нём чаще других "
        "(по количеству повторов: "
        + ", ".join(f"{DAYS_RU[d]} — {n}" for d, n in sorted(days_count.items()))
        + "). Выводы о сезонности по дням недели нужно проверять на более длинной выборке."
    )
