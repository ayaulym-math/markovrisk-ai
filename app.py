import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from core import analyze, read_upload, make_close

st.set_page_config(page_title="MarkovRisk AI", page_icon="📈", layout="centered")

TEXTS = {
    "kk": {
        "subtitle": "Марков тізбектері мен жасанды интеллект негізінде қаржылық тәуекелді болжау жүйесі",
        "stock": "Акцияны таңдаңыз",
        "custom": "Немесе тикерді өзіңіз жазыңыз (мысалы, HSBK.IL)",
        "upload": "Немесе CSV / Excel файлын жүктеңіз",
        "date_col": "Күн бағаны", "price_col": "Баға бағаны",
        "start": "Деректің басталу жылы",
        "run": "Талдау жасау",
        "loading": "Деректер жүктеліп, модельдер үйретілуде...",
        "err_ticker": "Мұндай акция табылмады немесе деректер жүктелмеді.",
        "err_short": "Деректер аз (кемінде 400 күн керек). Басталу жылын ертерек таңдаңыз.",
        "err_file": "Файлды оқу мүмкін болмады. Күн және баға бағандарын тексеріңіз.",
        "src": "Дереккөз: жүктелген файл ({f}).",
        "data_h": "Деректер",
        "price_chart": "Бағаның тарихи динамикасы",
        "ret_chart": "Күнделікті кірістілік",
        "info_data": "Деректер: {a} – {b} ({n} күн).",
        "info_split": "Оқыту: {c} дейін ({ntr} күн). Тексеру: {d} – {e} ({nte} күн).",
        "train_note": "Өту матрицасы және барлық модельдер тек оқыту кезеңінен есептелген. Тексеру күндері модельге көрсетілмеген.",
        "matrix_h": "1. Тәуекел күйлерінің өту матрицасы",
        "matrix_note": "Қатар — бүгінгі күй, баған — ертеңгі күй. Мысалы, «Жоғары» қатары мен «Жоғары» бағанының қиылысы: бүгін тәуекел жоғары болса, ертең де жоғары болу ықтималдығы.",
        "low": "Төмен", "mid": "Орташа", "high": "Жоғары",
        "res_h": "2. Модельдерді салыстыру",
        "res_note": "AUC — «ертең жоғары тәуекелді күн бола ма?» деген сұрақты модель қаншалықты жақсы ажыратады. 0,5 — кездейсоқ болжам, 1,0 — мінсіз. F1 — үш күйді тең есептейтін орташа көрсеткіш.",
        "model": "Модель", "acc": "Дәлдік (Accuracy)", "f1": "F1-score", "auc": "AUC",
        "base": "Қарапайым болжам", "markov": "Марков тізбегі", "lr": "Logistic Regression",
        "rf": "Random Forest", "comb": "Біріктірілген модель",
        "chart": "Жоғары тәуекел күнін болжау (AUC)",
        "disc": "Бұл оқу-зерттеу жұмысы, инвестициялық кеңес емес.",
    },
    "ru": {
        "subtitle": "Система прогнозирования финансовых рисков на основе цепей Маркова и искусственного интеллекта",
        "stock": "Выберите акцию",
        "custom": "Или введите тикер сами (например, HSBK.IL)",
        "upload": "Или загрузите файл CSV / Excel",
        "date_col": "Столбец с датой", "price_col": "Столбец с ценой",
        "start": "Год начала данных",
        "run": "Выполнить анализ",
        "loading": "Загрузка данных и обучение моделей...",
        "err_ticker": "Акция не найдена или данные не загрузились.",
        "err_short": "Данных мало (нужно минимум 400 дней). Выберите более ранний год начала.",
        "err_file": "Не удалось прочитать файл. Проверьте столбцы с датой и ценой.",
        "src": "Источник: загруженный файл ({f}).",
        "data_h": "Данные",
        "price_chart": "Историческая динамика цены",
        "ret_chart": "Дневная доходность",
        "info_data": "Данные: {a} – {b} ({n} дн.).",
        "info_split": "Обучение: до {c} ({ntr} дн.). Проверка: {d} – {e} ({nte} дн.).",
        "train_note": "Матрица переходов и все модели рассчитаны только по обучающему периоду. Дни проверки модели не видели.",
        "matrix_h": "1. Матрица переходов уровней риска",
        "matrix_note": "Строка — сегодняшнее состояние, столбец — завтрашнее. Например, пересечение «Высокий» и «Высокий»: вероятность того, что при высоком риске сегодня он останется высоким завтра.",
        "low": "Низкий", "mid": "Средний", "high": "Высокий",
        "res_h": "2. Сравнение моделей",
        "res_note": "AUC показывает, насколько хорошо модель отличает дни с высоким риском. 0,5 — случайное угадывание, 1,0 — идеал. F1 — средний показатель, учитывающий три состояния поровну.",
        "model": "Модель", "acc": "Точность (Accuracy)", "f1": "F1-score", "auc": "AUC",
        "base": "Простой прогноз", "markov": "Цепь Маркова", "lr": "Logistic Regression",
        "rf": "Random Forest", "comb": "Комбинированная модель",
        "chart": "Прогноз дня с высоким риском (AUC)",
        "disc": "Это учебно-исследовательская работа, а не инвестиционный совет.",
    },
    "tr": {
        "subtitle": "Markov zincirleri ve yapay zekâ temelli finansal risk tahmin sistemi",
        "stock": "Hisse senedini seçin",
        "custom": "Veya sembolü kendiniz yazın (örn. HSBK.IL)",
        "upload": "Veya CSV / Excel dosyası yükleyin",
        "date_col": "Tarih sütunu", "price_col": "Fiyat sütunu",
        "start": "Verinin başlangıç yılı",
        "run": "Analiz yap",
        "loading": "Veriler yükleniyor ve modeller eğitiliyor...",
        "err_ticker": "Hisse bulunamadı veya veri yüklenemedi.",
        "err_short": "Veri yetersiz (en az 400 gün gerekli). Daha erken bir başlangıç yılı seçin.",
        "err_file": "Dosya okunamadı. Tarih ve fiyat sütunlarını kontrol edin.",
        "src": "Kaynak: yüklenen dosya ({f}).",
        "data_h": "Veriler",
        "price_chart": "Fiyatın tarihsel seyri",
        "ret_chart": "Günlük getiri",
        "info_data": "Veri: {a} – {b} ({n} gün).",
        "info_split": "Eğitim: {c} tarihine kadar ({ntr} gün). Test: {d} – {e} ({nte} gün).",
        "train_note": "Geçiş matrisi ve tüm modeller yalnızca eğitim dönemi verisiyle hesaplanmıştır. Test günleri modele gösterilmemiştir.",
        "matrix_h": "1. Risk durumlarının geçiş matrisi",
        "matrix_note": "Satır bugünkü durum, sütun yarınki durumdur. Örneğin «Yüksek» satırı ile «Yüksek» sütununun kesişimi: bugün risk yüksekse yarın da yüksek olma olasılığı.",
        "low": "Düşük", "mid": "Orta", "high": "Yüksek",
        "res_h": "2. Modellerin karşılaştırılması",
        "res_note": "AUC, modelin yüksek riskli günleri ne kadar iyi ayırt ettiğini gösterir. 0,5 rastgele tahmin, 1,0 mükemmeldir. F1, üç durumu eşit ağırlıkla değerlendiren ortalama bir ölçüdür.",
        "model": "Model", "acc": "Doğruluk (Accuracy)", "f1": "F1-score", "auc": "AUC",
        "base": "Basit tahmin", "markov": "Markov zinciri", "lr": "Logistic Regression",
        "rf": "Random Forest", "comb": "Birleşik model",
        "chart": "Yüksek riskli günün tahmini (AUC)",
        "disc": "Bu bir eğitim-araştırma çalışmasıdır, yatırım tavsiyesi değildir.",
    },
    "en": {
        "subtitle": "Financial risk forecasting system based on Markov chains and artificial intelligence",
        "stock": "Choose a stock",
        "custom": "Or type a ticker yourself (e.g. HSBK.IL)",
        "upload": "Or upload a CSV / Excel file",
        "date_col": "Date column", "price_col": "Price column",
        "start": "Data start year",
        "run": "Run analysis",
        "loading": "Downloading data and training models...",
        "err_ticker": "Ticker not found or data could not be loaded.",
        "err_short": "Not enough data (at least 400 days needed). Choose an earlier start year.",
        "err_file": "Could not read the file. Check the date and price columns.",
        "src": "Source: uploaded file ({f}).",
        "data_h": "Data",
        "price_chart": "Historical price dynamics",
        "ret_chart": "Daily return",
        "info_data": "Data: {a} – {b} ({n} days).",
        "info_split": "Training: up to {c} ({ntr} days). Testing: {d} – {e} ({nte} days).",
        "train_note": "The transition matrix and all models are computed from the training period only. The test days were never shown to the models.",
        "matrix_h": "1. Transition matrix of risk states",
        "matrix_note": "Row = today's state, column = tomorrow's state. For example, the cell «High» row / «High» column is the probability that high risk today is followed by high risk tomorrow.",
        "low": "Low", "mid": "Medium", "high": "High",
        "res_h": "2. Model comparison",
        "res_note": "AUC shows how well a model separates high-risk days. 0.5 is a random guess, 1.0 is perfect. F1 is an average that weighs the three states equally.",
        "model": "Model", "acc": "Accuracy", "f1": "F1-score", "auc": "AUC",
        "base": "Naive baseline", "markov": "Markov chain", "lr": "Logistic Regression",
        "rf": "Random Forest", "comb": "Combined model",
        "chart": "Predicting a high-risk day (AUC)",
        "disc": "This is an educational research project, not investment advice.",
    },
}

LANGS = {"Қазақша": "kk", "Русский": "ru", "Türkçe": "tr", "English": "en"}

STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "JPMorgan (JPM)": "JPM",
    "S&P 500 (^GSPC)": "^GSPC",
}

lang_name = st.sidebar.selectbox("Til / Тіл / Язык", list(LANGS.keys()))
t = TEXTS[LANGS[lang_name]]


@st.cache_data(show_spinner=False)
def load_close(ticker, start_year):
    df = yf.download(ticker, start=f"{start_year}-01-01", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    return df["Close"].squeeze().dropna()


st.title("MarkovRisk AI")
st.caption(t["subtitle"])

choice = st.sidebar.selectbox(t["stock"], list(STOCKS.keys()))
custom = st.sidebar.text_input(t["custom"]).strip().upper()
start_year = st.sidebar.slider(t["start"], 2012, 2022, 2018)
up = st.sidebar.file_uploader(t["upload"], type=["csv", "xlsx", "xls"])

up_df, date_col, price_col = None, None, None
if up is not None:
    try:
        up_df = read_upload(up, up.name)
        cols = list(up_df.columns)
        d_def = next((i for i, c in enumerate(cols) if str(c).lower() in ("date", "дата", "tarih")), 0)
        p_def = next((i for i, c in enumerate(cols) if str(c).lower() in ("close", "adj close", "price", "цена", "fiyat")),
                     len(cols) - 1)
        date_col = st.sidebar.selectbox(t["date_col"], cols, index=d_def)
        price_col = st.sidebar.selectbox(t["price_col"], cols, index=p_def)
    except Exception:
        st.sidebar.error(t["err_file"])
        up_df = None

ticker = custom if custom else STOCKS[choice]

if st.sidebar.button(t["run"], type="primary"):
    with st.spinner(t["loading"]):
        close = None
        if up is not None:
            if up_df is None:
                st.error(t["err_file"])
                st.stop()
            try:
                close = make_close(up_df, date_col, price_col)
            except Exception:
                close = None
            if close is None or len(close) == 0:
                st.error(t["err_file"])
                st.stop()
        else:
            try:
                close = load_close(ticker, start_year)
            except Exception:
                close = None
            if close is None or len(close) == 0:
                st.error(t["err_ticker"])
                st.stop()
        if len(close) < 400:
            st.error(t["err_short"])
            st.stop()
        M, res, info = analyze(close)

    if up is not None:
        st.caption(t["src"].format(f=up.name))

    st.subheader(t["data_h"])
    st.success(t["info_data"].format(a=info["start"], b=info["end"], n=info["n"]) + "\n\n" +
               t["info_split"].format(c=info["train_end"], ntr=info["n_train"], d=info["test_start"],
                                      e=info["test_end"], nte=info["n_test"]))
    st.caption(t["price_chart"])
    st.line_chart(close)
    st.caption(t["ret_chart"])
    st.line_chart(close.pct_change().dropna())

    st.subheader(t["matrix_h"])
    labels = [t["low"], t["mid"], t["high"]]
    M.index, M.columns = labels, labels
    st.dataframe(M.round(3), use_container_width=True)
    st.caption(t["matrix_note"])
    st.caption(t["train_note"])

    st.subheader(t["res_h"])
    table = res.copy()
    table[t["model"]] = table["key"].map(lambda k: t[k])
    table[t["acc"]] = table["acc"].map(lambda v: f"{v:.3f}")
    table[t["f1"]] = table["f1"].map(lambda v: f"{v:.3f}")
    table[t["auc"]] = table["auc"].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    st.dataframe(table[[t["model"], t["acc"], t["f1"], t["auc"]]].set_index(t["model"]),
                 use_container_width=True)

    plot = res.dropna()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar([t[k] for k in plot["key"]], plot["auc"], color=["#4c72b0", "#8172b3", "#dd8452", "#55a868"])
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set_ylim(0.4, max(0.7, float(plot["auc"].max()) + 0.05))
    ax.set_ylabel("AUC")
    ax.set_title(t["chart"])
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(t["res_note"])
    st.caption(t["disc"])
else:
    st.info("👈 " + t["run"])
