import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

from core import analyze

st.set_page_config(page_title="MarkovRisk AI", page_icon="📈", layout="centered")

TEXTS = {
    "kk": {
        "title": "MarkovRisk AI",
        "subtitle": "Марков тізбектері мен жасанды интеллект негізінде қаржылық тәуекелді болжау жүйесі",
        "stock": "Акцияны таңдаңыз",
        "custom": "Немесе тикерді өзіңіз жазыңыз (мысалы, HSBK.IL)",
        "start": "Деректің басталу жылы",
        "run": "Талдау жасау",
        "loading": "Деректер жүктеліп, модельдер үйретілуде...",
        "err_ticker": "Мұндай акция табылмады немесе деректер жүктелмеді.",
        "err_short": "Деректер аз (кемінде 400 күн керек). Басталу жылын ертерек таңдаңыз.",
        "info": "Талданған күн: {n}. Соңғы {t} күн тексеру үшін қолданылды.",
        "matrix_h": "1. Тәуекел күйлерінің өту матрицасы",
        "matrix_note": "Қатар — бүгінгі күй, баған — ертеңгі күй. Мысалы, «Жоғары» қатары мен «Жоғары» бағанының қиылысы: бүгін тәуекел жоғары болса, ертең де жоғары болу ықтималдығы.",
        "low": "Төмен", "mid": "Орташа", "high": "Жоғары",
        "res_h": "2. Модельдерді салыстыру",
        "res_note": "AUC — «ертең жоғары тәуекелді күн бола ма?» деген сұрақты модель қаншалықты жақсы ажыратады. 0,5 — кездейсоқ болжам, 1,0 — мінсіз.",
        "model": "Модель", "acc": "Дәлдік (Accuracy)", "auc": "AUC",
        "base": "Қарапайым болжам", "markov": "Марков тізбегі",
        "rf": "Random Forest", "comb": "Біріктірілген модель",
        "chart": "Жоғары тәуекел күнін болжау (AUC)",
        "disc": "Бұл оқу-зерттеу жұмысы, инвестициялық кеңес емес.",
    },
    "ru": {
        "title": "MarkovRisk AI",
        "subtitle": "Система прогнозирования финансовых рисков на основе цепей Маркова и искусственного интеллекта",
        "stock": "Выберите акцию",
        "custom": "Или введите тикер сами (например, HSBK.IL)",
        "start": "Год начала данных",
        "run": "Выполнить анализ",
        "loading": "Загрузка данных и обучение моделей...",
        "err_ticker": "Акция не найдена или данные не загрузились.",
        "err_short": "Данных мало (нужно минимум 400 дней). Выберите более ранний год начала.",
        "info": "Проанализировано дней: {n}. Последние {t} дней использованы для проверки.",
        "matrix_h": "1. Матрица переходов уровней риска",
        "matrix_note": "Строка — сегодняшнее состояние, столбец — завтрашнее. Например, пересечение «Высокий» и «Высокий»: вероятность того, что при высоком риске сегодня он останется высоким завтра.",
        "low": "Низкий", "mid": "Средний", "high": "Высокий",
        "res_h": "2. Сравнение моделей",
        "res_note": "AUC показывает, насколько хорошо модель отличает дни с высоким риском. 0,5 — случайное угадывание, 1,0 — идеал.",
        "model": "Модель", "acc": "Точность (Accuracy)", "auc": "AUC",
        "base": "Простой прогноз", "markov": "Цепь Маркова",
        "rf": "Random Forest", "comb": "Комбинированная модель",
        "chart": "Прогноз дня с высоким риском (AUC)",
        "disc": "Это учебно-исследовательская работа, а не инвестиционный совет.",
    },
    "tr": {
        "title": "MarkovRisk AI",
        "subtitle": "Markov zincirleri ve yapay zekâ temelli finansal risk tahmin sistemi",
        "stock": "Hisse senedini seçin",
        "custom": "Veya sembolü kendiniz yazın (örn. HSBK.IL)",
        "start": "Verinin başlangıç yılı",
        "run": "Analiz yap",
        "loading": "Veriler yükleniyor ve modeller eğitiliyor...",
        "err_ticker": "Hisse bulunamadı veya veri yüklenemedi.",
        "err_short": "Veri yetersiz (en az 400 gün gerekli). Daha erken bir başlangıç yılı seçin.",
        "info": "Analiz edilen gün: {n}. Son {t} gün test için kullanıldı.",
        "matrix_h": "1. Risk durumlarının geçiş matrisi",
        "matrix_note": "Satır bugünkü durum, sütun yarınki durumdur. Örneğin «Yüksek» satırı ile «Yüksek» sütununun kesişimi: bugün risk yüksekse yarın da yüksek olma olasılığı.",
        "low": "Düşük", "mid": "Orta", "high": "Yüksek",
        "res_h": "2. Modellerin karşılaştırılması",
        "res_note": "AUC, modelin yüksek riskli günleri ne kadar iyi ayırt ettiğini gösterir. 0,5 rastgele tahmin, 1,0 mükemmeldir.",
        "model": "Model", "acc": "Doğruluk (Accuracy)", "auc": "AUC",
        "base": "Basit tahmin", "markov": "Markov zinciri",
        "rf": "Random Forest", "comb": "Birleşik model",
        "chart": "Yüksek riskli günün tahmini (AUC)",
        "disc": "Bu bir eğitim-araştırma çalışmasıdır, yatırım tavsiyesi değildir.",
    },
    "en": {
        "title": "MarkovRisk AI",
        "subtitle": "Financial risk forecasting system based on Markov chains and artificial intelligence",
        "stock": "Choose a stock",
        "custom": "Or type a ticker yourself (e.g. HSBK.IL)",
        "start": "Data start year",
        "run": "Run analysis",
        "loading": "Downloading data and training models...",
        "err_ticker": "Ticker not found or data could not be loaded.",
        "err_short": "Not enough data (at least 400 days needed). Choose an earlier start year.",
        "info": "Days analysed: {n}. The last {t} days were used for testing.",
        "matrix_h": "1. Transition matrix of risk states",
        "matrix_note": "Row = today's state, column = tomorrow's state. For example, the cell «High» row / «High» column is the probability that high risk today is followed by high risk tomorrow.",
        "low": "Low", "mid": "Medium", "high": "High",
        "res_h": "2. Model comparison",
        "res_note": "AUC shows how well a model separates high-risk days. 0.5 is a random guess, 1.0 is perfect.",
        "model": "Model", "acc": "Accuracy", "auc": "AUC",
        "base": "Naive baseline", "markov": "Markov chain",
        "rf": "Random Forest", "comb": "Combined model",
        "chart": "Predicting a high-risk day (AUC)",
        "disc": "This is an educational research project, not investment advice.",
    },
}

LANGS = {"🇰🇿 Қазақша": "kk", "🇷🇺 Русский": "ru", "🇹🇷 Türkçe": "tr", "🇬🇧 English": "en"}

STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "JPMorgan (JPM)": "JPM",
    "S&P 500 (^GSPC)": "^GSPC",
}

lang_name = st.sidebar.selectbox("🌐", list(LANGS.keys()))
t = TEXTS[LANGS[lang_name]]


@st.cache_data(show_spinner=False)
def load_close(ticker, start_year):
    df = yf.download(ticker, start=f"{start_year}-01-01", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    return df["Close"].squeeze().dropna()


st.title(t["title"])
st.caption(t["subtitle"])

choice = st.sidebar.selectbox(t["stock"], list(STOCKS.keys()))
custom = st.sidebar.text_input(t["custom"]).strip().upper()
start_year = st.sidebar.slider(t["start"], 2012, 2022, 2018)
ticker = custom if custom else STOCKS[choice]

if st.sidebar.button(t["run"], type="primary"):
    with st.spinner(t["loading"]):
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
        M, res, n_test = analyze(close)

    st.success(t["info"].format(n=len(close), t=n_test))

    st.subheader(t["matrix_h"])
    labels = [t["low"], t["mid"], t["high"]]
    M.index, M.columns = labels, labels
    st.dataframe(M.round(3), use_container_width=True)
    st.caption(t["matrix_note"])

    st.subheader(t["res_h"])
    table = res.copy()
    table[t["model"]] = table["key"].map(lambda k: t[k])
    table[t["acc"]] = table["acc"].round(3)
    table[t["auc"]] = table["auc"].round(3)
    st.dataframe(table[[t["model"], t["acc"], t["auc"]]].set_index(t["model"]),
                 use_container_width=True)

    plot = res.dropna()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar([t[k] for k in plot["key"]], plot["auc"], color=["#4c72b0", "#dd8452", "#55a868"])
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
