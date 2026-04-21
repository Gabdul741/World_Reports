import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import cm
import io

#font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
import os
base_dir = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(base_dir, "fonts", "DejaVuSans.ttf")
font_bold_path = os.path.join(base_dir, "fonts", "DejaVuSans-Bold.ttf")
pdfmetrics.registerFont(TTFont("DejaVu", font_path))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", font_bold_path))
pdfmetrics.registerFont(TTFont("DejaVu", font_path))
#pdfmetrics.registerFont(TTFont("DejaVu-Bold",
#    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))


lang = st.sidebar.selectbox("🌐 Language / Язык:", ["Русский", "English"])
if lang == "English":
    t = {
        "title": "🌍 World Data Reports",
        "indicator": "Select indicator:",
        "scale": "Scale:",
        "countries": "Select countries:",
        "year_from": "From year:",
        "year_to": "To year:",
        "data": "📋 Data:",
        "stats": "📈 Statistics:",
        "chart": "📊 Chart:",
        "pdf_btn": "📄 Create PDF Report",
        "download": "📥 Download PDF",
        "sources": "📚 Data Sources:",
        "exit_btn": "🚪 Exit",
        "exit_msg": "Thank you! Please close the browser tab.",
        "scale_options": ["Original", "Thousands", "Millions", "Billions"],
        "indicators": ["GDP", "Population", "Inflation", "Unemployment",
            "Life Expectancy", "GDP per capita", "CO2 emissions",
            "Military expenditure", "Education expenditure", "Literacy rate"],
    }
else:
    t = {
        "title": "🌍 Отчёты по мировым данным",
        "indicator": "Выберите показатель:",
        "scale": "Масштаб значений:",
        "countries": "Выберите страны:",
        "year_from": "С года:",
        "year_to": "По год:",
        "data": "📋 Данные:",
        "stats": "📈 Статистика:",
        "chart": "📊 График:",
        "pdf_btn": "📄 Создать PDF отчёт",
        "download": "📥 Скачать PDF",
        "sources": "📚 Источники данных:",
        "exit_btn": "🚪 Выход",
        "exit_msg": "Спасибо! Закройте вкладку браузера.",
        "scale_options": ["Исходные", "Тысячи", "Миллионы", "Миллиарды"],
        "indicators": ["ВВП", "Население", "Инфляция", "Безработица",
            "Продолжительность жизни", "ВВП на душу населения",
            "CO2 выбросы", "Военные расходы", "Расходы на образование",
            "Грамотность населения"],
    }

st.title(t["title"])
with st.expander("❓ Help — Инструкция"):
    st.markdown("""
    **Как пользоваться приложением:**
    
    1. **Выберите показатель** — ВВП, Население, Инфляция и другие
    2. **Выберите масштаб** — Исходные, Тысячи, Миллионы, Миллиарды
    3. **Выберите страны** — можно выбрать несколько стран
    4. **Задайте период** — выберите начальный и конечный год
    5. **Просмотрите данные** — таблица и график обновятся автоматически
    6. **Создайте PDF** — нажмите кнопку для скачивания отчёта
    """)

with st.expander(" About"):
    st.markdown("""
    **🌍 Отчёты по мировым данным**
    
    Приложение для анализа и визуализации мировых экономических показателей.
    
    **Версия:** 1.0  
    **Данные:** World Bank, WHO, UNESCO, SIPRI, Our World in Data  
    **Технологии:** Python, Streamlit, Pandas, Matplotlib, ReportLab  
    **Совместное производство:** Gabdul741 и Claude (Sonnet 4.6) — Anthropic
    **Ссылка:** world-reports-gabdul741.streamlit.app
    """)
#indicator = st.selectbox("Выберите показатель:", ["ВВП", "Население", "Инфляция", "Безработица"])
#indicator = st.selectbox("Выберите показатель:", ["ВВП", "Население", "Инфляция", "Безработица", "Продолжительность жизни"])
indicator = st.selectbox("Выберите показатель:", [
    "ВВП", "Население", "Инфляция", "Безработица", 
    "Продолжительность жизни", "ВВП на душу населения",
    "CO2 выбросы", "Военные расходы", "Расходы на образование",
    "Грамотность населения"
])
scale = st.selectbox("Масштаб значений:", [ "Исходные", "Тысячи", "Миллионы", "Миллиарды"])

if indicator == "ВВП":
    df = pd.read_csv(os.path.join(base_dir, "gdp.csv"))
    ylabel = "ВВП"
    title = "ВВП по странам"
    units = "долл. США"
elif indicator == "Население":
    df = pd.read_csv(os.path.join(base_dir, "population.csv"))
    ylabel = "Население"
    title = "Население по странам"
    units = "чел."
elif indicator == "Инфляция":
    df = pd.read_csv(os.path.join(base_dir, "inflation.csv"))
    df = df.rename(columns={"Country": "Country Name", "Inflation": "Value"})
    ylabel = "Инфляция"
    title = "Инфляция по странам"
    units = "%"
elif indicator == "Безработица":
    df = pd.read_csv(os.path.join(base_dir, "unemployment.csv"))
    df = df.rename(columns={"Unemployment Rate": "Value"})
    ylabel = "Безработица"
    title = "Безработица по странам"
    units = "%"
#else:
#    df = pd.read_csv(os.path.join(base_dir, "life_expectancy.csv"))
#    df = df.rename(columns={"Country": "Country Name", "Life expectancy": "Value"})
#    ylabel = "Продолжительность жизни"
#    title = "Продолжительность жизни по странам"
#    units = "лет"
elif indicator == "Продолжительность жизни":
    df = pd.read_csv(os.path.join(base_dir, "life_expectancy.csv"))
    df = df.rename(columns={"Country": "Country Name", "Life expectancy": "Value"})
    ylabel = "Продолжительность жизни"
    title = "Продолжительность жизни по странам"
    units = "лет"
elif indicator == "ВВП на душу населения":
    df = pd.read_csv(os.path.join(base_dir, "gdp_per_capita.csv"))
    df = df.rename(columns={"Entity": "Country Name", "GDP per capita": "Value"})
    ylabel = "ВВП на душу населения"
    title = "ВВП на душу населения"
    units = "долл. США"
elif indicator == "CO2 выбросы":
    df = pd.read_csv(os.path.join(base_dir, "co2.csv"))
    df = df.rename(columns={"Entity": "Country Name", "CO₂ emissions per capita": "Value"})
    ylabel = "CO2 выбросы"
    title = "CO2 выбросы на душу населения"
    units = "т"
elif indicator == "Военные расходы":
    df = pd.read_csv(os.path.join(base_dir, "military.csv"))
    df = df.rename(columns={"Entity": "Country Name", "Military expenditure (% of GDP)": "Value"})
    ylabel = "Военные расходы"
    title = "Военные расходы (% от ВВП)"
    units = "%"
elif indicator == "Расходы на образование":
    df = pd.read_csv(os.path.join(base_dir, "education.csv"))
    df = df.rename(columns={"Entity": "Country Name", "Total across all levels of education": "Value"})
    ylabel = "Расходы на образование"
    title = "Расходы на образование (% от ВВП)"
    units = "%"
else:
    df = pd.read_csv(os.path.join(base_dir, "literacy.csv"))
    df = df.rename(columns={"Entity": "Country Name", "Literacy rate among adults": "Value"})
    ylabel = "Грамотность населения"
    title = "Грамотность населения"
    units = "%"

countries = df["Country Name"].unique().tolist()
#selected = st.multiselect("Выберите страны:", countries,
#    default=["Russian Federation", "United States", "China", "Germany"])
#if indicator == "Продолжительность жизни":
#    default_countries = ["Russia", "United States", "China", "Germany"]
#else:
#    default_countries = ["Russian Federation", "United States", "China", "Germany"]

#default_countries = [c for c in default_countries if c in countries]
#if indicator == "Продолжительность жизни":.
#    default_countries = ["Russia", "United States", "China", "Germany"]

#else:
#    default_countries = ["Russian Federation", "United States", "China", "Germany"]
if indicator in ["Продолжительность жизни", "ВВП на душу населения",
                 "CO2 выбросы", "Военные расходы",
                 "Расходы на образование", "Грамотность населения"]:
    default_countries = ["Russia", "United States", "China", "Germany"]
else:
    default_countries = ["Russian Federation", "United States", "China", "Germany"]

default_countries = [c for c in default_countries if c in countries]

selected = st.multiselect("Выберите страны:", countries,
    default=default_countries)

max_year = int(df["Year"].max())
min_year = int(df["Year"].min())
year_from = st.slider("С года:", min_year, max_year, 2000)
year_to = st.slider("По год:", min_year, max_year, max_year)

filtered = df[
    (df["Country Name"].isin(selected)) &
    (df["Year"] >= year_from) &
    (df["Year"] <= year_to)
].copy()

if scale == "Тысячи":
    filtered["Value"] = filtered["Value"] / 1_000
    scale_label = f" (тыс. {units})"
elif scale == "Миллионы":
    filtered["Value"] = filtered["Value"] / 1_000_000
    scale_label = f" (млн. {units})"
elif scale == "Миллиарды":
    filtered["Value"] = filtered["Value"] / 1_000_000_000
    scale_label = f" (млрд. {units})"
else:
    scale_label = f" ({units})"

ylabel_full = ylabel + scale_label

if len(filtered) > 0:
    st.subheader("📋 Данные:")
    pivot_screen = filtered.pivot(index="Year",
                                   columns="Country Name",
                                   values="Value")
    pivot_screen.columns.name = None
    pivot_screen.index = pivot_screen.index.astype(int)
    pivot_screen = pivot_screen.round(2)
    st.dataframe(pivot_screen)

    fig, ax = plt.subplots(figsize=(10, 5))
    for country in selected:
        data = filtered[filtered["Country Name"] == country]
        ax.plot(data["Year"].astype(int), data["Value"],
                marker="o", label=country)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Год")
    ax.set_ylabel(ylabel_full)
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.subheader("📊 График:")
    st.pyplot(fig)

    if st.button("📄 Создать PDF отчёт"):
        buffer = io.BytesIO()
        page_title = f"{title} — {ylabel_full}"

        def header_footer(canvas_obj, doc_obj):
            canvas_obj.saveState()
            canvas_obj.setFont("DejaVu-Bold", 12)
            canvas_obj.drawCentredString(
                landscape(A4)[0] / 2,
                landscape(A4)[1] - 30,
                page_title
            )
            canvas_obj.setFont("DejaVu", 8)
            canvas_obj.drawRightString(
                landscape(A4)[0] - 30,
                20,
                f"Страница {doc_obj.page}"
            )
            canvas_obj.restoreState()

        doc = SimpleDocTemplate(buffer, pagesize=landscape(A4),
            topMargin=1.5*cm)
        elements = []
        elements.append(Spacer(1, 0.5*cm))

        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format="PNG", bbox_inches="tight")
        img_buffer.seek(0)
        elements.append(Image(img_buffer, width=22*cm, height=10*cm))
        elements.append(Spacer(1, 0.5*cm))

        pivot = filtered.pivot(index="Year",
                               columns="Country Name",
                               values="Value")
        pivot = pivot.reset_index()
        pivot.columns.name = None

        rows = []
        for row in pivot.values.tolist():
            new_row = []
            for i, val in enumerate(row):
                if i == 0:
                    new_row.append(str(int(val)))
                else:
                    if val == val:
                        new_row.append(str(round(float(val), 2)))
                    else:
                        new_row.append("")
            rows.append(new_row)
        data_table = [list(pivot.columns)] + rows

        table = Table(data_table, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.blue),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("FONTNAME", (0,0), (-1,-1), "DejaVu"),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("GRID", (0,0), (-1,-1), 1, colors.black),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("BACKGROUND", (0,1), (-1,-1), colors.lightblue),
        ]))
        elements.append(table)
        doc.build(elements,
            onFirstPage=header_footer,
            onLaterPages=header_footer)
        buffer.seek(0)

        st.download_button(
            label="📥 Скачать PDF",
            data=buffer,
            file_name=f"{indicator}_report.pdf",
            mime="application/pdf"
)

st.markdown("---")
st.subheader("📚 Источники данных:")
sources = {
    "ВВП": ("gdp.csv", "World Bank — ВВП в долл. США", "github.com/datasets/gdp"),
    "Население": ("population.csv", "World Bank — Население стран", "github.com/datasets/population"),
    "Инфляция": ("inflation.csv", "World Bank — Инфляция потребительских цен (%)", "github.com/datasets/inflation"),
    "Безработица": ("unemployment.csv", "World Bank/ILO — Уровень безработицы (%)", "github.com/ShinjiniShome/world_unemployment_dataviz"),
    "Продолжительность жизни": ("life_expectancy.csv", "WHO — Продолжительность жизни при рождении", "github.com/Sid-149/Life-Expectancy-Predictor"),
    "ВВП на душу населения": ("gdp_per_capita.csv", "World Bank — ВВП на душу населения в долл. США", "ourworldindata.org/grapher/gdp-per-capita-worldbank"),
    "CO2 выбросы": ("co2.csv", "Our World in Data — CO2 выбросы на душу населения (т)", "ourworldindata.org/grapher/co-emissions-per-capita"),
    "Военные расходы": ("military.csv", "SIPRI — Военные расходы (% от ВВП)", "ourworldindata.org/grapher/military-expenditure-as-a-share-of-gdp"),
    "Расходы на образование": ("education.csv", "UNESCO — Расходы на образование (% от ВВП)", "ourworldindata.org/grapher/total-government-expenditure-on-education-gdp"),
    "Грамотность населения": ("literacy.csv", "UNESCO — Уровень грамотности взрослых (%)", "ourworldindata.org/grapher/literacy-rate-adults"),
}
for indicator_name, (filename, description, source) in sources.items():
    st.markdown(f"- **{indicator_name}:** {description} | Файл: `{filename}` | Источник: {source}")
st.markdown("---")
if st.button("🚪 Выход"):
    st.success("Спасибо за использование приложения! Закройте вкладку браузера.")
    st.stop()

