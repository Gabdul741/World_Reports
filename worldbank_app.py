# ===== ЗАЩИТА ДАННЫХ =====
DATA_SOURCE = "World Bank Open Data"
REPORT_NAME = "World_Bank_Report"

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== НАСТРОЙКИ PDF =====
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import plotly.io as pio
    import tempfile
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ===== НАСТРОЙКИ =====
st.set_page_config(page_title="World Bank Explorer", layout="wide")
st.title("📊 World Bank Data Explorer")

# ===== ПОКАЗАТЕЛИ (25+ шт) =====
# ===== ПЕРЕВОДЫ =====
TEXTS = {
    "Русский": {
        "title": "📊 World Bank Data Explorer",
        "settings": "⚙️ Настройки",
        "search_country": "🔍 Поиск страны",
        "search_country_placeholder": "Россия, США, Германия...",
        "countries": "🌍 Страны",
        "search_indicator": "🔍 Поиск показателя",
        "search_indicator_placeholder": "ВВП, население...",
        "indicator": "📈 Показатель",
        "scale": "📏 Масштаб",
        "year_from": "Год от",
        "year_to": "Год до",
        "load": "🔄 Загрузить данные",
        "exit": "🚪 EXIT / Перезагрузка",
        "exit_tip": "💡 **Совет:** Если не видите кнопку EXIT, потяните правую границу боковой панели →",
        "help_title": "❓ HELP - Инструкция",
        "about_title": "ℹ️ ABOUT - О приложении",
        "no_data": "❌ Нет данных для выбранных параметров",
        "loading": "Загрузка данных...",
        "data_table": "📋 Таблица данных",
        "dynamics": "📈 Динамика показателя",
        "source": "Источник: World Bank Open Data",
        "file": "Файл: World_Bank_Report",
        "period": "Период",
        "countries_label": "Страны",
        "date": "Дата",
        "scale_label": "Масштаб",
        "year": "Год",
        "value": "Значение",
        "help_text": """
**📋 Как пользоваться приложением:**

1. **Выберите страны** из списка (можно несколько)
2. **Найдите показатель** через поиск или выберите из списка
3. **Выберите масштаб** (исходный, тысячи, миллионы, миллиарды)
4. **Укажите период** (годы от и до)
5. **Нажмите "Загрузить данные"**

**📊 Результаты:**
- Таблица с данными (годы по строкам, страны по колонкам)
- График динамики показателя
- Кнопки для скачивания CSV и PDF

**💾 Экспорт:**
- **CSV** - для Excel и других таблиц
- **PDF** - готовый отчет с графиком и таблицей

**🔄 Перезагрузка:** кнопка EXIT внизу панели

**⚠️ Важное примечание:**
- **ВВП** и **Население** — доступны для большинства стран (200+)
- **Другие показатели** — могут быть доступны не для всех стран
- Если нет данных — в таблице отображается прочерк "–"
        """,
        "about_text": """
**🌍 World Bank Data Explorer**

Приложение для анализа данных Всемирного банка.

**Данные:** World Bank Open Data (API)

**Показатели:** ВВП, население, инфляция, безработица, энергетика, здравоохранение и другие

**Разработка:** GABDUL741 + DeepSeek AI

**Технологии:** Python, Streamlit, Plotly, Matplotlib, ReportLab

**Версия:** 1.0
        """
    },
    "English": {
        "title": "📊 World Bank Data Explorer",
        "settings": "⚙️ Settings",
        "search_country": "🔍 Search country",
        "search_country_placeholder": "Russia, USA, Germany...",
        "countries": "🌍 Countries",
        "search_indicator": "🔍 Search indicator",
        "search_indicator_placeholder": "GDP, population...",
        "indicator": "📈 Indicator",
        "scale": "📏 Scale",
        "year_from": "Year from",
        "year_to": "Year to",
        "load": "🔄 Load data",
        "exit": "🚪 EXIT / Reload",
        "exit_tip": "💡 **Tip:** If you don't see the EXIT button, drag the right border of the sidebar →",
        "help_title": "❓ HELP - Instructions",
        "about_title": "ℹ️ ABOUT - About app",
        "no_data": "❌ No data for selected parameters",
        "loading": "Loading data...",
        "data_table": "📋 Data table",
        "dynamics": "📈 Indicator dynamics",
        "source": "Source: World Bank Open Data",
        "file": "File: World_Bank_Report",
        "period": "Period",
        "countries_label": "Countries",
        "date": "Date",
        "scale_label": "Scale",
        "year": "Year",
        "value": "Value",
        "help_text": """
**📋 How to use the app:**

1. **Select countries** from the list (multiple selection allowed)
2. **Find indicator** via search or select from list
3. **Select scale** (original, thousands, millions, billions)
4. **Specify period** (years from and to)
5. **Click "Load data"**

**📊 Results:**
- Data table (years as rows, countries as columns)
- Indicator dynamics chart
- CSV and PDF download buttons

**💾 Export:**
- **CSV** - for Excel and other spreadsheets
- **PDF** - ready report with chart and table

**🔄 Reload:** EXIT button at the bottom of the sidebar

**⚠️ Important note:**
- **GDP** and **Population** — available for most countries (200+)
- **Other indicators** — may not be available for all countries
- If no data — dash "–" is displayed in the table
        """,
        "about_text": """
**🌍 World Bank Data Explorer**

Application for analyzing World Bank data.

**Data:** World Bank Open Data (API)

**Indicators:** GDP, population, inflation, unemployment, energy, healthcare and others

**Development:** GABDUL741 + DeepSeek AI

**Technologies:** Python, Streamlit, Plotly, Matplotlib, ReportLab

**Version:** 1.0
        """
    }
}# ===== ПЕРЕВОДЫ =====
TEXTS = {
    "Русский": {
        "title": "📊 World Bank Data Explorer",
        "settings": "⚙️ Настройки",
        "search_country": "🔍 Поиск страны",
        "search_country_placeholder": "Россия, США, Германия...",
        "countries": "🌍 Страны",
        "search_indicator": "🔍 Поиск показателя",
        "search_indicator_placeholder": "ВВП, население...",
        "indicator": "📈 Показатель",
        "scale": "📏 Масштаб",
        "year_from": "Год от",
        "year_to": "Год до",
        "load": "🔄 Загрузить данные",
        "exit": "🚪 EXIT / Перезагрузка",
        "exit_tip": "💡 **Совет:** Если не видите кнопку EXIT, потяните правую границу боковой панели →",
        "help_title": "❓ HELP - Инструкция",
        "about_title": "ℹ️ ABOUT - О приложении",
        "no_data": "❌ Нет данных для выбранных параметров",
        "loading": "Загрузка данных...",
        "data_table": "📋 Таблица данных",
        "dynamics": "📈 Динамика показателя",
        "source": "Источник: World Bank Open Data",
        "file": "Файл: World_Bank_Report",
        "period": "Период",
        "countries_label": "Страны",
        "date": "Дата",
        "scale_label": "Масштаб",
        "year": "Год",
        "value": "Значение",
        "help_text": """
**📋 Как пользоваться приложением:**

1. **Выберите страны** из списка (можно несколько)
2. **Найдите показатель** через поиск или выберите из списка
3. **Выберите масштаб** (исходный, тысячи, миллионы, миллиарды)
4. **Укажите период** (годы от и до)
5. **Нажмите "Загрузить данные"**

**📊 Результаты:**
- Таблица с данными (годы по строкам, страны по колонкам)
- График динамики показателя
- Кнопки для скачивания CSV и PDF

**💾 Экспорт:**
- **CSV** - для Excel и других таблиц
- **PDF** - готовый отчет с графиком и таблицей

**🔄 Перезагрузка:** кнопка EXIT внизу панели

**⚠️ Важное примечание:**
- **ВВП** и **Население** — доступны для большинства стран (200+)
- **Другие показатели** — могут быть доступны не для всех стран
- Если нет данных — в таблице отображается прочерк "–"
        """,
        "about_text": """
**🌍 World Bank Data Explorer**

Приложение для анализа данных Всемирного банка.

**Данные:** World Bank Open Data (API)

**Показатели:** ВВП, население, инфляция, безработица, энергетика, здравоохранение и другие

**Разработка:** GABDUL741 + DeepSeek AI

**Технологии:** Python, Streamlit, Plotly, Matplotlib, ReportLab

**Версия:** 1.0
        """
    },
    "English": {
        "title": "📊 World Bank Data Explorer",
        "settings": "⚙️ Settings",
        "search_country": "🔍 Search country",
        "search_country_placeholder": "Russia, USA, Germany...",
        "countries": "🌍 Countries",
        "search_indicator": "🔍 Search indicator",
        "search_indicator_placeholder": "GDP, population...",
        "indicator": "📈 Indicator",
        "scale": "📏 Scale",
        "year_from": "Year from",
        "year_to": "Year to",
        "load": "🔄 Load data",
        "exit": "🚪 EXIT / Reload",
        "exit_tip": "💡 **Tip:** If you don't see the EXIT button, drag the right border of the sidebar →",
        "help_title": "❓ HELP - Instructions",
        "about_title": "ℹ️ ABOUT - About app",
        "no_data": "❌ No data for selected parameters",
        "loading": "Loading data...",
        "data_table": "📋 Data table",
        "dynamics": "📈 Indicator dynamics",
        "source": "Source: World Bank Open Data",
        "file": "File: World_Bank_Report",
        "period": "Period",
        "countries_label": "Countries",
        "date": "Date",
        "scale_label": "Scale",
        "year": "Year",
        "value": "Value",
        "help_text": """
**📋 How to use the app:**

1. **Select countries** from the list (multiple selection allowed)
2. **Find indicator** via search or select from list
3. **Select scale** (original, thousands, millions, billions)
4. **Specify period** (years from and to)
5. **Click "Load data"**

**📊 Results:**
- Data table (years as rows, countries as columns)
- Indicator dynamics chart
- CSV and PDF download buttons

**💾 Export:**
- **CSV** - for Excel and other spreadsheets
- **PDF** - ready report with chart and table

**🔄 Reload:** EXIT button at the bottom of the sidebar

**⚠️ Important note:**
- **GDP** and **Population** — available for most countries (200+)
- **Other indicators** — may not be available for all countries
- If no data — dash "–" is displayed in the table
        """,
        "about_text": """
**🌍 World Bank Data Explorer**

Application for analyzing World Bank data.

**Data:** World Bank Open Data (API)

**Indicators:** GDP, population, inflation, unemployment, energy, healthcare and others

**Development:** GABDUL741 + DeepSeek AI

**Technologies:** Python, Streamlit, Plotly, Matplotlib, ReportLab

**Version:** 1.0
        """
    }
}
#INDICATORS 
  # ===== ПОКАЗАТЕЛИ (Русский/English) =====
INDICATORS_RU = {
    "NY.GDP.MKTP.CD": "💵 ВВП (долл. США)",
    "NY.GDP.PCAP.CD": "💵 ВВП на душу (долл. США)",
    "NY.GDP.MKTP.KD.ZG": "📈 Рост ВВП (годовой %)",
    "NE.CON.TOTL.CD": "🛒 Расходы на потребление",
    "NE.IMP.GNFS.CD": "📥 Импорт товаров и услуг",
    "NE.EXP.GNFS.CD": "📤 Экспорт товаров и услуг",
    "SP.POP.TOTL": "👥 Население, всего",
    "SP.POP.GROW": "📊 Рост населения (%)",
    "SP.POP.0014.TO.ZS": "🧒 Население 0-14 лет (%)",
    "SP.POP.1564.TO.ZS": "👨 Население 15-64 лет (%)",
    "SP.POP.65UP.TO.ZS": "👴 Население 65+ лет (%)",
    "SP.DYN.LE00.IN": "❤️ Продолжительность жизни",
    "SP.URB.TOTL.IN.ZS": "🏙️ Городское население (%)",
    "SL.UEM.TOTL.ZS": "⚠️ Безработица (%)",
    "SL.TLF.TOTL.IN": "💼 Рабочая сила",
    "SL.EMP.TOTL.SP.ZS": "✅ Занятость (% населения)",
    "SH.DYN.MORT": "👶 Детская смертность (на 1000)",
    "SH.XPD.CHEX.GD.ZS": "🏥 Расходы на здравоохранение (% ВВП)",
    "EN.ATM.CO2E.PC": "🌍 Выбросы CO2 (тонн/чел)",
    "EG.USE.COMM.GD.PP.KD": "⚡ Энергопотребление на душу",
    "AG.LND.FRST.ZS": "🌲 Лесистость (%)",
    "SE.XPD.TOTL.GD.ZS": "📚 Расходы на образование (% ВВП)",
    "NV.AGR.TOTL.ZS": "🌾 Сельское хозяйство (% ВВП)",
    "NV.IND.TOTL.ZS": "🏭 Промышленность (% ВВП)",
    "NV.SRV.TOTL.ZS": "💻 Услуги (% ВВП)",
    "FP.CPI.TOTL.ZG": "📈 Инфляция (%)",
}

INDICATORS_EN = {
    "NY.GDP.MKTP.CD": "💵 GDP (USD)",
    "NY.GDP.PCAP.CD": "💵 GDP per capita (USD)",
    "NY.GDP.MKTP.KD.ZG": "📈 GDP growth (annual %)",
    "NE.CON.TOTL.CD": "🛒 Final consumption expenditure",
    "NE.IMP.GNFS.CD": "📥 Imports of goods and services",
    "NE.EXP.GNFS.CD": "📤 Exports of goods and services",
    "SP.POP.TOTL": "👥 Population, total",
    "SP.POP.GROW": "📊 Population growth (%)",
    "SP.POP.0014.TO.ZS": "🧒 Population ages 0-14 (%)",
    "SP.POP.1564.TO.ZS": "👨 Population ages 15-64 (%)",
    "SP.POP.65UP.TO.ZS": "👴 Population ages 65+ (%)",
    "SP.DYN.LE00.IN": "❤️ Life expectancy at birth",
    "SP.URB.TOTL.IN.ZS": "🏙️ Urban population (%)",
    "SL.UEM.TOTL.ZS": "⚠️ Unemployment (%)",
    "SL.TLF.TOTL.IN": "💼 Labor force, total",
    "SL.EMP.TOTL.SP.ZS": "✅ Employment to population ratio (%)",
    "SH.DYN.MORT": "👶 Mortality rate, under-5 (per 1,000)",
    "SH.XPD.CHEX.GD.ZS": "🏥 Health expenditure (% of GDP)",
    "EN.ATM.CO2E.PC": "🌍 CO2 emissions (metric tons per capita)",
    "EG.USE.COMM.GD.PP.KD": "⚡ Energy use (kg per capita)",
    "AG.LND.FRST.ZS": "🌲 Forest area (%)",
    "SE.XPD.TOTL.GD.ZS": "📚 Education expenditure (% of GDP)",
    "NV.AGR.TOTL.ZS": "🌾 Agriculture (% of GDP)",
    "NV.IND.TOTL.ZS": "🏭 Industry (% of GDP)",
    "NV.SRV.TOTL.ZS": "💻 Services (% of GDP)",
    "FP.CPI.TOTL.ZG": "📈 Inflation (%)",
}

# ===== МАСШТАБЫ (Русский/English) =====
SCALES_RU = {
    "Исходный": 1,
    "Тысячи": 1_000,
    "Миллионы": 1_000_000,
    "Миллиарды": 1_000_000_000
}

SCALES_EN = {
    "Original": 1,
    "Thousands": 1_000,
    "Millions": 1_000_000,
    "Billions": 1_000_000_000
}
# ===== ФУНКЦИЯ ЗАГРУЗКИ СПИСКА СТРАН =====
@st.cache_data(ttl=86400)

 # ===== ФУНКЦИЯ ЗАГРУЗКИ ВСЕХ СТРАН =====
@st.cache_data(ttl=86400)
def get_countries_list():
    """Загружает список стран из World Bank API с защитой от зависания"""
    import time
    
    countries = {}
    url = "http://api.worldbank.org/v2/country?format=json&per_page=300"
    
    try:
        # Устанавливаем таймаут 10 секунд
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if len(data) > 1:
                for country in data[1]:
                    # Исключаем агрегатные группы
                    if country.get('region', {}).get('id') != 'Aggregates':
                        code = country.get('iso2Code', '')
                        name = country.get('name', '')
                        if code and name and len(code) == 2:
                            countries[code] = name
    except requests.exceptions.Timeout:
        st.warning("⚠️ Таймаут загрузки списка стран. Использую резервный список.")
    except Exception as e:
        st.warning(f"⚠️ Ошибка загрузки стран: {e}")
    
    # Резервный список (если API не ответил)
    if not countries:
        countries = {
            "RU": "Россия", "US": "США", "DE": "Германия",
            "CN": "Китай", "IN": "Индия", "GB": "Великобритания",
            "FR": "Франция", "JP": "Япония", "BR": "Бразилия",
            "IT": "Италия", "CA": "Канада", "AU": "Австралия",
            "KR": "Южная Корея", "MX": "Мексика", "TR": "Турция",
            "NL": "Нидерланды", "CH": "Швейцария", "SE": "Швеция",
            "NO": "Норвегия", "DK": "Дания", "FI": "Финляндия",
            "PL": "Польша", "CZ": "Чехия", "AT": "Австрия",
            "BE": "Бельгия", "PT": "Португалия", "GR": "Греция",
        }
    
    return dict(sorted(countries.items(), key=lambda x: x[1]))
    countries_dict = get_countries_list()   # ← ЭТА СТРОКА

#return dict(sorted(countries.items(), key=lambda x: x[1]))   # ← ТОЛЬКО ОДНА СТРОКА
#    return dict(sorted(countries.items(), key=lambda x: x[1]))

# ===== ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ =====
@st.cache_data(ttl=3600)
def load_data_from_wb(indicator_code, countries_list, start_year, end_year, country_names):
    all_data = []
    for country_code in countries_list:
        url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json&date={start_year}:{end_year}&per_page=100"
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                continue
            data = response.json()
            if len(data) < 2 or not data[1]:
                continue
            for item in data[1]:
                if item.get('value') is not None and item['value'] != '':
                    all_data.append({
                        'country': country_names.get(country_code, country_code),
                        'date': item['date'],
                        'value': float(item['value'])
                    })
        except:
            continue
    if not all_data:
        return pd.DataFrame()
    return pd.DataFrame(all_data)

# ===== ФУНКЦИЯ ЭКСПОРТА В PDF =====
def export_to_pdf(df, pivot, indicator_name, scale_name, countries, start_year, end_year, lang):
    if not PDF_AVAILABLE:
        return None
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import tempfile
    import io
    import os
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
    # Получаем переводы
    t = TEXTS[lang]
    
    try:
        pdfmetrics.registerFont(TTFont('DejaVu', 'DejaVuSans.ttf'))
        FONT_NAME = 'DejaVu'
    except:
        FONT_NAME = 'Helvetica'
    
    df_clean = df.dropna(subset=['value_scaled']).copy()
    df_clean['date'] = df_clean['date'].astype(str)
    df_clean = df_clean.sort_values('date', ascending=True)
    
    if df_clean.empty:
        return None
    
    # ===== ГРАФИК =====
    fig, ax = plt.subplots(figsize=(10, 5))
    unique_countries = df_clean['country'].unique()
    color_list = plt.cm.Set1(range(len(unique_countries)))
    
    for i, country in enumerate(unique_countries):
        country_data = df_clean[df_clean['country'] == country]
        ax.plot(country_data['date'], country_data['value_scaled'], 
                marker='o', label=country, color=color_list[i], linewidth=2, markersize=4)
    
    ax.set_xlabel(t["year"])
    ax.set_ylabel(f'{t["value"]} ({scale_name})')
    ax.set_title(indicator_name)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
        chart_path = tmp.name
    plt.close()
    
    # ===== PDF =====
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=15*mm, leftMargin=15*mm, topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=14, fontName=FONT_NAME, spaceAfter=10)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#666666'), fontName=FONT_NAME)
    table_title_style = ParagraphStyle('TableTitle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#4472C4'), fontName=FONT_NAME)
    
    story = []
    
    # ===== СТРАНИЦА 1: ЗАГОЛОВОК И ГРАФИК =====
    story.append(Paragraph(f"{indicator_name} ({scale_name})", title_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph(f"{t['period']}: {start_year} - {end_year}", subtitle_style))
    story.append(Paragraph(f"{t['countries_label']}: {', '.join(countries)}", subtitle_style))
    story.append(Paragraph(f"{t['date']}: {datetime.now().strftime('%d.%m.%Y %H:%M')}", subtitle_style))
    story.append(Paragraph(t["source"], subtitle_style))
    story.append(Paragraph(t["file"], subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph(t["dynamics"], title_style))
    story.append(Spacer(1, 5))
    story.append(Image(chart_path, width=500, height=280))
    story.append(Spacer(1, 15))
    
    # ===== ПОДГОТОВКА ТАБЛИЦЫ =====
    table_df = df_clean.pivot(index="date", columns="country", values="value_scaled").round(2).sort_index()
    headers = [t["year"]] + list(table_df.columns)
    
    # Функция форматирования строки
    def format_row(row):
        row_data = [str(row.name)]
        for val in row:
            if pd.isna(val):
                row_data.append("-")
            else:
                row_data.append(f"{val:,.2f}".replace(',', ' '))
        return row_data
    
    # Количество строк на страницу
    rows_per_page = 28
    total_rows = len(table_df)
    first_page_rows = 15
    
    # ===== ПЕРВАЯ СТРАНИЦА ТАБЛИЦЫ =====
    story.append(Paragraph(f"{indicator_name} ({scale_name})", table_title_style))
    story.append(Spacer(1, 5))
    
    page_data = table_df.iloc[0:first_page_rows]
    table_data = [headers]
    for idx, row in page_data.iterrows():
        table_data.append(format_row(row))
    
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFFFFF')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), FONT_NAME),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (-1, -1), FONT_NAME),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#CCCCCC')),
    ]))
    story.append(table)
    
    # ===== ПОСЛЕДУЮЩИЕ СТРАНИЦЫ ТАБЛИЦЫ =====
    remaining_rows = total_rows - first_page_rows
    if remaining_rows > 0:
        story.append(PageBreak())
        current_start = first_page_rows
        
        while current_start < total_rows:
            # ЗАГОЛОВОК НА КАЖДОЙ СТРАНИЦЕ
            story.append(Paragraph(f"{indicator_name} ({scale_name})", table_title_style))
            story.append(Spacer(1, 5))
            
            end_idx = min(current_start + rows_per_page, total_rows)
            page_data = table_df.iloc[current_start:end_idx]
            
            table_data = [headers]
            for idx, row in page_data.iterrows():
                table_data.append(format_row(row))
            
            table = Table(table_data, repeatRows=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#FFFFFF')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), FONT_NAME),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTNAME', (0, 1), (-1, -1), FONT_NAME),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#CCCCCC')),
            ]))
            story.append(table)
            
            current_start = end_idx
            if current_start < total_rows:
                story.append(PageBreak())
    
    doc.build(story)
    
    os.unlink(chart_path)
    buffer.seek(0)
    return buffer.getvalue()
# ===== ЗАГРУЗКА СПИСКА СТРАН =====
countries_dict = get_countries_list()

# ===== БОКОВАЯ ПАНЕЛЬ =====
# ===== БОКОВАЯ ПАНЕЛЬ =====
with st.sidebar:
    # Переключатель языка
    language = st.radio("🌐 Language / Язык", ["Русский", "English"], key="lang")
        # Выбор словарей в зависимости от языка
    if language == "Русский":
        INDICATORS = INDICATORS_RU
        SCALES = SCALES_RU
    else:
        INDICATORS = INDICATORS_EN
        SCALES = SCALES_EN
    st.divider()
    
    # Выбираем нужный словарь
    if language == "Русский":
        t = TEXTS["Русский"]
    else:
        t = TEXTS["English"]
    
    st.header(t["settings"])
    
    # Поиск стран
    country_search = st.text_input(t["search_country"], placeholder=t["search_country_placeholder"], key="country_search")
    
    if country_search:
        filtered_countries = {
            code: name for code, name in countries_dict.items()
            if country_search.lower() in name.lower()
        }
    else:
        filtered_countries = countries_dict
    
    selected_countries = st.multiselect(
        t["countries"],
        options=list(filtered_countries.keys()),
        format_func=lambda x: filtered_countries[x],
        default=["RU", "US", "DE"] if "RU" in filtered_countries else [],
        key="countries"
    )
    
    # Поиск показателей
    search_term = st.text_input(t["search_indicator"], placeholder=t["search_indicator_placeholder"], key="search")
    
    if search_term:
        filtered_indicators = {k: v for k, v in INDICATORS.items() if search_term.lower() in v.lower()}
    else:
        filtered_indicators = INDICATORS
    
    selected_indicator = st.selectbox(
        t["indicator"],
        options=list(filtered_indicators.keys()),
        format_func=lambda x: filtered_indicators[x],
        key="indicator"
    )
    
    selected_scale = st.selectbox(
        t["scale"],
        options=list(SCALES.keys()),
        key="scale"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.number_input(t["year_from"], 1960, 2023, 2000, key="start")
    with col2:
        end_year = st.number_input(t["year_to"], 1960, 2023, 2023, key="end")
    
    st.divider()
    
    # HELP
    with st.expander(t["help_title"]):
        st.markdown(t["help_text"])
    
    # ABOUT
    with st.expander(t["about_title"]):
        st.markdown(t["about_text"])
    
    st.divider()
    
    load_button = st.button(t["load"], type="primary", key="load")
    exit_button = st.button(t["exit"], type="secondary", key="exit")
    
    st.caption(t["exit_tip"])

# ===== EXIT С ПЕРЕЗАГРУЗКОЙ =====
if exit_button:
    st.cache_data.clear()
    st.rerun()

# ===== ОСНОВНОЙ БЛОК =====
if load_button and selected_countries:
    # Получаем язык
    if 'lang' in st.session_state:
        lang = st.session_state.lang
    else:
        lang = "Русский"
    
    if lang == "Русский":
        t = TEXTS["Русский"]
    else:
        t = TEXTS["English"]
    
    with st.spinner(t["loading"]):
        df = load_data_from_wb(selected_indicator, selected_countries, start_year, end_year, countries_dict)
        
        if df.empty:
            st.error(t["no_data"])
        else:
            scale_factor = SCALES[selected_scale]
            df["value_scaled"] = df["value"] / scale_factor
            
            pivot = df.pivot(index="date", columns="country", values="value_scaled").round(2)
            pivot = pivot.sort_index()
            
            scale_suffix = f" ({selected_scale})" if selected_scale != "Исходный" else ""
            st.subheader(f"📋 {INDICATORS[selected_indicator]}{scale_suffix}")
            st.dataframe(pivot, use_container_width=True)
            
            # ГРАФИК (теперь t определена)
            fig = px.line(
                df, x="date", y="value_scaled", color="country", markers=True,
                labels={"date": t["year"], "value_scaled": f'{t["value"]} ({selected_scale})', "country": t["countries_label"]}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ... остальной код (CSV, PDF) ...
            
            # Экспорт CSV в том же стиле, что и PDF
            # Создаем сводную таблицу как в PDF

            # Экспорт CSV в том же масштабе, что и PDF
            # Используем value_scaled (уже с масштабом)
                   # CSV экспорт (с переводом и форматированием)
            csv_pivot = df.pivot(index="date", columns="country", values="value_scaled").round(2)
            csv_pivot = csv_pivot.sort_index()
            
            # Форматируем числа с пробелами
            for col in csv_pivot.columns:
                csv_pivot[col] = csv_pivot[col].apply(
                    lambda x: f"{x:,.2f}".replace(',', ' ') if pd.notna(x) else ""
                )
            
            csv_pivot = csv_pivot.reset_index()
            csv_pivot = csv_pivot.rename(columns={'date': t["year"]})
            
            # Заголовок на выбранном языке
            scale_suffix = f" ({selected_scale})" if selected_scale != "Исходный" else ""
            csv_header = f'"{INDICATORS[selected_indicator]}{scale_suffix}"\n'
            csv_header += f'{t["period"]}: {start_year} - {end_year}\n'
            csv_header += f'{t["countries_label"]}: {", ".join([countries_dict[c] for c in selected_countries])}\n'
            csv_header += f'{t["scale_label"]}: {selected_scale}\n'
            csv_header += f'{t["date"]}: {datetime.now().strftime("%d.%m.%Y %H:%M")}\n'
            csv_header += f'{t["source"]}\n'
            csv_header += f'{t["file"]}\n\n'
            
            csv_data = csv_pivot.to_csv(index=False)
            full_csv = csv_header + csv_data
            csv_bytes = full_csv.encode('utf-8-sig')
            
            st.download_button(
                t["load_csv"],
                csv_bytes,
                f"{REPORT_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                key="csv"
            )
            if PDF_AVAILABLE:
                pdf_data = export_to_pdf(df, pivot, INDICATORS[selected_indicator], selected_scale,
                            [countries_dict[c] for c in selected_countries], start_year, end_year, language)
                if pdf_data:
                    st.download_button("📄 PDF", pdf_data, f"report.pdf", "application/pdf", key="pdf")
