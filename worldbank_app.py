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
INDICATORS = {
    # ===== ЭКОНОМИКА =====
    "NY.GDP.MKTP.CD": "💵 ВВП (долл. США)",
    "NY.GDP.PCAP.CD": "💵 ВВП на душу (долл. США)",
    "NY.GDP.MKTP.KD.ZG": "📈 Рост ВВП (%)",
    "NE.CON.TOTL.CD": "🛒 Расходы на потребление (долл. США)",
    "NE.IMP.GNFS.CD": "📥 Импорт товаров и услуг (долл. США)",
    "NE.EXP.GNFS.CD": "📤 Экспорт товаров и услуг (долл. США)",
    
    # ===== НАСЕЛЕНИЕ =====
    "SP.POP.TOTL": "👥 Население (чел)",
    "SP.POP.GROW": "📊 Рост населения (%)",
    "SP.POP.0014.TO.ZS": "🧒 Население 0-14 лет (%)",
    "SP.POP.1564.TO.ZS": "👨 Население 15-64 лет (%)",
    "SP.POP.65UP.TO.ZS": "👴 Население 65+ лет (%)",
    "SP.DYN.LE00.IN": "❤️ Продолжительность жизни (лет)",
    "SP.URB.TOTL.IN.ZS": "🏙️ Городское население (%)",
    
    # ===== РЫНОК ТРУДА =====
    "SL.UEM.TOTL.ZS": "⚠️ Безработица (%)",
    "SL.TLF.TOTL.IN": "💼 Рабочая сила (чел)",
    "SL.EMP.TOTL.SP.ZS": "✅ Занятость (% населения)",
    "SL.UEM.1524.ZS": "🎓 Безработица молодежи (%)",
    
    # ===== ЗДРАВООХРАНЕНИЕ =====
    "SH.DYN.MORT": "👶 Детская смертность (на 1000)",
    "SH.XPD.CHEX.GD.ZS": "🏥 Расходы на здравоохранение (% ВВП)",
    
    # ===== ЭНЕРГЕТИКА =====
    "EN.ATM.CO2E.PC": "🌍 Выбросы CO2 (тонн/чел)",
    "EG.USE.COMM.GD.PP.KD": "⚡ Энергопотребление на душу (кг)",
    "AG.LND.FRST.ZS": "🌲 Лесистость (%)",
    
    # ===== ОБРАЗОВАНИЕ =====
    "SE.XPD.TOTL.GD.ZS": "📚 Расходы на образование (% ВВП)",
    
    # ===== СТРУКТУРА ЭКОНОМИКИ =====
    "NV.AGR.TOTL.ZS": "🌾 Сельское хозяйство (% ВВП)",
    "NV.IND.TOTL.ZS": "🏭 Промышленность (% ВВП)",
    "NV.SRV.TOTL.ZS": "💻 Услуги (% ВВП)",
    
    # ===== ФИНАНСЫ =====
    "FP.CPI.TOTL.ZG": "📈 Инфляция (%)",
}

# ===== МАСШТАБЫ =====
SCALES = {
    "Исходный": 1,
    "Тысячи": 1_000,
    "Миллионы": 1_000_000,
    "Миллиарды": 1_000_000_000
}

# ===== ФУНКЦИЯ ЗАГРУЗКИ СПИСКА СТРАН =====
@st.cache_data(ttl=86400)
def get_countries_list():
    countries = {
        "RU": "Россия", "US": "США", "DE": "Германия",
        "CN": "Китай", "IN": "Индия", "GB": "Великобритания",
        "FR": "Франция", "JP": "Япония", "BR": "Бразилия",
        "IT": "Италия", "CA": "Канада", "AU": "Австралия",
    }
    return dict(sorted(countries.items(), key=lambda x: x[1]))

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
def export_to_pdf(df, pivot, indicator_name, scale_name, countries, start_year, end_year):
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
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    
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
    
    ax.set_xlabel('Год')
    ax.set_ylabel(f'Значение ({scale_name})')
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
    
    story.append(Paragraph(f"{indicator_name} ({scale_name})", title_style))
    story.append(Spacer(1, 5))
    story.append(Paragraph(f"Период: {start_year} - {end_year}", subtitle_style))
    story.append(Paragraph(f"Страны: {', '.join(countries)}", subtitle_style))
    story.append(Paragraph(f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}", subtitle_style))
    story.append(Paragraph(f"Источник: World Bank Open Data", subtitle_style))
    story.append(Paragraph(f"Файл: World_Bank_Report", subtitle_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Динамика показателя", title_style))
    story.append(Spacer(1, 5))
    story.append(Image(chart_path, width=500, height=280))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph(f"{indicator_name} ({scale_name})", table_title_style))
    story.append(Spacer(1, 5))
    
    table_df = df_clean.pivot(index="date", columns="country", values="value_scaled").round(2).sort_index()
    headers = ['Год'] + list(table_df.columns)
    table_data = [headers]
    
    for idx, row in table_df.iterrows():
        row_data = [str(idx)]
        for val in row:
            if pd.isna(val):
                row_data.append("-")
            else:
                row_data.append(f"{val:,.2f}".replace(',', ' '))
        table_data.append(row_data)
    
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
    
    doc.build(story)
    
    os.unlink(chart_path)
    buffer.seek(0)
    return buffer.getvalue()
    
# ===== ЗАГРУЗКА СПИСКА СТРАН =====
countries_dict = get_countries_list()

# ===== БОКОВАЯ ПАНЕЛЬ =====
with st.sidebar:
    st.header("⚙️ Настройки")
    st.divider()
    st.caption(f"📊 Источник: {DATA_SOURCE}")
    st.caption(f"📄 Файл: {REPORT_NAME}")    
    selected_countries = st.multiselect(
        "🌍 Страны",
        options=list(countries_dict.keys()),
        format_func=lambda x: countries_dict[x],
        default=["RU", "US", "DE"],
        key="countries"
    )
    
    # Поиск показателей
    search_term = st.text_input("🔍 Поиск показателя", placeholder="ВВП, население...", key="search")
    
    if search_term:
        filtered_indicators = {k: v for k, v in INDICATORS.items() if search_term.lower() in v.lower()}
    else:
        filtered_indicators = INDICATORS
    
    selected_indicator = st.selectbox(
        "📈 Показатель",
        options=list(filtered_indicators.keys()),
        format_func=lambda x: filtered_indicators[x],
        key="indicator"
    )
    
    selected_scale = st.selectbox(
        "📏 Масштаб",
        options=list(SCALES.keys()),
        key="scale"
    )
    
    start_year = st.number_input("Год от", 1960, 2023, 2000, key="start")
    end_year = st.number_input("Год до", 1960, 2023, 2023, key="end")
    
    load_button = st.button("🔄 Загрузить данные", type="primary", key="load")
    exit_button = st.button("🚪 EXIT", type="secondary", key="exit")

# ===== EXIT =====
if exit_button:
    st.stop()

# ===== ОСНОВНОЙ БЛОК =====
if load_button and selected_countries:
    with st.spinner("Загрузка..."):
        df = load_data_from_wb(selected_indicator, selected_countries, start_year, end_year, countries_dict)
        if df.empty:
            st.error("Нет данных")
        else:
            scale_factor = SCALES[selected_scale]
            df["value_scaled"] = df["value"] / scale_factor
            pivot = df.pivot(index="date", columns="country", values="value_scaled").round(2)
            
            scale_suffix = f" ({selected_scale})" if selected_scale != "Исходный" else ""
            st.subheader(f"📋 {INDICATORS[selected_indicator]}{scale_suffix}")
            st.dataframe(pivot)
            
            fig = px.line(df, x="date", y="value_scaled", color="country", markers=True)
            st.plotly_chart(fig)
            
            # Экспорт CSV в том же стиле, что и PDF
            # Создаем сводную таблицу как в PDF

            # Экспорт CSV в том же масштабе, что и PDF
            # Используем value_scaled (уже с масштабом)
            csv_pivot = df.pivot(index="date", columns="country", values="value_scaled").round(2)
            csv_pivot = csv_pivot.sort_index()
            
            # Форматируем числа с пробелами
            for col in csv_pivot.columns:
                csv_pivot[col] = csv_pivot[col].apply(
                    lambda x: f"{x:,.2f}".replace(',', ' ') if pd.notna(x) else ""
                )
            
            csv_pivot = csv_pivot.reset_index()
            csv_pivot = csv_pivot.rename(columns={'date': 'Год'})
            
            # Заголовок
            scale_suffix = f" ({selected_scale})" if selected_scale != "Исходный" else ""
            csv_header = f'"{INDICATORS[selected_indicator]}{scale_suffix}"\n'
            csv_header += f'Период: {start_year} - {end_year}\n'
            csv_header += f'Страны: {", ".join([countries_dict[c] for c in selected_countries])}\n'
            csv_header += f'Масштаб: {selected_scale}\n'
            csv_header += f'Дата: {datetime.now().strftime("%d.%m.%Y %H:%M")}\n'
            csv_header += f'Источник: {DATA_SOURCE}\n'
            csv_header += f'Файл: {REPORT_NAME}\n\n'
            
            csv_data = csv_pivot.to_csv(index=False)
            full_csv = csv_header + csv_data
            csv_bytes = full_csv.encode('utf-8-sig')
            
            st.download_button(
                "📥 Скачать CSV",
                csv_bytes,
                f"{REPORT_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                key="csv"
            )
            if PDF_AVAILABLE:
                pdf_data = export_to_pdf(df, pivot, INDICATORS[selected_indicator], selected_scale, 
                                        [countries_dict[c] for c in selected_countries], start_year, end_year)
                if pdf_data:
                    st.download_button("📄 PDF", pdf_data, f"report.pdf", "application/pdf", key="pdf")
