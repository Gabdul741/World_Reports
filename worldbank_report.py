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

st.title("🌍 Отчёты по мировым данным")

indicator = st.selectbox("Выберите показатель:", ["ВВП", "Население", "Инфляция", "Безработица"])
scale = st.selectbox("Масштаб значений:", [ "Исходные", "Тысячи", "Миллионы", "Миллиарды"])

if indicator == "ВВП":
    df = pd.read_csv("/home/pikis/gdp.csv")
    ylabel = "ВВП"
    title = "ВВП по странам"
    units = "долл. США"
elif indicator == "Население":
    df = pd.read_csv("/home/pikis/population.csv")
    ylabel = "Население"
    title = "Население по странам"
    units = "чел."
elif indicator == "Инфляция":
    df = pd.read_csv("/home/pikis/inflation.csv")
    df = df.rename(columns={"Country": "Country Name", "Inflation": "Value"})
    ylabel = "Инфляция"
    title = "Инфляция по странам"
    units = "%"
else:
    df = pd.read_csv("/home/pikis/unemployment.csv")
    df = df.rename(columns={"Unemployment Rate": "Value"})
    ylabel = "Безработица"
    title = "Безработица по странам"
    units = "%"
countries = df["Country Name"].unique().tolist()
selected = st.multiselect("Выберите страны:", countries,
    default=["Russian Federation", "United States", "China", "Germany"])

year_from = st.slider("С года:", 1960, 2023, 2000)
year_to = st.slider("По год:", 1960, 2023, 2023)

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
