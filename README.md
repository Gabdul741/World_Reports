# World_Reports
# Портфельный контролёр с ИИ (Prophet)

**Прогноз на 7 дней · Цветные сигналы 🟢🟡🔴 · Для частных инвесторов**

## 🧠 Что это

Приложение на Python, которое:
- скачивает данные по любым активам (акции, нефть, золото, индексы)
- строит прогноз на 7 дней с помощью модели Prophet (Meta)
- сравнивает текущую цену с прогнозной вилкой
- выдаёт цветной сигнал: **купить / держать / продавать**

## 🚀 Как запустить

```bash
git clone https://github.com/ВАШ_ЛОГИН/portfolio-controller
cd portfolio-controller
pip install -r requirements.txt
streamlit run portfolio_controller.py
