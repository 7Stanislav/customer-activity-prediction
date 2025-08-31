# 🛍️ Customer Activity Prediction / Прогноз активности покупателя

## Description / Описание
**EN:**  
Binary classification of customer activity using scikit-learn pipelines and hyperparameter tuning. The project covers end-to-end preprocessing (missing-value imputation, One-Hot Encoding, scaling), model training/evaluation, SHAP-based feature importance, and customer segmentation for actionable recommendations.

**RU:**  
Бинарная классификация признака **«активность»** с помощью пайплайнов scikit-learn и подбора гиперпараметров. Проект включает полный цикл предобработки (импутация пропусков, OHE-кодирование, масштабирование), обучение и оценку моделей, интерпретацию важности признаков через **SHAP**, а также сегментацию покупателей с практическими рекомендациями.

---

## Data & Features / Данные и признаки
**EN:**  
- Mixed feature set: numeric and categorical (e.g., `сервис`, `информирование`, `популярная_категория`).  
- Target: `активность` (binary).  
- IDs excluded from modeling.

**RU:**  
- Смешанный набор признаков: числовые и категориальные (например, `сервис`, `информирование`, `популярная_категория`).  
- Целевая переменная: `активность` (бинарная).  
- Технические поля (ID) исключены из моделирования.

---

## Methodology / Методология
**EN:**  
1. **Preprocessing:** `SimpleImputer` (most_frequent / numeric), `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`, `StandardScaler` for numeric.  
2. **Pipelines & Tuning:** unified `Pipeline` with grid search over models (e.g., Logistic Regression, SVC).  
3. **Evaluation:** standard classification metrics (e.g., accuracy/F1; ROC-AUC if appropriate).  
4. **Interpretability:** **SHAP** for feature importance and business interpretation.  
5. **Segmentation:** post-model segmentation and recommendations per segment.

**RU:**  
1. **Предобработка:** `SimpleImputer`, `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`, масштабирование числовых признаков.  
2. **Пайплайны и подбор:** единый `Pipeline` с перебором гиперпараметров (напр., Logistic Regression, SVC).  
3. **Оценка:** стандартные метрики классификации (accuracy/F1; при необходимости ROC-AUC).  
4. **Интерпретация:** **SHAP** для оценки важности признаков и бизнес-выводов.  
5. **Сегментация:** сегментация и рекомендации по работе с группами клиентов.

---

## Key Findings / Ключевые выводы
**EN:**  
- SHAP analysis highlights the most influential categorical and numeric drivers of activity.  
- Preprocessing (imputation, OHE, scaling) is critical for stable performance across models.  
- The tuned baseline (e.g., Logistic Regression / SVC) achieves solid classification quality and is easy to operationalize.

**RU:**  
- Анализ **SHAP** показал наиболее значимые драйверы активности среди категориальных и числовых признаков.  
- Корректная предобработка (импутация, OHE, масштабирование) критична для стабильного качества.  
- Настроенная базовая модель (например, Logistic Regression / SVC) даёт уверенное качество и проста в продакшене.

---

## Tech Stack / Технологии
- Python, NumPy, Pandas  
- scikit-learn (Pipelines, GridSearch, OneHotEncoder, StandardScaler, LogisticRegression, SVC)  
- SHAP  
- Matplotlib / Seaborn  
- Jupyter Notebook

---

## How to Run / Как запустить
**EN:**  
1. Clone the repo  
   ```
   git clone https://github.com/7Stanislav/customer-activity-prediction
   cd customer-activity-prediction
   ```
2. (Optional) Create venv and install deps  
   ```
   pip install -r requirements.txt
   ```
   or minimal:
   ```
   pip install numpy pandas scikit-learn shap matplotlib seaborn
   ```
3. Open `project.ipynb` in Jupyter Lab/Notebook and run all cells.  
4. Place data files into `./data/` if they are not included.

**RU:**  
1. Клонируйте репозиторий  
   ```
   git clone https://github.com/7Stanislav/customer-activity-prediction
   cd customer-activity-prediction
   ```
2. (Опционально) Создайте окружение и установите зависимости  
   ```
   pip install -r requirements.txt
   ```
   или минимально:
   ```
   pip install numpy pandas scikit-learn shap matplotlib seaborn
   ```
3. Откройте `project.ipynb` в Jupyter и выполните все ячейки.  
4. Поместите файлы данных в `./data/`, если они не входят в репозиторий.
