# sms-spam-detection-pipeline
🛡️ SMS Spam Shield: Analysis & ML Detection
A complete end-to-end Data Science project that explores the anatomy of spam messages and deploys a high-accuracy classifier using Linear SVC and Streamlit.

📌 Project Overview
Messaging platforms are flooded with fraudulent content. This project moves beyond simple keyword filtering by using Natural Language Processing (NLP) to identify complex spam patterns.

This repository covers:

Exploratory Data Analysis (EDA): Visualizing the statistical differences between "Ham" and "Spam".

Rule-Based Segmentation: Identifying "Spam Signals" (URLs, phone numbers, urgency words).

Machine Learning: Training and comparing 4 different classifiers.

Deployment: A live interactive dashboard for real-time message testing.

📊 Key Insights (from our EDA)
The Length Factor: Spam messages are significantly longer (median ~150 chars) compared to legitimate messages (median ~50 chars).

The "3-Signal" Rule: Our analysis showed that any message containing 3 or more spam signals (e.g., a URL + currency symbol + urgency word) is 100% likely to be spam.

Vocabulary: Words like Free, Claim, Txt, and Urgent are the highest predictors of spam.

⚙️ Tech Stack
Language: Python 3.14

Libraries: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

Interface: Streamlit (Dashboard)

Deployment: Hugging Face Spaces
