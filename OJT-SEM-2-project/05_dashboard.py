# run this with:  python -m streamlit run 05_dashboard.py

import sys, json, os
sys.path.insert(0, "D:/py_libs")   # scikit-learn installed here

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

import joblib
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score
)

# ── global chart style ───────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "font.size"         : 12,
    "axes.titlesize"    : 14,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 12,
    "axes.labelweight"  : "bold",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.35,
    "grid.linestyle"    : "--",
    "xtick.labelsize"   : 11,
    "ytick.labelsize"   : 11,
    "legend.fontsize"   : 11,
    "figure.dpi"        : 120,
})

SPAM_COLOR = "#E74C3C"
HAM_COLOR  = "#2ECC71"
MED_COLOR  = "#e67e22"

# ── page setup ───────────────────────────────────────────────
st.set_page_config(
    page_title = "SMS Spam Explorer",
    page_icon  = "📱",
    layout     = "wide"
)

# ── loading data ─────────────────────────────────────────────
@st.cache_data
def load():
    return pd.read_csv("outputs/spam_cleaned.csv")

@st.cache_resource
def load_model():
    """Load trained sklearn pipeline from disk."""
    path = "outputs/spam_model.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_ml_results():
    path = "outputs/ml_results.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

data = load()
spam = data[data["label"] == "spam"]
ham  = data[data["label"] == "ham"]
ml_model   = load_model()
ml_results = load_ml_results()

# ── sidebar navigation ───────────────────────────────────────
st.sidebar.title("📱 SMS Spam Explorer")
st.sidebar.markdown("**Alok Chauhan & Aman Kumar**")
st.sidebar.markdown("Batch 2C")
st.sidebar.markdown("---")

page = st.sidebar.radio("Go to:", [
    "🏠 Home",
    "📊 Charts",
    "🔤 Word Analysis",
    "🎯 Segments",
    "🔍 Check a Message",
    "🤖 ML Model"
])

if ml_model:
    st.sidebar.markdown("---")
    meta = (ml_results or {}).get("_meta", {})
    st.sidebar.success(f"Model loaded: **{meta.get('best_model','—')}**")

# ── HOME PAGE ─────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("📱 SMS Spam Data Exploration")
    st.markdown("**Alok Chauhan & Aman Kumar | Batch 2C**")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Messages",     f"{len(data):,}")
    col2.metric("Spam Messages",      f"{len(spam):,}",
                delta=f"{len(spam)/len(data)*100:.1f}% of total",
                delta_color="inverse")
    col3.metric("Ham Messages",       f"{len(ham):,}",
                delta=f"{len(ham)/len(data)*100:.1f}% of total")
    col4.metric("Duplicates Removed", "403")

    st.markdown("---")

    # overview pie chart
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("📋 What We Did")
        st.markdown("""
- Loaded the SMS spam dataset from UCI (5,572 messages)
- Removed **403 duplicate** messages
- Added **11 new feature** columns
- Found patterns that separate spam from ham
- Built a rule-based detection system
        """)
        st.subheader("🏆 Top Findings")
        st.success("📞 Phone numbers in messages → 99.7% spam rate")
        st.error("🔗 URLs are **57×** more common in spam")
        st.warning("⚡ 3 or more spam signals → 100% spam rate")
        st.info("📏 Spam messages are **2× longer** than ham")

    with col_b:
        st.subheader("📊 Dataset Composition")
        fig, ax = plt.subplots(figsize=(5, 5))
        sizes  = [len(spam), len(ham)]
        labels = [f"Spam\n{len(spam):,} msgs\n({len(spam)/len(data)*100:.1f}%)",
                  f"Ham\n{len(ham):,} msgs\n({len(ham)/len(data)*100:.1f}%)"]
        colors = [SPAM_COLOR, HAM_COLOR]
        wedges, texts = ax.pie(
            sizes, labels=labels, colors=colors,
            startangle=90, textprops={"fontsize": 12, "fontweight": "bold"},
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax.set_title("Spam vs Ham Distribution", pad=15)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── CHARTS PAGE ───────────────────────────────────────────────
elif page == "📊 Charts":
    st.title("📊 EDA Charts")
    st.markdown("Visual comparison of spam and ham messages across key features.")
    st.markdown("---")

    # ── row 1: message length distribution + word count boxplot ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📏 Message Length Distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, color in [("Spam", SPAM_COLOR), ("Ham", HAM_COLOR)]:
            d = data[data["label"] == label.lower()]["char_count"]
            ax.hist(d, bins=50, alpha=0.65, color=color, label=label, density=True)

        sp_med = spam["char_count"].median()
        hm_med = ham["char_count"].median()
        ax.axvline(sp_med, color=SPAM_COLOR, linestyle="--", lw=2,
                   label=f"Spam median: {sp_med:.0f} chars")
        ax.axvline(hm_med, color=HAM_COLOR,  linestyle="--", lw=2,
                   label=f"Ham median:  {hm_med:.0f} chars")

        ax.set_title("How Long Are Spam vs Ham Messages?")
        ax.set_xlabel("Number of Characters")
        ax.set_ylabel("Density (proportion of messages)")
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("💡 Spam messages tend to be much longer than ham messages.")

    with col2:
        st.subheader("📝 Word Count Comparison")
        fig, ax = plt.subplots(figsize=(7, 4))
        bp = ax.boxplot(
            [spam["word_count"], ham["word_count"]],
            tick_labels=["Spam", "Ham"],
            patch_artist=True,
            notch=True,
            widths=0.45,
            medianprops={"color": "black", "linewidth": 2.5},
            flierprops={"marker": "o", "markersize": 4, "alpha": 0.4}
        )
        bp["boxes"][0].set_facecolor(SPAM_COLOR)
        bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_facecolor(HAM_COLOR)
        bp["boxes"][1].set_alpha(0.75)

        ax.set_title("Word Count: Spam vs Ham (Box Plot)")
        ax.set_xlabel("Message Type")
        ax.set_ylabel("Number of Words per Message")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("💡 The box shows the middle 50% of messages. Spam has a much wider spread.")

    st.markdown("---")

    # ── row 2: feature comparison grouped bar chart ──────────────
    st.subheader("🔍 Feature Presence: Spam vs Ham (%)")
    names  = ["Has URL", "Has Number", "Has Prize Word",
              "Has 'FREE'", "Has 'CALL'", "Has 'TXT/TEXT'"]
    scols  = ["has_url", "has_number", "has_currency",
              "has_free", "has_call", "has_txt"]
    srates = [spam[c].mean() * 100 for c in scols]
    hrates = [ham[c].mean()  * 100 for c in scols]

    x      = np.arange(len(names))
    width  = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_s = ax.bar(x - width/2, srates, width, label="Spam",
                    color=SPAM_COLOR, alpha=0.85, zorder=3)
    bars_h = ax.bar(x + width/2, hrates, width, label="Ham",
                    color=HAM_COLOR,  alpha=0.85, zorder=3)

    # value labels on bars
    for bar in bars_s:
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=SPAM_COLOR)
    for bar in bars_h:
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                    f"{h:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#27AE60")

    ax.set_title("How Often Does Each Feature Appear in Spam vs Ham Messages?")
    ax.set_xlabel("Feature")
    ax.set_ylabel("% of Messages Containing Feature")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, max(srates) * 1.2)
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("💡 Every feature is significantly more common in spam than ham.")

    st.markdown("---")
    st.subheader("📋 Feature Comparison Table")
    table = pd.DataFrame({
        "Feature"    : names,
        "Spam %"     : [round(s, 1) for s in srates],
        "Ham %"      : [round(h, 1) for h in hrates],
        "Odds Ratio" : [round(s / max(h, 0.1), 1) for s, h in zip(srates, hrates)]
    })
    st.dataframe(table, width="stretch")
    st.caption("Odds Ratio = Spam% ÷ Ham% — higher means the feature is more exclusive to spam.")

# ── WORD ANALYSIS PAGE ────────────────────────────────────────
elif page == "🔤 Word Analysis":
    st.title("🔤 Word Frequency Analysis")
    st.markdown("Most common words found in spam and ham messages (stopwords removed).")
    st.markdown("---")

    STOPWORDS = [
        "i","me","my","we","our","you","your","he","him","his",
        "she","her","it","its","they","them","this","that","am",
        "is","are","was","were","be","been","have","has","had",
        "do","does","did","a","an","the","and","but","if","or",
        "of","at","by","for","with","to","from","in","on","not",
        "no","so","u","ur","r","ok","hi","hey","get","go","got",
        "ll","just","now","will","can","all","up","out","about",
        "what","when","how","also","then","back","more","over"
    ]

    def get_words(message):
        message = str(message).lower()
        words   = message.split()
        clean   = []
        for word in words:
            word = word.strip(".,!?:;()[]\"'")
            if word not in STOPWORDS and len(word) > 2:
                clean.append(word)
        return clean

    n = st.slider("Number of top words to show:", 10, 30, 15)

    with st.spinner("Counting words..."):
        spam_words = []
        for msg in spam["message"]:
            spam_words += get_words(msg)
        ham_words = []
        for msg in ham["message"]:
            ham_words += get_words(msg)
        spam_count = Counter(spam_words)
        ham_count  = Counter(ham_words)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔴 Top Words in Spam Messages")
        top_spam = pd.Series(dict(spam_count.most_common(n))).sort_values()
        bar_height = max(n * 0.42, 4)
        fig, ax = plt.subplots(figsize=(6.5, bar_height))
        bars = ax.barh(top_spam.index, top_spam.values,
                       color=SPAM_COLOR, alpha=0.82, edgecolor="white", height=0.65)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + top_spam.values.max() * 0.01, bar.get_y() + bar.get_height()/2,
                    f"{int(w):,}", va="center", ha="left", fontsize=10)
        ax.set_title(f"Top {n} Words Found in Spam Messages", pad=10)
        ax.set_xlabel("Number of Occurrences")
        ax.set_ylabel("Word")
        ax.set_xlim(0, top_spam.values.max() * 1.15)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.grid(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("🟢 Top Words in Ham Messages")
        top_ham = pd.Series(dict(ham_count.most_common(n))).sort_values()
        bar_height = max(n * 0.42, 4)
        fig, ax = plt.subplots(figsize=(6.5, bar_height))
        bars = ax.barh(top_ham.index, top_ham.values,
                       color=HAM_COLOR, alpha=0.82, edgecolor="white", height=0.65)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + top_ham.values.max() * 0.01, bar.get_y() + bar.get_height()/2,
                    f"{int(w):,}", va="center", ha="left", fontsize=10)
        ax.set_title(f"Top {n} Words Found in Ham Messages", pad=10)
        ax.set_xlabel("Number of Occurrences")
        ax.set_ylabel("Word")
        ax.set_xlim(0, top_ham.values.max() * 1.15)
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.grid(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.caption("💡 Spam heavily uses words like 'free', 'call', 'claim', 'win'. "
               "Ham is more conversational.")

# ── SEGMENTS PAGE ─────────────────────────────────────────────
elif page == "🎯 Segments":
    st.title("🎯 Segmentation Analysis")
    st.markdown("Spam rate (%) broken down by different message characteristics.")
    st.markdown("---")

    avg = data["label_num"].mean() * 100

    choice = st.selectbox("Select a grouping variable:", [
        "Message Length (characters)",
        "Spam Signal Score",
        "Phone Number Present",
        "Exclamation Marks Count"
    ])

    if choice == "Message Length (characters)":
        data2 = data.copy()
        data2["group"] = pd.cut(
            data2["char_count"],
            bins=[0, 50, 100, 160, 300, 9999],
            labels=["0–50 chars", "51–100 chars", "101–160 chars",
                    "161–300 chars", "300+ chars"]
        )
        seg   = data2.groupby("group", observed=True).agg(
            total=("label", "count"),
            spam_count=("label_num", "sum")
        )
        seg["spam_rate"] = (seg["spam_count"] / seg["total"] * 100).round(1)
        title   = "Spam Rate by Message Length"
        x_label = "Message Length Group"

    elif choice == "Spam Signal Score":
        seg = data.groupby("spam_signals").agg(
            total=("label", "count"),
            spam_count=("label_num", "sum")
        )
        seg["spam_rate"] = (seg["spam_count"] / seg["total"] * 100).round(1)
        seg.index = [f"{i} signal{'s' if i != 1 else ''}" for i in seg.index]
        title   = "Spam Rate by Number of Spam Signals Detected"
        x_label = "Number of Spam Signals"

    elif choice == "Phone Number Present":
        seg = data.groupby("has_phone").agg(
            total=("label", "count"),
            spam_count=("label_num", "sum")
        )
        seg["spam_rate"] = (seg["spam_count"] / seg["total"] * 100).round(1)
        seg.index = ["No Phone Number", "Has Phone Number"]
        title   = "Spam Rate: Messages With vs Without a Phone Number"
        x_label = "Phone Number in Message"

    else:
        data2 = data.copy()
        data2["group"] = pd.cut(
            data2["exclamation"],
            bins=[-1, 0, 1, 2, 9999],
            labels=["0 marks", "1 mark", "2 marks", "3+ marks"]
        )
        seg = data2.groupby("group", observed=True).agg(
            total=("label", "count"),
            spam_count=("label_num", "sum")
        )
        seg["spam_rate"] = (seg["spam_count"] / seg["total"] * 100).round(1)
        title   = "Spam Rate by Number of Exclamation Marks"
        x_label = "Exclamation Marks in Message"

    labels = [str(i) for i in seg.index]
    rates  = seg["spam_rate"].values
    totals = seg["total"].values
    colors = [SPAM_COLOR if r > 50 else MED_COLOR if r > 20 else HAM_COLOR
              for r in rates]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, rates, color=colors, width=0.5,
                  alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3)

    # average line
    ax.axhline(avg, color="#2980B9", linestyle="--", lw=2.5, zorder=4,
               label=f"Dataset average: {avg:.1f}%")

    # annotate each bar
    y_max = max(rates) if len(rates) > 0 else 100
    for bar, rate, total in zip(bars, rates, totals):
        bx = bar.get_x() + bar.get_width() / 2
        # rate % label on top of bar
        ax.text(bx, bar.get_height() + y_max * 0.02,
                f"{rate:.1f}%",
                ha="center", va="bottom",
                fontsize=13, fontweight="bold")
        # sample size below the bar
        ax.text(bx, -y_max * 0.07,
                f"n={total:,}",
                ha="center", va="top",
                fontsize=10, color="grey")

    ax.set_title(title, pad=14)
    ax.set_xlabel(x_label, labelpad=12)
    ax.set_ylabel("Spam Rate (%)", labelpad=10)
    ax.set_ylim(-y_max * 0.12, y_max * 1.22)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

    # colour legend
    p_red    = mpatches.Patch(color=SPAM_COLOR, alpha=0.85, label="High spam  (> 50%)")
    p_orange = mpatches.Patch(color=MED_COLOR,  alpha=0.85, label="Medium spam (20–50%)")
    p_green  = mpatches.Patch(color=HAM_COLOR,  alpha=0.85, label="Low spam   (< 20%)")
    ax.legend(handles=[p_red, p_orange, p_green,
                        plt.Line2D([0],[0], color="#2980B9",
                                   linestyle="--", lw=2.5, label=f"Average: {avg:.1f}%")],
              loc="upper left", framealpha=0.9)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("📋 Detailed Numbers")
    display_seg = seg.rename(columns={
        "total"      : "Total Messages",
        "spam_count" : "Spam Count",
        "spam_rate"  : "Spam Rate (%)"
    })
    st.dataframe(display_seg, width="stretch")

# ── CHECK A MESSAGE PAGE ──────────────────────────────────────
elif page == "🔍 Check a Message":
    st.title("🔍 Check Any Message for Spam")
    st.markdown("Type any SMS message below to see how many spam signals it contains.")
    st.markdown("---")

    msg = st.text_area("✏️ Type your message here:",
                        height=120,
                        placeholder="e.g.  FREE prize! Call 08001234567 NOW to claim!")

    if msg:
        signals = {
            "Has a URL (http/www)"       : "http" in msg.lower() or "www" in msg.lower(),
            "Has a phone number (10+ digits)": any(len(w) >= 10 and w.isdigit()
                                                     for w in msg.split()),
            "Has prize words (prize/cash/win)": any(w in msg.lower()
                                                      for w in ["prize","cash","win"]),
            "Has the word FREE"          : "free" in msg.lower(),
            "Has the word CALL"          : "call" in msg.lower(),
            "Has TXT or TEXT"            : "txt" in msg.lower() or "text" in msg.lower(),
            "Has urgency words (urgent/claim/expire)": any(w in msg.lower()
                                                      for w in ["urgent","claim","expire"]),
            "Message is long (> 100 chars)": len(msg) > 100,
            "Has exclamation mark (!)"   : "!" in msg,
        }

        score = sum(signals.values())

        # metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Characters", f"{len(msg):,}")
        col2.metric("Words",      len(msg.split()))
        col3.metric("Spam Score", f"{score} / 9",
                    delta="signals triggered",
                    delta_color="off")

        st.markdown("---")

        # verdict
        if score >= 5:
            st.error(f"🚨 **HIGH SPAM RISK** — {score}/9 signals triggered. "
                     "This message has many characteristics of spam.")
        elif score >= 3:
            st.warning(f"⚠️ **MEDIUM SPAM RISK** — {score}/9 signals triggered. "
                       "Some spam indicators found.")
        elif score >= 1:
            st.info(f"ℹ️ **LOW SPAM RISK** — {score}/9 signals triggered. "
                    "A few minor indicators found.")
        else:
            st.success("✅ **LOOKS LIKE A NORMAL MESSAGE** — No spam signals detected.")

        st.markdown("---")

        # signal breakdown chart
        st.subheader("📊 Signal Breakdown")
        sig_labels = list(signals.keys())
        sig_vals   = [int(v) for v in signals.values()]
        bar_colors = [SPAM_COLOR if v else "#AEB6BF" for v in sig_vals]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(sig_labels, sig_vals, color=bar_colors,
                       alpha=0.85, edgecolor="white", height=0.55)
        ax.set_xlim(0, 1.4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not Found", "Found"], fontsize=12)
        ax.set_title("Which Spam Signals Were Detected in Your Message?", pad=10)
        ax.set_xlabel("Signal Status")
        ax.xaxis.grid(False)

        for bar, val in zip(bars, sig_vals):
            label = "✔  Found" if val else "✘  Not found"
            ax.text(bar.get_width() + 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    label, va="center", fontsize=10,
                    color=SPAM_COLOR if val else "grey")

        found_patch   = mpatches.Patch(color=SPAM_COLOR, alpha=0.85, label="Signal found")
        missing_patch = mpatches.Patch(color="#AEB6BF", alpha=0.85, label="Not found")
        ax.legend(handles=[found_patch, missing_patch], loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("📋 Sample Messages from Dataset")
    n    = st.slider("How many messages to show?", 3, 10, 5)
    filt = st.radio("Filter by:", ["All", "Spam only", "Ham only"],
                    horizontal=True)

    label_map = {"Spam only": "spam", "Ham only": "ham"}
    if filt == "All":
        sample = data.sample(n, random_state=42)
    else:
        sample = data[data["label"] == label_map[filt]].sample(n, random_state=42)

    st.dataframe(
        sample[["label", "message", "char_count", "spam_signals"]]
              .rename(columns={
                  "label"        : "Label",
                  "message"      : "Message",
                  "char_count"   : "Chars",
                  "spam_signals" : "Spam Signals"
              })
              .reset_index(drop=True),
        width="stretch"
    )

# ── ML MODEL PAGE ─────────────────────────────────────────────
elif page == "🤖 ML Model":
    st.title("🤖 Machine Learning Spam Classifier")
    st.markdown("Three text-classification models trained on the SMS spam dataset using **TF-IDF + n-grams**.")
    st.markdown("---")

    if ml_results is None or ml_model is None:
        st.error("Trained model not found. Run `python train_model.py` once, then reload this page.")
        st.stop()

    meta     = ml_results["_meta"]
    names    = meta["model_names"]

    # ── Section 1: Dataset split info ────────────────────────
    st.subheader("📂 Training Setup")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Messages",   f"{meta['total_rows']:,}")
    c2.metric("Training Set",     f"{meta['train_size']:,} (80%)")
    c3.metric("Test Set",         f"{meta['test_size']:,} (20%)")
    c4.metric("Best Model",       meta["best_model"])
    st.markdown("**Vectorizer:** TF-IDF · max 6,000 features · unigrams + bigrams · sublinear TF")
    st.markdown("---")

    # ── Section 1b: Metric Explanations ───────────────────────
    st.subheader("📖 What Each Evaluation Metric Means")
    ea, eb, ec, ed, ee = st.columns(5)
    ea.info("**Accuracy**\n\n% of ALL messages classified correctly.\n\n_Can be misleading with imbalanced data._")
    eb.success("**Precision** (Important)\n\nOf all flagged spam, how many were actually spam?\n\n_High precision = fewer false alarms._")
    ec.warning("**Recall** (VERY Important)\n\nOf all real spam, how many did we catch?\n\n_Missing spam is bad — maximise this!_")
    ed.error("**F1 Score** ⭐\n\nHarmonic mean of Precision & Recall.\n\n_Best overall metric for imbalanced data._")
    ee.info("**Confusion Matrix**\n\nShows True/False Positives & Negatives.\n\n_Reveals exact error types._")
    st.markdown("---")

    # ── Section 2: Model comparison table ────────────────────
    st.subheader("📊 Model Performance Comparison (on 20% held-out test set)")

    rows = []
    for nm in names:
        r = ml_results[nm]
        rows.append({
            "Model"          : nm,
            "Accuracy"       : f"{r['accuracy']*100:.2f}%",
            "Precision"      : f"{r['precision']*100:.2f}%",
            "Recall"         : f"{r['recall']*100:.2f}%",
            "F1 Score ⭐"    : f"{r['f1']*100:.2f}%",
            "ROC-AUC"        : f"{r['roc_auc']*100:.2f}%",
            "5-Fold CV F1"   : f"{r['cv_f1']*100:.2f}%",
        })
    comp_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(comp_df, width="stretch")

    # highlight best model
    best = meta["best_model"]
    b = ml_results[best]
    st.success(
        f"**Best Model: {best}** — "
        f"Accuracy {b['accuracy']*100:.2f}%  |  "
        f"Precision {b['precision']*100:.2f}%  |  "
        f"Recall {b['recall']*100:.2f}%  |  "
        f"F1 Score ⭐ {b['f1']*100:.2f}%  |  "
        f"AUC {b['roc_auc']*100:.2f}%"
    )
    st.markdown("---")

    # ── Section 3: Metric bar chart ───────────────────────────
    st.subheader("📈 Visual Model Comparison")
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    x = np.arange(len(metric_labels))
    width_b = 0.18
    bar_colors = ["#3498DB", "#9B59B6", "#E67E22", "#27AE60"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (nm, col) in enumerate(zip(names, bar_colors)):
        vals = [ml_results[nm][k] * 100 for k in metric_keys]
        bars = ax.bar(x + i * width_b, vals, width_b,
                      label=nm, color=col, alpha=0.85, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x + width_b * (len(names) - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(85, 102)
    ax.set_ylabel("Score (%)")
    ax.set_title("All Four Models Compared Across Key Metrics")
    ax.legend(loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown("---")

    # ── Section 4: Confusion matrices ─────────────────────────
    st.subheader("🔲 Confusion Matrices (Test Set)")
    st_cols = st.columns(len(names))
    for st_col, nm in zip(st_cols, names):
        cm_arr = np.array(ml_results[nm]["confusion_matrix"])
        tn, fp, fn, tp = cm_arr.ravel()
        fig, ax = plt.subplots(figsize=(4.2, 3.5))
        im = ax.imshow(cm_arr, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Ham", "Pred: Spam"], fontsize=10)
        ax.set_yticklabels(["Actual: Ham", "Actual: Spam"], fontsize=10)
        ax.set_title(nm, fontsize=11, fontweight="bold")
        for ri in range(2):
            for ci in range(2):
                val  = cm_arr[ri, ci]
                tcol = "white" if val > cm_arr.max() / 1.5 else "black"
                ax.text(ci, ri, f"{val:,}", ha="center", va="center",
                        fontsize=16, fontweight="bold", color=tcol)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        st_col.pyplot(fig)
        plt.close()
        st_col.caption(f"TP={tp:,} | TN={tn:,} | FP={fp:,} | FN={fn:,}")
    st.markdown("---")

    # ── Section 5: ROC curves ──────────────────────────────────
    st.subheader("📉 ROC Curves (Area Under Curve)")
    fig, ax = plt.subplots(figsize=(7, 5))
    roc_colors = ["#3498DB", "#9B59B6", "#E67E22"]
    for nm, roc_col in zip(names, roc_colors):
        fpr_v = ml_results[nm]["roc_fpr"]
        tpr_v = ml_results[nm]["roc_tpr"]
        auc_v = ml_results[nm]["roc_auc"]
        ax.plot(fpr_v, tpr_v, color=roc_col, lw=2.5,
                label=f"{nm}  (AUC = {auc_v:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random classifier (AUC=0.50)")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="grey")
    ax.set_xlabel("False Positive Rate (Ham predicted as Spam)")
    ax.set_ylabel("True Positive Rate (Spam correctly detected)")
    ax.set_title("ROC Curves — How Well Each Model Detects Spam")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("Higher AUC = better at distinguishing spam from ham across all thresholds.")
    st.markdown("---")

    # ── Section 6: Live ML-powered predictor ──────────────────
    st.subheader("🔍 Live Spam Predictor  (powered by " + best + ")")
    st.markdown("Type any message and the trained ML model will classify it instantly.")

    user_msg = st.text_area(
        "Type a message to classify:",
        height=110,
        placeholder="e.g.  Congratulations! You've won a FREE iPhone. Click here to claim now.",
        key="ml_msg_input"
    )

    if user_msg.strip():
        pred       = ml_model.predict([user_msg])[0]
        prob       = ml_model.predict_proba([user_msg])[0]
        spam_prob  = prob[1] * 100
        ham_prob   = prob[0] * 100

        # verdict
        if pred == 1:
            st.error(f"🚨 **SPAM** — Model confidence: {spam_prob:.1f}%")
        else:
            st.success(f"✅ **HAM (not spam)** — Model confidence: {ham_prob:.1f}%")

        # confidence stacked bar
        fig, ax = plt.subplots(figsize=(9, 1.6))
        ax.barh([""], [ham_prob],  color=HAM_COLOR,  alpha=0.88,
                label=f"Ham   {ham_prob:.1f}%")
        ax.barh([""], [spam_prob], left=[ham_prob], color=SPAM_COLOR, alpha=0.88,
                label=f"Spam  {spam_prob:.1f}%")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Model Confidence (%)")
        ax.set_title("Prediction Confidence Breakdown")
        ax.xaxis.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", framealpha=0.9)
        MIN_W = 8   # only draw label when segment is wide enough to read
        if ham_prob >= MIN_W:
            ax.text(ham_prob / 2, 0, f"{ham_prob:.1f}%",
                    ha="center", va="center", fontsize=13,
                    fontweight="bold", color="white")
        if spam_prob >= MIN_W:
            ax.text(ham_prob + spam_prob / 2, 0, f"{spam_prob:.1f}%",
                    ha="center", va="center", fontsize=13,
                    fontweight="bold", color="white")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        # quick stats
        ms1, ms2, ms3 = st.columns(3)
        ms1.metric("Characters", f"{len(user_msg):,}")
        ms2.metric("Words",      len(user_msg.split()))
        ms3.metric("ML Verdict", "SPAM" if pred == 1 else "HAM")

        # feature importance explanation
        st.markdown("---")
        st.subheader("🧠 Why did the model decide that?")
        try:
            tfidf_step = ml_model.named_steps["tfidf"]
            vec        = tfidf_step.transform([user_msg])
            feat_names = tfidf_step.get_feature_names_out()
            nz_idx     = vec.nonzero()[1]
            scores     = sorted(
                [(feat_names[i], float(vec[0, i])) for i in nz_idx],
                key=lambda x: x[1], reverse=True
            )[:12]
            if scores:
                words, weights = zip(*scores)
                bc = SPAM_COLOR if pred == 1 else HAM_COLOR
                fig, ax = plt.subplots(figsize=(7, max(len(scores)*0.45, 3)))
                ax.barh(list(words), list(weights), color=bc,
                        alpha=0.82, edgecolor="white", height=0.6)
                ax.set_xlabel("TF-IDF Score (influence on decision)")
                ax.set_title(
                    f"Top words that led to '{('SPAM' if pred==1 else 'HAM')}' prediction"
                )
                ax.xaxis.grid(True, linestyle="--", alpha=0.4)
                ax.yaxis.grid(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.caption(
                    "Higher TF-IDF score = the word appeared more distinctively "
                    "in this message and had greater influence on the prediction."
                )
            else:
                st.info("No significant words found in this message.")
        except Exception:
            st.info("Feature importance view is not available for this model.")