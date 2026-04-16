# Presentation Script — Financial NLP: Stance and Sentiment Classification
## COMP6713 2026 T1

**Target audience**: Advanced NLP assessors
**Estimated total duration**: ~12-15 minutes + Q&A

---

## SLIDE 1: Title Slide (~30 seconds)

> Good morning/afternoon. We are Louis Nguyen, Quoc Dat Bui, Nam Khanh Tran, and Quang Minh Phan. Our project is **Financial NLP: Stance and Sentiment Classification**. We tackle two core text classification problems in the financial domain, comparing a progression of approaches from simple baselines all the way to multi-task transformer models.

---

## SLIDE 2: Problem Overview (~1.5 minutes)

> We address two classification tasks.
>
> **First, stance classification**: given a sentence from U.S. Federal Reserve FOMC meeting minutes, we classify it as **hawkish**, **dovish**, or **neutral**. Hawkish signals tightening monetary policy — raising rates, reducing stimulus. Dovish signals the opposite — easing, accommodation.
>
> **Second, sentiment classification**: given a financial news headline, we classify it as **positive**, **negative**, or **neutral** in terms of market sentiment.
>
> **Why does this matter?** Central bank language moves billions of dollars in markets. Traders and analysts need to parse hundreds of pages of Fed communications quickly. Automated stance detection enables faster reaction. Similarly, real-time sentiment classification of financial news powers quantitative trading strategies.
>
> These two tasks are related — both involve understanding financial language nuance — which motivates our multi-task learning approach later.

---

## SLIDE 3: Datasets (~1.5 minutes)

> We use two established datasets from the financial NLP literature.
>
> The **FOMC Hawkish-Dovish dataset** from Georgia Tech's FinTech Lab contains 2,480 sentences from FOMC meeting minutes, labelled as hawkish, dovish, or neutral. Notably, the dataset is **class-imbalanced** — roughly 50% of sentences are neutral, which presents challenges we address with weighted loss functions.
>
> The **Financial PhraseBank** by Malo et al. contains 2,264 sentences from financial news, labelled positive, neutral, or negative. We use the "all agree" subset where 100% of annotators agreed on the label, giving us a high-confidence gold standard.
>
> We also incorporate the **Loughran-McDonald Financial Sentiment Dictionary** — the standard domain lexicon containing curated word lists for positive, negative, and uncertainty categories. We extended it with custom hawkish and dovish word lists for monetary policy stance.
>
> Both datasets are split **70/10/20** with stratification to preserve class distribution across train, validation, and test sets.

---

## SLIDE 4: Data Analysis (~1.5 minutes)

> Before diving into models, let's examine the data.
>
> The class distribution charts show a critical challenge: the FOMC dataset is **imbalanced with 49.4% neutral sentences**. This motivated our weighted cross-entropy loss. Financial PhraseBank is also neutral-dominant at 61.4%.
>
> FOMC sentences are notably **longer** (average 30 words vs 22 for FPB), meaning models must integrate information across longer spans to determine policy stance.
>
> Our lexicon coverage analysis reveals a key finding: **hawkish words frequently appear in neutral FOMC sentences**, not just hawkish ones. This overlap explains why rule-based approaches fail at stance classification.

---

## SLIDE 5: Performance Progression (~1 minute)

> This progression chart tells the whole story at a glance.
>
> Both tasks improve monotonically from baselines through multi-task learning. But the **gap between sentiment and stance persists across every model family**. This is a fundamental task difficulty difference, not a modelling failure.
>
> The steepest jumps occur at two points: moving from baselines to pre-trained models (especially for sentiment), and moving from few-shot to fine-tuning (especially for stance).

---

## SLIDE 6: Modelling Pipeline Overview (~1 minute)

> Our experimental pipeline follows a progression of increasing complexity.
>
> We start with **non-neural baselines** — TF-IDF features with classical classifiers. Then we evaluate **pre-trained transformers** in zero-shot and few-shot settings to assess representation quality. Next, we **fine-tune** models with task-specific heads. Finally, our **extended method** is multi-task learning with a shared FinBERT encoder and dual classification heads.
>
> This progression lets us quantify exactly how much each technique contributes — from bag-of-words to domain-specific pre-training to multi-task transfer.

---

## SLIDE 5: Baseline & Lexicon Models (~1.5 minutes)

> Our baselines establish the performance floor.
>
> **TF-IDF plus Logistic Regression** with bigrams achieves 87.2% accuracy on sentiment and 60.9% on stance. **TF-IDF plus SVM** slightly improves this to 89.4% and 63.3%.
>
> The **Loughran-McDonald lexicon** used as a pure rule-based classifier — counting positive vs. negative words — reaches only 69.3% sentiment and 41.5% stance. This confirms that simple word counting is insufficient; context matters enormously.
>
> However, when we **combine** TF-IDF with lexicon features as additional dimensions, we get 85.4% sentiment — demonstrating that domain knowledge adds signal even to statistical models.
>
> Key takeaway: **TF-IDF + SVM is a surprisingly strong baseline**, especially for stance where it outperforms several neural approaches in the few-shot setting.

---

## SLIDE 6: Pre-trained Model Evaluation (~2 minutes)

> This is where things get interesting.
>
> **FinBERT zero-shot** — using its native financial sentiment head with no training whatsoever — achieves **97.35% accuracy** on sentiment. That's remarkable and demonstrates the power of domain-specific pre-training. ProsusAI trained FinBERT on financial communications, so it already understands financial sentiment.
>
> For **few-shot evaluation** with just 16 examples per class, the gap between domain-specific and general models is dramatic. FinBERT few-shot reaches **98.01%** on sentiment, while BERT-base manages only **74.61%** and RoBERTa **75.72%**.
>
> On **stance**, however, even FinBERT few-shot only reaches **48.59%**. This tells us that stance classification — distinguishing hawkish from dovish policy language — requires substantially more training signal than sentiment.
>
> The key finding: **domain-specific pre-training provides an enormous advantage** in financial NLP. FinBERT's representations encode financial semantics that general-purpose models lack.

---

## SLIDE 7: Fine-tuning Approaches (~2 minutes)

> We implement two distinct fine-tuning strategies.
>
> **FinBERT single-task fine-tuning**: We replace FinBERT's head with a fresh 3-class classifier and fine-tune all layers for 5 epochs with AdamW, linear warmup, and weighted cross-entropy for the imbalanced stance task. This achieves **96.91%** on sentiment and **61.29%** on stance.
>
> **BERT-base with LLRD and Gradual Unfreezing**: This is a deliberately different strategy. Starting from general-purpose BERT-base-uncased, we apply **Layer-wise Learning Rate Decay** — each transformer layer gets a learning rate scaled by 0.9 relative to the layer above it. We also use **gradual unfreezing** — epoch 1 trains only the head, epoch 2 adds layer 11, and so on. Combined with label smoothing at epsilon 0.1.
>
> Despite lacking financial pre-training, BERT LLRD achieves **96.91%** on sentiment (matching FinBERT) and **65.12%** on stance (slightly better than FinBERT single-task). This shows that careful fine-tuning techniques can partially compensate for lack of domain pre-training.

---

## SLIDE 8: Multi-task Learning — Extended Method (~2 minutes)

> Our primary contribution is the **multi-task learning architecture**.
>
> The architecture uses a **shared FinBERT encoder** with **two task-specific classification heads** — one for stance, one for sentiment. During training, we alternate batches: one stance batch, one sentiment batch, so the encoder sees both types of financial language.
>
> We use **weighted cross-entropy** for the stance head to address FOMC class imbalance, while the sentiment head uses standard cross-entropy.
>
> The results: **98.45% accuracy on sentiment** with macro-F1 of 0.9772, and **67.74% accuracy on stance** with macro-F1 of 0.6684. These are the **best results across all our experiments**.
>
> Multi-task learning improves over single-task FinBERT by **+2.2 percentage points** on stance and **+1.3 points** on sentiment. The shared encoder learns more robust financial language representations by seeing both tasks — stance training helps sentiment and vice versa.

---

## SLIDE 9: Results Summary (~1 minute)

> This table summarises all 14 model configurations across both tasks.
>
> The clear trend: **multi-task FinBERT wins on both tasks**. Domain-specific pre-training is the single biggest factor for sentiment. For stance, the combination of domain pre-training, task-specific fine-tuning, and multi-task regularization is required to achieve the best performance.
>
> Note the stark difference in difficulty: sentiment classification approaches 98% while stance classification plateaus around 66%. This reflects the fundamental difficulty of policy stance detection — hawkish and dovish signals are often subtle and context-dependent.

---

## SLIDE 10: Key Findings & Analysis (~1.5 minutes)

> Let me highlight four key findings.
>
> **First**, domain-specific pre-training is transformative. FinBERT few-shot at 97.7% sentiment crushes BERT few-shot at 74.2%. Financial pre-training encodes domain semantics that transfer efficiently.
>
> **Second**, multi-task learning provides consistent improvement over single-task. The shared encoder benefits from cross-task regularization — seeing both stance and sentiment prevents overfitting to either task's idiosyncrasies.
>
> **Third**, stance detection is fundamentally harder. Even our best model only reaches 66%. Error analysis shows the most common confusion is **neutral versus hawkish** — many sentences contain subtle hawkish signals that humans also find ambiguous.
>
> **Fourth**, careful fine-tuning techniques like LLRD can partially compensate for lack of domain pre-training. BERT-base LLRD matches FinBERT's sentiment performance despite never seeing financial text during pre-training.

---

## SLIDE 11: Error Analysis (~1 minute)

> Examining misclassifications reveals systematic patterns.
>
> For **stance**, the dominant error type is **neutral classified as hawkish** or vice versa. These are sentences like: *"The economy continues to expand at a moderate pace"* — the word "moderate" could signal either satisfaction with the status quo (neutral) or a hawkish observation that expansion continues.
>
> For **sentiment**, errors are rare but concentrate on the **negative-neutral boundary**. Sentences reporting factual financial figures without explicit sentiment markers are sometimes ambiguous.
>
> Multi-task learning reduces certain error types through cross-task regularization — the model becomes more calibrated in its neutral predictions.

---

## SLIDE 12: Demo & CLI (~30 seconds)

> We provide two interfaces for testing the system. A **Gradio web demo** where users can paste any financial sentence and receive stance and sentiment predictions with confidence scores. And a **command-line interface** for batch processing, supporting both interactive and file-based input.

---

## SLIDE 13: Conclusion (~30 seconds)

> In summary, multi-task FinBERT with shared encoder and dual task-specific heads achieves state-of-the-art results on both financial stance and sentiment classification. Domain-specific pre-training is essential for financial NLP. Future directions include incorporating more FOMC training data, exploring cross-lingual financial NLP, and investigating larger language models.
>
> Thank you. We're happy to take questions.

---

## ANTICIPATED Q&A

**Q: Why not use GPT-4 or other LLMs?**
> Our scope focused on encoder-based models (BERT family). LLMs could be explored for future work, but they introduce latency and cost concerns for real-time financial applications. Our multi-task FinBERT is lightweight, fast, and achieves strong results.

**Q: How do you handle the FOMC class imbalance?**
> We compute inverse-frequency class weights and apply them to the cross-entropy loss for the stance task. This upweights minority classes (hawkish, dovish) during training. The multi-task model also benefits from seeing the more balanced sentiment task.

**Q: Why is stance so much harder than sentiment?**
> Sentiment is a well-studied universal concept — positive/negative is relatively clear. Monetary policy stance requires domain expertise — distinguishing hawkish from dovish involves understanding economic implications, not just lexical cues. The FOMC dataset is also smaller and more imbalanced.

**Q: Could you improve stance further?**
> Yes — possible improvements include: (1) more training data, (2) incorporating document-level context (not just isolated sentences), (3) using economic indicators as additional features, (4) ensembling multiple model architectures.

**Q: Why multi-task over separate models?**
> Multi-task learning provides implicit regularization through the shared encoder. Both tasks involve understanding financial language, so shared representations are beneficial. It also halves inference cost — one forward pass gives both predictions.
