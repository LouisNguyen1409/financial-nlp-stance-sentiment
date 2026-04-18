# Tài Liệu Kỹ Thuật — Phân Loại Lập Trường & Cảm Xúc Tài Chính

Đây là tài liệu kỹ thuật đầy đủ cho đồ án môn **COMP6713 Natural Language Processing — UNSW 2026 Term 1**.
Phiên bản này viết bằng **tiếng Việt**; bản tiếng Anh tương đương nằm ở `DOC.md` cùng thư mục.
Tài liệu được viết cho người đọc **mới bắt đầu học NLP**: chúng ta sẽ đi từ những khái niệm nền tảng nhất (từ điển, TF-IDF, Logistic Regression) cho tới các kỹ thuật tiên tiến (FinBERT, học đa nhiệm, LLRD, Gradual Unfreezing).
Mọi con số đánh giá, siêu tham số, và chi tiết implementation đều khớp chính xác với mã nguồn trong `src/*.py`, `config.py`, và kết quả lưu tại `results/all_results_summary.json`.

Mục tiêu: sau khi đọc xong, bạn có thể giải thích tại sao mô hình này hoạt động, tái hiện lại được pipeline, và biết mỗi file Python trong project phụ trách việc gì.

---

## Mục Lục

1. [Dự Án Này Làm Gì](#1-dự-án-này-làm-gì)
2. [Lý Thuyết Nền Tảng](#2-lý-thuyết-nền-tảng)
3. [Môi Trường & Thư Viện](#3-môi-trường--thư-viện)
4. [Kiến Trúc Dự Án](#4-kiến-trúc-dự-án)
5. [Chi Tiết Bộ Dữ Liệu](#5-chi-tiết-bộ-dữ-liệu)
6. [Từ Điển Loughran-McDonald](#6-từ-điển-loughran-mcdonald)
7. [Mô Hình Cơ Sở TF-IDF](#7-mô-hình-cơ-sở-tf-idf)
8. [Đánh Giá Transformer Huấn Luyện Sẵn](#8-đánh-giá-transformer-huấn-luyện-sẵn)
9. [Tinh Chỉnh FinBERT và BERT-base](#9-tinh-chỉnh-finbert-và-bert-base)
10. [Học Đa Nhiệm](#10-học-đa-nhiệm)
11. [Phương Pháp Đánh Giá](#11-phương-pháp-đánh-giá)
12. [CLI, Demo, và HuggingFace](#12-cli-demo-và-huggingface)
13. [Các Lỗi Gặp Phải](#13-các-lỗi-gặp-phải)
14. [Phân Tích Kết Quả](#14-phân-tích-kết-quả)
15. [Bài Học Rút Ra](#15-bài-học-rút-ra)
16. [Tham Chiếu Code Đầy Đủ](#16-tham-chiếu-code-đầy-đủ)

---

## 1. Dự Án Này Làm Gì

Đồ án giải quyết **hai bài toán phân loại văn bản tài chính** trên cùng một hệ thống thống nhất:

1. **Stance classification (phân loại lập trường)** — cho một câu lấy từ thông cáo của Federal Open Market Committee (FOMC) của Cục Dự trữ Liên bang Mỹ, dự đoán quan điểm chính sách tiền tệ là:
   - `hawkish` (diều hâu): thắt chặt, tăng lãi suất, rút thanh khoản
   - `dovish` (bồ câu): nới lỏng, giảm lãi suất, bơm thanh khoản
   - `neutral` (trung lập): không thể hiện thiên hướng
2. **Sentiment classification (phân loại cảm xúc)** — cho một câu từ tin tức tài chính của bộ Financial PhraseBank, dự đoán cảm xúc là:
   - `positive` (tích cực)
   - `neutral` (trung tính)
   - `negative` (tiêu cực)

### Vì Sao Hai Task Này Cùng Được Chọn?

Cả hai đều là **phân loại 3 lớp trên văn bản tài chính ngắn (≤ 128 token)**, nên chúng chia sẻ rất nhiều thành phần pipeline: cùng loại tokenizer, cùng kích thước batch, cùng cách đo macro-F1, và quan trọng nhất — ta có thể chia sẻ bộ mã hoá (encoder) của Transformer giữa chúng để thử học đa nhiệm. Tuy giống nhau về hình thức, hai task khác nhau về bản chất: sentiment liên quan đến tác động thị trường (tin tốt / tin xấu), trong khi stance liên quan đến tín hiệu chính sách (tăng / giảm lãi suất). Một câu có thể có cảm xúc tích cực nhưng lập trường diều hâu, hoặc ngược lại. Đặt hai bài toán cạnh nhau giúp ta thấy rõ đâu là kiến thức đặc thù theo miền (domain knowledge) và đâu là biểu diễn ngôn ngữ chung.

### Đối Chiếu Với Bảng Scope Của Môn Học

Bảng dưới đây liệt kê các hạng mục tính điểm theo scope table của COMP6713:

| Hạng mục | Điểm scope | Project làm gì |
|----------|-----------:|----------------|
| **Part A: Dataset evaluation** | 20 | FOMC (2,480 câu) + Financial PhraseBank allagree (2,264 câu); stratified split 70/10/20 với `random_state=42` |
| **Part B: Lexicon-based** | 20 | Loughran-McDonald lexicon: rule-based classifier + hybrid TF-IDF + 8 đặc trưng lexicon |
| **Part C: Pre-trained + fine-tuning** | 80+ | Zero-shot FinBERT native head, 16-shot linear probe trên FinBERT / BERT-base / RoBERTa-base, fine-tune FinBERT đầy đủ, BERT-base với LLRD + Gradual Unfreezing |
| **Part D: Multi-task** | 35 | Shared FinBERT encoder + hai head riêng cho stance / sentiment, alternating batch training, weighted cross-entropy cho stance |
| **Tổng (mục tiêu)** | **155+** | Hoàn thành đầy đủ |

Tất cả các nội dung trên được thực thi bởi 6 bước trong `run_experiments.py`, tập trung output tại `results/all_results_summary.json`.

---

## 2. Lý Thuyết Nền Tảng

Phần này giải thích các khái niệm nền tảng mà bạn cần hiểu trước khi đọc phần implementation. Nếu đã quen với NLP, có thể bỏ qua.

### 2.1 Text Classification Là Gì?

Phân loại văn bản là bài toán: cho một chuỗi văn bản `x`, dự đoán một nhãn `y ∈ {c₁, c₂, …, cK}`. Về mặt toán học, ta học một hàm `f: văn_bản → nhãn`. Hàm `f` có thể là cây quyết định dựa trên đếm từ (lexicon), một mô hình tuyến tính trên đặc trưng TF-IDF, hay một Transformer có hàng trăm triệu tham số. Tất cả đều tuân theo cùng một giao thức đánh giá: chia dữ liệu thành train / val / test, tối ưu hàm mất mát trên train, điều chỉnh trên val, báo số cuối cùng trên test.

### 2.2 TF-IDF

**TF-IDF** (Term Frequency — Inverse Document Frequency) biểu diễn một văn bản bằng một vector thưa trong không gian có chiều bằng kích thước từ vựng. Mỗi phần tử là:

```
tfidf(t, d) = tf(t, d) × idf(t)
idf(t)      = log(N / df(t))
```

Trong đó `tf(t, d)` là số lần token `t` xuất hiện trong văn bản `d`, `df(t)` là số văn bản chứa `t`, và `N` là tổng số văn bản. Trực giác: các từ xuất hiện thường xuyên trong một văn bản **nhưng hiếm trong toàn corpus** có giá trị phân biệt cao. Ví dụ, từ `inflation` xuất hiện nhiều trong một câu sẽ có TF cao; nếu từ đó cũng hiếm trong các câu khác, IDF sẽ cao; kết quả là TF-IDF lớn.

Trong project này, `TfidfVectorizer` của scikit-learn được dùng với `sublinear_tf=True` (thay TF bằng `1 + log(TF)`), `strip_accents="unicode"`, và `ngram_range=(1, 2)` (xét cả unigram và bigram) cho baseline gốc.

### 2.3 Logistic Regression

Hồi quy logistic là mô hình tuyến tính cho phân loại đa lớp:

```
P(y = k | x) = softmax(Wₖ · x + bₖ)
```

Huấn luyện cực tiểu hoá cross-entropy loss:

```
L = −Σᵢ log P(yᵢ | xᵢ)
```

Tham số `class_weight="balanced"` trong sklearn tự động nhân ngược tần suất vào loss để tránh mô hình bỏ qua lớp thiểu số. Logistic Regression làm việc rất tốt trên đặc trưng TF-IDF thưa vì cả hai đều tuyến tính và đều có nhiều chiều.

### 2.4 Transformer và BERT

Transformer (Vaswani et al., 2017) là kiến trúc chỉ dùng cơ chế self-attention thay cho RNN. **BERT** (Devlin et al., 2018) là Transformer encoder được pre-train bằng hai nhiệm vụ tự giám sát trên 3 tỉ token:
- **Masked Language Modeling (MLM):** che ngẫu nhiên 15% token và dự đoán chúng.
- **Next Sentence Prediction (NSP):** đoán xem câu B có tiếp câu A không.

Mỗi câu sau khi tokenize được bọc bởi token đặc biệt `[CLS] ... [SEP]`. Vector đầu ra tại vị trí `[CLS]` được coi là biểu diễn của toàn câu và thường dùng để phân loại. `bert-base-uncased` có 12 transformer layers, hidden size 768, khoảng 110M tham số.

### 2.5 FinBERT

**FinBERT** (`ProsusAI/finbert`) là BERT-base được tiếp tục pre-train (domain-adaptive pre-training) trên corpus tài chính (Reuters TRC2-financial), sau đó fine-tune trên Financial PhraseBank cho sentiment. Do đó FinBERT đã có sẵn một head sentiment với ba nhãn `positive / negative / neutral`. Project khai thác điều này theo hai hướng:
- Dùng head gốc làm zero-shot baseline cho sentiment.
- Gắn head mới (3 lớp) rồi fine-tune lại cho stance.

### 2.6 Zero-shot / Few-shot / Fine-tuning

| Chiến lược | Mô tả | Khi nào dùng |
|------------|-------|--------------|
| **Zero-shot** | Dùng mô hình đã pre-train (hoặc đã fine-tune sẵn trên task khác) để đánh giá trực tiếp. Không huấn luyện gì thêm. | Khi không có dữ liệu đánh nhãn, hoặc để thiết lập baseline đo "giá trị miễn phí" của pre-training. |
| **Few-shot** | Chỉ dùng `k` ví dụ mỗi lớp (thường 4–32) để học. Trong project ta làm **linear probe**: đóng băng encoder, chỉ huấn luyện một tầng Linear trên `[CLS]` embedding. | Khi dữ liệu gán nhãn cực ít, và muốn so sánh **chất lượng biểu diễn** giữa các encoder. |
| **Fine-tuning** | Cập nhật toàn bộ (hoặc gần như toàn bộ) tham số mô hình trên tập huấn luyện đầy đủ. | Khi có vài nghìn câu và muốn đạt accuracy cao nhất. |

### 2.7 Học Đa Nhiệm (Multi-task Learning)

Thay vì huấn luyện hai mô hình riêng, ta có **một encoder chia sẻ** (shared encoder) và **nhiều classification heads** — mỗi task một head. Các batch của hai task được đan xen trong mỗi epoch. Lý thuyết: các lớp thấp của Transformer học biểu diễn ngôn ngữ chung hữu ích cho cả hai task, trong khi các head trên cùng đặc thù hoá theo task. Ưu điểm: hiệu quả tham số, có thể xảy ra **positive transfer** (task này giúp task kia). Nhược điểm: nếu hai task quá khác nhau, xảy ra **negative transfer**.

### 2.8 Weighted Cross-Entropy

Khi một lớp đông áp đảo (neutral chiếm gần nửa FOMC train), mô hình có xu hướng dự đoán lớp đó. Cách khắc phục: nhân hệ số vào loss theo lớp, lớp hiếm có hệ số cao hơn. Công thức dùng trong `compute_class_weights`:

```
wₖ = N / (K · countₖ)
```

trong đó `N` là tổng mẫu, `K` là số lớp, `countₖ` là số mẫu của lớp `k`. Với FOMC train (dovish 455, hawkish 424, neutral 857), ta có weights xấp xỉ **dovish: 1.272, hawkish: 1.365, neutral: 0.675**, đúng như giá trị thực tế in ra trong logs.

### 2.9 Layer-wise Learning Rate Decay (LLRD)

Trong fine-tuning, các lớp thấp của BERT đã học các tính chất cú pháp rất ổn định (dấu câu, cấu trúc từ), còn các lớp cao học đặc trưng ngữ nghĩa trừu tượng hơn. Nếu ta dùng cùng một learning rate cho tất cả, lớp thấp có thể bị phá hỏng. **LLRD** gán LR giảm dần khi đi xuống:

```
lrᵈ = lr_base · decay^depth
```

Trong project, `LLRD_BASE_LR = 2e-5` và `LLRD_DECAY = 0.9`. Head nhận full LR; layer 11 nhận `2e-5 × 0.9`; layer 10 nhận `2e-5 × 0.9²`; ... ; embeddings nhận `2e-5 × 0.9¹³`.

### 2.10 Gradual Unfreezing

Lịch huấn luyện của Howard & Ruder (ULMFiT, 2018):
- Epoch 1: chỉ train head.
- Epoch 2: train head + layer trên cùng.
- Epoch 3: thêm một layer tiếp theo.
- ...

Mục đích: gradient ngẫu nhiên ban đầu từ head chưa được huấn luyện không "quét sạch" các biểu diễn pre-trained ở lớp thấp. Gradual unfreezing đóng vai trò như warmup ngầm (implicit warmup).

### 2.11 Label Smoothing

Cross-entropy tiêu chuẩn sử dụng phân phối đích one-hot (`[0, 1, 0]`). **Label smoothing** với `ε = 0.1` thay bằng `[ε/(K−1), 1−ε, ε/(K−1)] = [0.05, 0.9, 0.05]`. Mô hình được khuyến khích không quá tự tin → giảm overfitting trên tập nhỏ. Project dùng `LABEL_SMOOTHING = 0.1` trong BERT LLRD.

---

## 3. Môi Trường & Thư Viện

### 3.1 Phần Cứng

- **Apple M3 Max** (16-core CPU, 40-core GPU, 64 GB RAM) — máy phát triển chính. GPU được truy cập qua **MPS** (Metal Performance Shaders) backend của PyTorch.
- **NVIDIA H200 cluster** (80 GB VRAM) — dùng cho batch job lớn. Chỉ cần đổi `DEVICE` sang `cuda`, code chạy không cần sửa.

Logic auto-detect trong `config.py`:

```python
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
```

### 3.2 Python và Thư Viện

- **Python 3.12** — Python 3.14 tại thời điểm viết chưa có wheel PyTorch ổn định, nên ta cố định 3.12.
- **torch** — backend tensor, auto-grad, optimizer.
- **transformers** — HuggingFace library, cung cấp `AutoTokenizer`, `AutoModel`, `AutoModelForSequenceClassification`, `pipeline`, `get_linear_schedule_with_warmup`.
- **datasets** — tải dataset từ HuggingFace Hub (FOMC, Financial PhraseBank).
- **scikit-learn** — `TfidfVectorizer`, `LogisticRegression`, `LinearSVC`, `classification_report`, `confusion_matrix`, `train_test_split`.
- **pandas** + **numpy** — thao tác DataFrame và tensor thấp cấp.
- **matplotlib** + **seaborn** — vẽ confusion matrix và 10 biểu đồ phân tích.
- **gradio** — web demo chạy trên cổng `7860`.
- **accelerate** — phụ trợ cho PyTorch khi chạy multi-GPU / mixed precision.
- **tqdm** — thanh progress.

Cài đặt:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3.3 Giải Thích Các Hằng Số Trong `config.py`

```python
MAX_SEQ_LENGTH  = 128     # Cắt/pad về 128 token — đủ cho câu ngắn
BATCH_SIZE      = 32      # Fit M3 Max 64GB; tăng lên 64/128 trên H200
LEARNING_RATE   = 2e-5    # Chuẩn BERT fine-tuning của Devlin et al.
WEIGHT_DECAY    = 0.01    # Chuẩn AdamW
FINETUNE_EPOCHS = 5       # FinBERT single-task
MULTITASK_EPOCHS= 8       # Multi-task cần nhiều epoch hơn để cân bằng hai task
FEW_SHOT_K      = 16      # 16 ví dụ mỗi lớp (Part C yêu cầu k ∈ {4,8,16,32})
WARMUP_RATIO    = 0.1     # 10% bước đầu warmup tuyến tính
SEED            = 42      # Reproducibility
TEST_SIZE       = 0.2
VAL_SIZE        = 0.1
```

Mỗi giá trị được dùng bởi nhiều file; ai thay đổi `config.py` sẽ đồng thời ảnh hưởng toàn bộ pipeline.

### 3.4 Hằng Số Đặc Thù Của BERT LLRD (trong `src/finetune_bert.py`)

```python
LLRD_BASE_LR    = 2e-5
LLRD_DECAY      = 0.9
LABEL_SMOOTHING = 0.1
BERT_EPOCHS     = 10      # Cần ≥ 10 để unfreeze đủ layer groups
NUM_BERT_LAYERS = 12      # bert-base-uncased
```

---

## 4. Kiến Trúc Dự Án

### 4.1 Cây File

```
Project/
├── config.py                    # Siêu tham số + đường dẫn + device
├── run_experiments.py           # Điểm vào chính (6 bước)
├── cli.py                       # CLI inference
├── demo.py                      # Gradio demo (port 7860)
├── data_analysis.py             # Sinh 10 biểu đồ phân tích
├── push_to_hf.py                # Đẩy 5 repo lên HuggingFace Hub
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Tải FOMC + FPB, stratified split, few-shot, class weights
│   ├── baseline.py              # TF-IDF + LR, TF-IDF + SVM, TF-IDF (1-3 grams) + LR
│   ├── lexicon.py               # 5 word lists LM + 8 đặc trưng + rule-based + hybrid
│   ├── pretrained_eval.py       # Zero-shot FinBERT + 16-shot linear probe
│   ├── finetune_fineBert.py     # Fine-tune FinBERT đơn nhiệm
│   ├── finetune_bert.py         # BERT-base + LLRD + Gradual Unfreezing
│   ├── multitask.py             # MultiTaskFinBERT + alternating training
│   └── evaluate.py              # compute_metrics, confusion matrix, error analysis
├── data/                        # (cache local, không commit)
├── models/                      # Checkpoints các mô hình đã huấn luyện
├── results/                     # JSON kết quả + PNG confusion matrix
├── analysis/                    # 10 plot phân tích
└── presentation/                # Slide Beamer
```

### 4.2 Luồng Dữ Liệu 6 Bước

1. **Load**: `step1_load_data()` tải FPB và FOMC → `DatasetDict{train, val, test}`.
2. **Non-neural baselines + Lexicon**: `step2_baseline` chạy TF-IDF+LR, TF-IDF+SVM, TF-IDF(1-3grams)+LR; `step2b_lexicon` chạy rule-based và hybrid.
3. **Pretrained**: `step3_pretrained` chạy zero-shot FinBERT native + 16-shot linear probe trên FinBERT, BERT-base, RoBERTa-base.
4. **Single-task fine-tuning**: `step4_finetune` fine-tune FinBERT riêng cho stance (weighted loss) và sentiment (plain loss).
5. **Multi-task**: `step5_multitask` huấn luyện `MultiTaskFinBERT` trên cả hai task với alternating batches.
6. **BERT LLRD**: `step6_finetune_bert_llrd` fine-tune BERT-base với LLRD + Gradual Unfreezing + label smoothing.

Cuối mỗi bước, metrics được ghi vào `results/*.json`; sau bước cuối, `run_experiments.main()` gộp tất cả thành `results/all_results_summary.json`.

---

## 5. Chi Tiết Bộ Dữ Liệu

### 5.1 FOMC Hawkish-Dovish

- Nguồn: `gtfintechlab/fomc_communication` trên HuggingFace.
- Nội dung: câu trích từ các FOMC statements, minutes, và speeches, được gán nhãn 3 lớp: `dovish (0)`, `hawkish (1)`, `neutral (2)`.
- Kích thước: **2,480 câu**.
- Split (stratified, `random_state=42`):

| Split | Tổng | dovish | hawkish | neutral |
|-------|-----:|-------:|--------:|--------:|
| train | 1,736 | 455 | 424 | 857 |
| val   | 248  | 65  | 61  | 122 |
| test  | 496  | 130 | 121 | 245 |

Lưu ý: lớp `neutral` chiếm gần một nửa train set → cần weighted loss.

### 5.2 Financial PhraseBank (allagree)

- Nguồn: `gtfintechlab/financial_phrasebank_sentences_allagree`, subset `5768`.
- Nội dung: câu tin tài chính được gán nhãn 3 lớp: `negative (0)`, `neutral (1)`, `positive (2)`.
- FPB có nhiều subset theo mức độ đồng thuận (50%, 66%, 75%, 100%). Project dùng **allagree** — chỉ giữ các câu mà 100% annotator đồng ý. Đánh đổi: dữ liệu sạch hơn nhưng nhỏ hơn, và nhãn gần ranh giới (như một số trường hợp tranh cãi) bị loại.
- Kích thước: **2,264 câu**.
- Split:

| Split | Tổng | negative | neutral | positive |
|-------|-----:|---------:|--------:|---------:|
| train | 1,584 | 212 | 973 | 399 |
| val   | 227  | 30  | 140 | 57  |
| test  | 453  | 61  | 278 | 114 |

Lớp `neutral` cũng áp đảo ở đây, nhưng hầu hết mô hình mạnh vẫn đạt accuracy > 0.97 → ta không cần weighted loss cho sentiment.

### 5.3 Chiến Lược Split

Từ `src/data_loader.py`:

```python
# Stratified split: train 70% / val 10% / test 20%
train_df, test_df = train_test_split(
    df, test_size=TEST_SIZE, stratify=df["label"], random_state=SEED,
)
train_df, val_df = train_test_split(
    train_df,
    test_size=VAL_SIZE / (1 - TEST_SIZE),   # = 0.125 của phần còn lại
    stratify=train_df["label"], random_state=SEED,
)
```

Stratified giữ tỉ lệ lớp của từng split gần giống tỉ lệ gốc. `random_state=42` đảm bảo tái hiện được y nguyên giữa các lần chạy.

---

## 6. Từ Điển Loughran-McDonald

`src/lexicon.py` triển khai baseline lexicon-based theo hai cách: rule-based thuần và hybrid (lexicon + TF-IDF).

### 6.1 Năm Word Lists

1. **LM_POSITIVE** (~110 từ) — `achieve, benefit, gain, growth, profit, robust, surge, ...`
2. **LM_NEGATIVE** (~230 từ) — `bankruptcy, decline, loss, risk, slump, weakness, ...`
3. **LM_UNCERTAINTY** (~90 từ) — `approximate, contingent, likelihood, may, perhaps, ...`
4. **HAWKISH_WORDS** (~40 từ) — `hike, raise, tighten, inflation, restrictive, taper, ...`
5. **DOVISH_WORDS** (~50 từ) — `cut, ease, accommodate, stimulus, patient, subdued, ...`

LM_POSITIVE/NEGATIVE/UNCERTAINTY là tập con đã chọn lọc từ Master Dictionary của Loughran & McDonald (2011, cập nhật 2023); HAWKISH_WORDS và DOVISH_WORDS là mở rộng tự biên cho task stance, lắp ráp từ tài liệu central-banking.

### 6.2 Tám Đặc Trưng Trích Xuất

Hàm `extract_lexicon_features(texts)` trả về `np.ndarray` kích thước `(n_texts, 8)`:

| # | Đặc trưng | Ý nghĩa |
|---|-----------|---------|
| 0 | `positive_count` | Số từ positive |
| 1 | `negative_count` | Số từ negative |
| 2 | `uncertainty_count` | Số từ uncertainty |
| 3 | `hawkish_count` | Số từ hawkish |
| 4 | `dovish_count` | Số từ dovish |
| 5 | `net_sentiment` | `(positive − negative) / total_words` |
| 6 | `net_stance` | `(hawkish − dovish) / total_words` |
| 7 | `total_words` | Số token của câu |

Tokenizer đơn giản: `re.findall(r'\b[a-z]+\b', text.lower())`.

### 6.3 Rule-based Classifier (`lexicon_rule_based`)

Logic đơn thuần đếm từ:

```python
# Sentiment
if net_sentiment > 0.02:     return positive
elif net_sentiment < -0.02:  return negative
else:                         return neutral

# Stance
if hawkish_count > dovish_count:   return hawkish
elif dovish_count > hawkish_count: return dovish
else:                               return neutral
```

Không huấn luyện; chạy thẳng trên test set. Kết quả: **sentiment 0.6932 acc / 0.5315 macro-F1**, **stance 0.4153 acc / 0.3885 macro-F1**. Đây là floor có ý nghĩa — bất kỳ mô hình học máy nào cũng phải vượt qua.

### 6.4 Hybrid (`lexicon_plus_tfidf`)

Ghép TF-IDF (1-2 grams, 50k vocab, sublinear) với 8 đặc trưng lexicon đã chuẩn hoá qua `StandardScaler`, sau đó đưa vào `LogisticRegression(class_weight="balanced")`. Ý tưởng: TF-IDF bắt toàn bộ từ vựng, lexicon cung cấp kiến thức miền dạng structured. Kết quả: **sentiment 0.8543 acc / 0.8050 macro-F1**, **stance 0.6109 acc / 0.5863 macro-F1** — vượt xa rule-based nhưng trên sentiment vẫn kém TF-IDF thuần vì 8 đặc trưng thêm vào pha loãng tín hiệu.

---

## 7. Mô Hình Cơ Sở TF-IDF

`src/baseline.py` cung cấp ba pipeline:

### 7.1 TF-IDF + Logistic Regression (Baseline Gốc)

```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 2),
        sublinear_tf=True, strip_accents="unicode",
    )),
    ("clf", LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=SEED, solver="lbfgs",
    )),
])
```

### 7.2 TF-IDF + LinearSVC

```python
LinearSVC(max_iter=2000, class_weight="balanced", random_state=SEED, C=1.0)
```

SVM biên rộng thường vượt LR trên văn bản thưa. Project dùng `C=1.0` (regularisation mặc định) và `class_weight="balanced"`.

### 7.3 TF-IDF (1-3 grams) + LR

Bigram hữu ích (bigram `interest rate`), trigram có thể bắt được cụm chính sách (`raise interest rates`). Bản này dùng `ngram_range=(1, 3)`, `max_features=80_000`, `min_df=2`.

### 7.4 Bảng Kết Quả

| Mô hình | Sent. Acc | Sent. F1 | Stance Acc | Stance F1 |
|---------|----------:|---------:|-----------:|----------:|
| TF-IDF + LR                     | 0.8720 | 0.8232 | 0.6089 | 0.5873 |
| TF-IDF + SVM (LinearSVC, C=1)   | 0.8940 | 0.8534 | 0.6331 | 0.6061 |
| TF-IDF (1-3 gram) + LR          | 0.8786 | 0.8310 | 0.6109 | 0.5914 |
| TF-IDF + LM Lexicon (LR)        | 0.8543 | 0.8050 | 0.6109 | 0.5863 |

Quan sát:
- SVM thắng LR trên cả hai task (~2 pp).
- Thêm trigram giúp LR nhẹ nhưng không nhiều, do dữ liệu nhỏ nên trigram hiếm.
- Tất cả huấn luyện dưới 5 giây.

---

## 8. Đánh Giá Transformer Huấn Luyện Sẵn

`src/pretrained_eval.py` chia thành hai phần: zero-shot và few-shot.

### 8.1 Zero-shot FinBERT (Native Head)

FinBERT có head gốc 3 lớp `positive / negative / neutral`. Với sentiment, mapping trực tiếp:

```python
finbert_to_idx = {"negative": 0, "neutral": 1, "positive": 2}
```

Với stance ta **không có head hawkish/dovish**, nên dùng proxy từ sentiment:

```python
# negative → dovish, positive → hawkish, neutral → neutral
finbert_to_idx = {"negative": 0, "positive": 1, "neutral": 2}
```

Trực giác: ngôn ngữ tiêu cực về kinh tế thường đi đôi với tín hiệu giảm lãi suất (dovish), ngôn ngữ tích cực/lạm phát cao thường đi đôi với tăng lãi suất (hawkish). Đây là baseline cố ý yếu để chứng minh `head sentiment không đủ thay cho stance`.

Kết quả:
- Sentiment: **0.9735 acc / 0.9650 macro-F1** (gần như giải xong task!)
- Stance proxy: **0.4980 acc / 0.4874 macro-F1** (chỉ hơi tốt hơn random 3-class).

### 8.2 Few-shot Linear Probe (k=16)

Từ `evaluate_few_shot`:

1. Load `AutoModel` (không head) + `AutoTokenizer`.
2. Lấy 16 ví dụ mỗi lớp → tổng 48 câu training.
3. Encoder đóng băng; với mỗi câu lấy vector `[CLS]` (chiều 768).
4. Huấn luyện `FewShotClassifier`:

```python
nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(hidden_size, num_labels),
)
```

với Adam lr=1e-3, 200 epochs, plain CE loss trên 48 embedding.

Kết quả k=16:

| Model | Sent. Acc | Sent. F1 | Stance Acc | Stance F1 |
|-------|----------:|---------:|-----------:|----------:|
| FinBERT few-shot      | 0.9779 | 0.9670 | 0.4859 | 0.4534 |
| BERT-base few-shot    | 0.7417 | 0.6500 | 0.3851 | 0.3744 |
| RoBERTa-base few-shot | 0.7682 | 0.6722 | 0.3730 | 0.3600 |

Phát hiện: **FinBERT vượt BERT/RoBERTa ~30 pp trên sentiment** — pre-training theo miền rất giá trị khi dữ liệu ít. Stance thì cả ba đều yếu, vì stance cần kiến thức chính sách tiền tệ mà sentiment pre-training không bắt được.

---

## 9. Tinh Chỉnh FinBERT và BERT-base

Có hai file fine-tune độc lập, đại diện hai triết lý khác nhau.

### 9.1 Overview `src/finetune_fineBert.py`

FinBERT được fine-tune theo cách "chuẩn" (vanilla):
- Lấy `AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL, num_labels=3, ignore_mismatched_sizes=True)` — tham số `ignore_mismatched_sizes=True` thay head sentiment gốc bằng head mới 3 lớp (có thể trùng size nhưng weight reset).
- Toàn bộ layer đều train từ epoch đầu.
- Optimizer: `AdamW(lr=2e-5, weight_decay=0.01)`.
- Scheduler: linear warmup 10% rồi linear decay đến 0.
- Loss: weighted CE cho stance, plain CE cho sentiment.
- Epochs: 5.
- Gradient clipping: `max_norm=1.0`.
- Lưu checkpoint có macro-F1 val cao nhất → test evaluation.

### 9.2 TextClassificationDataset

Class PyTorch `Dataset` wrap một list text + list label, tokenize on-the-fly với `padding="max_length"`, `truncation=True`, `max_length=128`, trả về `{input_ids, attention_mask, labels}`.

### 9.3 Vòng Huấn Luyện

```python
for epoch in range(FINETUNE_EPOCHS):
    model.train()
    for batch in train_loader:
        outputs = model(input_ids=..., attention_mask=...)
        loss = criterion(outputs.logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    val_metrics = _evaluate_model(model, val_loader, label_names, device)
    if val_metrics["macro_f1"] > best_val_f1:
        best_model_state = clone(model.state_dict())
```

### 9.4 Class Weights Thực Tế

`compute_class_weights(fomc["train"])` trả về (đúng như trong ground truth):
- `dovish: 1.272, hawkish: 1.365, neutral: 0.675`

Lớp thiểu số (hawkish, dovish) được nhân lớn hơn; lớp đông (neutral) bị giảm.

### 9.5 Lưu Mô Hình

```python
save_path = os.path.join(MODELS_DIR, f"finbert_{task_name}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
```

### 9.6 Kết Quả FinBERT Fine-tune

| Task | Acc | Macro-F1 |
|------|----:|---------:|
| Sentiment | 0.9669 | 0.9459 |
| Stance    | 0.6371 | 0.6194 |

Sentiment cao hơn rule-based/TF-IDF nhiều; stance tăng so với baseline.

### 9.7 BERT-base với LLRD + Gradual Unfreezing (`src/finetune_bert.py`)

#### 9.7.1 Khác biệt so với FinBERT branch

| Khía cạnh | FinBERT branch | BERT LLRD branch |
|-----------|----------------|------------------|
| Base model | `ProsusAI/finbert` | `bert-base-uncased` |
| Fine-tune | Toàn bộ từ epoch 1 | Gradual unfreezing |
| Per-layer LR | Uniform 2e-5 | Decay 0.9 mỗi depth |
| Loss | CE (weighted cho stance) | Label-smoothed CE (ε=0.1), weighted cho stance |
| Warmup | Linear 10% | Ngầm qua unfreezing |
| Epochs | 5 | 10 |

#### 9.7.2 Layer Groups (`_build_layer_groups`)

Trả về list 14 group từ top xuống bottom:
- Group 0: `head` (classifier + pooler) — LR = `2e-5`
- Group 1: `encoder_layer_11` — LR = `2e-5 × 0.9`
- Group 2: `encoder_layer_10` — LR = `2e-5 × 0.9²`
- ...
- Group 12: `encoder_layer_0` — LR = `2e-5 × 0.9¹²`
- Group 13: `embeddings` — LR = `2e-5 × 0.9¹³`

#### 9.7.3 Lịch Unfreeze (`_build_optimizer`)

```
epoch 1  → chỉ head                       (1 group active)
epoch 2  → head + layer 11                (2 groups)
epoch 3  → head + layers 11, 10           (3 groups)
...
epoch 10 → 10 group hoạt động
```

Mỗi epoch xây lại optimizer: đóng băng toàn bộ param, sau đó unfreeze đúng `n_active = min(epoch, 14)` group đầu. Tham số `weight_decay = 0.0` cho head (tránh co logit) và `0.01` cho các group khác.

#### 9.7.4 Loss

Stance: CE weighted + label smoothing ε=0.1.
Sentiment: CE label smoothing ε=0.1 (không weighted vì FPB cân bằng hơn).

#### 9.7.5 Kết Quả BERT LLRD

| Task | Acc | Macro-F1 |
|------|----:|---------:|
| Sentiment | 0.9757 | 0.9670 |
| Stance    | 0.6512 | 0.6383 |

**Phát hiện đáng chú ý:** BERT-base LLRD đạt macro-F1 sentiment = 0.9670 — **bằng FinBERT few-shot và multi-task**; stance 0.6383 — **gần bằng multi-task (0.6384)**. Dùng công thức fine-tune cẩn thận (LLRD + gradual unfreezing + label smoothing) trên một BERT đa dụng bù được cho thiếu pre-training theo miền.

---

## 10. Học Đa Nhiệm

`src/multitask.py` cài đặt `MultiTaskFinBERT`.

### 10.1 Kiến Trúc

```python
class MultiTaskFinBERT(nn.Module):
    def __init__(self, num_stance_labels=3, num_sentiment_labels=3, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(FINBERT_MODEL)   # 110M params shared
        hidden_size = self.encoder.config.hidden_size              # 768
        self.dropout = nn.Dropout(dropout)
        self.stance_head    = nn.Linear(hidden_size, num_stance_labels)     # 768×3
        self.sentiment_head = nn.Linear(hidden_size, num_sentiment_labels)  # 768×3

    def forward(self, input_ids, attention_mask, task="stance"):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = outputs.last_hidden_state[:, 0, :]   # [CLS]
        pooled  = self.dropout(pooled)
        if task == "stance":
            return self.stance_head(pooled)
        else:
            return self.sentiment_head(pooled)
```

### 10.2 Alternating Batches

Mỗi epoch duyệt đồng thời `stance_train_loader` và `sentiment_train_loader` với hai `iter`. Trong vòng lặp `while not (stance_done and sentiment_done)`: lấy một batch stance → backprop trên `stance_head`, rồi một batch sentiment → backprop trên `sentiment_head`. Optimizer và scheduler **chung**, cập nhật toàn bộ encoder + cả hai head. Encoder học gradient hỗn hợp từ hai task.

### 10.3 Hàm Loss

- Stance: `CrossEntropyLoss(weight=[1.272, 1.365, 0.675])`
- Sentiment: `CrossEntropyLoss()` plain

Không có trọng số giữa hai loss (lambda) — mỗi batch đóng góp trọng số như nhau vì được xử lý tuần tự.

### 10.4 Optimizer / Scheduler

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
steps_per_epoch = len(stance_train_loader) + len(sentiment_train_loader)
total_steps = steps_per_epoch * MULTITASK_EPOCHS   # epochs=8
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps,
)
```

### 10.5 Selection Criterion

Mỗi epoch tính `avg_f1 = (val_stance_f1 + val_sentiment_f1) / 2`. Checkpoint có `avg_f1` cao nhất được giữ.

### 10.6 Kết Quả Multi-task FinBERT

| Task | Acc | Macro-F1 |
|------|----:|---------:|
| Sentiment | 0.9779 | 0.9666 |
| Stance    | 0.6492 | 0.6384 |

### 10.7 F1 Theo Lớp

**Stance (FOMC test):**

| Lớp | F1 |
|-----|---:|
| dovish  | 0.6174 |
| hawkish | 0.5869 |
| neutral | 0.7109 |

**Sentiment (FPB test):**

| Lớp | F1 |
|-----|---:|
| negative | 0.9500 |
| neutral  | 0.9928 |
| positive | 0.9569 |

Phân tích lỗi:
- Với stance, lớp `neutral` F1 cao nhất (0.7109) vì nó là đa số (245/496 ≈ 49%). `hawkish` F1 thấp nhất (0.5869) — FOMC nói về lạm phát nhưng không luôn đồng nghĩa tăng lãi suất ngay.
- Với sentiment, `neutral` gần như perfect (F1 = 0.9928) vì đây là lớp đa số và ngôn ngữ khách quan có mẫu đặc trưng rõ.

### 10.8 Positive Transfer

So với single-task FinBERT:
- Stance F1: 0.6194 → 0.6384 = **+1.90 pp**
- Sentiment F1: 0.9459 → 0.9666 = **+2.07 pp**

Học đa nhiệm cải thiện **cả hai** task — positive transfer cho cả hai hướng, dù stance hưởng lợi nhiều hơn một chút (về tuyệt đối).

---

## 11. Phương Pháp Đánh Giá

`src/evaluate.py` cung cấp:

### 11.1 `compute_metrics(y_true, y_pred, label_names)`

Trả về dict:
```python
{
  "accuracy":     round(accuracy_score, 4),
  "macro_f1":     round(f1_score(..., average="macro"), 4),
  "per_class_f1": {label: round(f1, 4) for label, f1 in ...},
  "report":       classification_report(...),   # chuỗi nhiều dòng
}
```

### 11.2 Macro-F1 vs Accuracy

- **Accuracy** = số dự đoán đúng / tổng. Dễ hiểu nhưng thiên vị theo lớp đa số.
- **Macro-F1** = trung bình F1 từng lớp. Mỗi lớp đóng góp như nhau, không phụ thuộc kích thước → chỉ số công bằng hơn cho dữ liệu imbalanced.

Project báo **macro-F1 là chỉ số chính** và accuracy chỉ là tham khảo.

### 11.3 Confusion Matrix

`plot_confusion_matrix` vẽ heatmap 3×3 với seaborn, lưu file `.png`. Đường chéo = dự đoán đúng; ô off-diagonal chỉ ra mô hình nhầm từ nhãn A sang nhãn B.

### 11.4 Error Analysis

`error_analysis` lọc các dự đoán sai, nhóm theo `(true → pred)` và lấy top 20 (sắp theo độ dài câu giảm dần — câu dài thường khó hơn). Output:
```
Misclassification breakdown (178 total errors):
  neutral → hawkish: 43
  hawkish → neutral: 38
  dovish → neutral: 34
  ...
```

---

## 12. CLI, Demo, và HuggingFace

### 12.1 CLI (`cli.py`)

Ba cách gọi:
```bash
python cli.py                                              # interactive
python cli.py --text "The Fed signaled further rate hikes"
python cli.py --file input.txt
python cli.py --model finetune --text "..."               # thay vì multitask
```

Flag `--model` nhận `{multitask, finetune}`. `load_multitask_model()` load checkpoint `models/multitask_finbert/model.pt`; `load_finetune_models()` load hai thư mục `models/finbert_stance/` và `models/finbert_sentiment/`. Đầu ra có label + confidence + histogram ASCII.

### 12.2 Demo Gradio (`demo.py`)

```bash
python demo.py           # mở http://localhost:7860
```

Load `MultiTaskFinBERT` từ `models/multitask_finbert/model.pt`. Giao diện: một ô textbox → hai `gr.Label` hiển thị stance và sentiment với confidence các lớp.

### 12.3 Data Analysis (`data_analysis.py`)

`main()` gọi 10 plot:
1. `plot_class_distributions` — barplot 3 lớp cho cả FPB và FOMC.
2. `plot_text_lengths` — histogram độ dài câu theo token.
3. `plot_top_words` — top-N từ thường gặp mỗi lớp (sau stopword filter).
4. `plot_model_comparison` — bar chart so sánh accuracy/F1 giữa các mô hình.
5. `plot_per_class_f1_heatmap` — heatmap F1 theo lớp × mô hình.
6. `plot_progression` — đường thể hiện F1 tăng qua các bước (rule → TF-IDF → FinBERT → multi-task).
7. `plot_domain_pretraining_gap` — so sánh FinBERT vs BERT vs RoBERTa ở few-shot.
8. `plot_multitask_improvement` — delta F1 giữa multi-task và single-task.
9. `plot_task_difficulty` — so sánh độ khó stance vs sentiment.
10. `plot_lexicon_coverage` — tỉ lệ câu chứa ít nhất 1 từ LM dictionary.

Các file PNG được lưu vào `analysis/`.

### 12.4 Push Lên HuggingFace (`push_to_hf.py`)

`main()` đẩy 5 repo:
- `finbert-stance` — FinBERT fine-tune stance
- `finbert-sentiment` — FinBERT fine-tune sentiment
- `bert-llrd-stance` — BERT-base LLRD stance
- `bert-llrd-sentiment` — BERT-base LLRD sentiment
- `multitask-finbert` — Multi-task FinBERT

Hai hàm helper:
- `push_hf_format_model(local_name, repo_name)` cho các model `AutoModelForSequenceClassification`.
- `push_multitask_model(local_name, repo_name)` cho multi-task (state_dict + config custom).

Cần `HF_TOKEN` trong biến môi trường.

---

## 13. Các Lỗi Gặp Phải

1. **`datasets 4.x` bỏ loading scripts** — phiên bản cũ của FOMC dataset dùng file `.py` tải manual; API mới yêu cầu Parquet. Ta phải dùng cho FPB subset `"5768"` (config name bắt buộc) của bản Parquet `gtfintechlab/financial_phrasebank_sentences_allagree`.

2. **`sklearn 1.8` bỏ tham số `multi_class`** — LogisticRegression cũ nhận `multi_class="multinomial"`. Phiên bản mới tự suy luận, nên ta bỏ param này.

3. **`pandas 3.0` thay đổi `groupby.apply`** — cột groupby bị drop khỏi kết quả; code cũ dựa vào cột đó sẽ hỏng. Ta phải gọi `reset_index()` để lấy lại tường minh.

4. **MPS pipeline không ổn định** — `transformers.pipeline(..., device="mps")` gây runtime error cho FinBERT native. Giải pháp: ép `device=-1` (CPU) cho zero-shot pipeline, chỉ dùng MPS cho custom loop.

5. **Python 3.14 chưa có wheel PyTorch ổn định** — project buộc phải dùng Python 3.12.

---

## 14. Phân Tích Kết Quả

### 14.1 Bảng Tổng Hợp (Từ `results/all_results_summary.json`)

| Mô hình                          | Sent. Acc | Sent. F1 | Stance Acc | Stance F1 |
|----------------------------------|----------:|---------:|-----------:|----------:|
| LM Lexicon (rule-based)          | 0.6932    | 0.5315   | 0.4153     | 0.3885    |
| TF-IDF + LR                      | 0.8720    | 0.8232   | 0.6089     | 0.5873    |
| TF-IDF + SVM (LinearSVC, C=1)    | 0.8940    | 0.8534   | 0.6331     | 0.6061    |
| TF-IDF (1-3 gram) + LR           | 0.8786    | 0.8310   | 0.6109     | 0.5914    |
| TF-IDF + LM Lexicon (LR)         | 0.8543    | 0.8050   | 0.6109     | 0.5863    |
| FinBERT zero-shot (native head)  | 0.9735    | 0.9650   | 0.4980     | 0.4874    |
| FinBERT few-shot (k=16 probe)    | 0.9779    | 0.9670   | 0.4859     | 0.4534    |
| BERT-base few-shot (k=16)        | 0.7417    | 0.6500   | 0.3851     | 0.3744    |
| RoBERTa-base few-shot (k=16)     | 0.7682    | 0.6722   | 0.3730     | 0.3600    |
| FinBERT (fine-tuned)             | 0.9669    | 0.9459   | 0.6371     | 0.6194    |
| BERT-base LLRD + Gradual UF      | 0.9757    | 0.9670   | 0.6512     | 0.6383    |
| Multi-task FinBERT               | 0.9779    | 0.9666   | 0.6492     | 0.6384    |

### 14.2 Sáu Phát Hiện Chính

1. **BERT-base LLRD ngang ngửa FinBERT và Multi-task FinBERT.** Sentiment macro-F1 hoà ở **0.9670** giữa LLRD và FinBERT few-shot, so với **0.9666** của multi-task. Stance macro-F1 hoà ở **0.6383** (LLRD) vs **0.6384** (multi-task). Công thức tinh chỉnh cẩn thận (LLRD + gradual unfreezing + label smoothing ε=0.1) trên một BERT đa dụng bù được cho thiếu pre-training theo miền như FinBERT.

2. **Học đa nhiệm giúp stance nhiều hơn sentiment.** So với single-task FinBERT, đa nhiệm tăng **+1.90 pp** stance macro-F1 và **+2.07 pp** sentiment macro-F1. Cả hai task đều hưởng lợi → positive transfer hai chiều.

3. **Stance khó hơn sentiment nhiều.** Stance macro-F1 tốt nhất là **0.6384** (~35% error rate), trong khi sentiment là **0.9670** (~3% error rate). Stance yêu cầu suy luận về chính sách tương lai dựa trên ngôn ngữ mơ hồ của FOMC.

4. **Pre-training theo miền có giá trị lớn nhất ở low-data.** FinBERT few-shot (k=16) đánh bại BERT/RoBERTa few-shot **~32 pp** trên sentiment. Khi fine-tune đầy đủ, khoảng cách giữa BERT LLRD và FinBERT thu hẹp còn **< 1 pp**. Bài học: domain pre-training quý khi dữ liệu ít, không thay thế được fine-tune dài ngày khi có dữ liệu.

5. **Zero-shot FinBERT đã giải được sentiment.** **0.9735 accuracy / 0.9650 macro-F1** trên Financial PhraseBank mà KHÔNG cần huấn luyện — head sentiment gốc của FinBERT đã bao trọn task này. Chi phí bổ sung cho mọi fine-tune chỉ mua được ~0.2 pp.

6. **TF-IDF baseline mạnh cho sentiment.** TF-IDF+SVM đạt **0.8940 accuracy** chỉ với **<5 giây training**. Cho hệ thống production không đòi hỏi top-tier, đây vẫn là lựa chọn hấp dẫn.

### 14.3 F1 Theo Lớp Cho Hai Mô Hình Tốt Nhất

**Multi-task FinBERT:**
- Stance: dovish **0.6174**, hawkish **0.5869**, neutral **0.7109**.
- Sentiment: negative **0.9500**, neutral **0.9928**, positive **0.9569**.

**BERT-base LLRD:**
- Stance: dovish **0.5831**, hawkish **0.6084**, neutral **0.7235**.
- Sentiment: negative **0.9600**, neutral **0.9891**, positive **0.9520**.

Nhận xét chung:
- Cả hai đều chia sẻ pattern: `neutral` F1 cao nhất ở stance, `neutral` gần perfect ở sentiment.
- Multi-task nhỉnh hơn BERT LLRD ở `dovish` stance (0.6174 vs 0.5831) → domain pre-training của FinBERT bắt được ngôn ngữ "easing" tốt hơn BERT.
- BERT LLRD nhỉnh hơn ở `hawkish` (0.6084 vs 0.5869) và `negative` sentiment (0.9600 vs 0.9500) → label smoothing giúp tránh over-confidence trên lớp thiểu số.

---

## 15. Bài Học Rút Ra

### Cho Người Mới Làm NLP

1. **Luôn bắt đầu từ baseline đơn giản.** TF-IDF + LR/SVM là đường biên cơ sở — nếu mô hình lớn không vượt qua đáng kể, có lỗi ở đâu đó.
2. **Macro-F1, không phải accuracy.** Dữ liệu thực hầu như luôn imbalanced.
3. **Stratified split + random_state cố định.** Bắt buộc để tái hiện.
4. **Weighted loss khi lớp lệch.** Không phải ma thuật, nhưng giúp đáng kể với lớp thiểu số.
5. **Label smoothing là biện pháp regularisation rẻ tiền và hiệu quả** trên tập huấn luyện nhỏ.
6. **LLRD + Gradual Unfreezing** biến `bert-base-uncased` thành đối thủ của mô hình pre-train theo miền — đáng thử khi không có FinBERT tương đương.
7. **Linear probe với k=16** là chuẩn vàng để **so sánh chất lượng encoder** không lẫn các yếu tố fine-tune.
8. **Zero-shot là nút kiểm tra hợp lý**: nếu mô hình pre-train đã có sẵn head phù hợp, phần gia tăng của fine-tune có thể nhỏ.

### Hướng Tương Lai

- **Ensemble** multi-task + BERT LLRD: hai mô hình có pattern lỗi khác nhau, có thể bù nhau.
- **Data augmentation** cho lớp hawkish (thiểu số nhất, F1 thấp nhất): back-translation, paraphrase bằng LLM.
- **Sequence-level context**: nối 2-3 câu kề nhau, có thể giúp stance vì FOMC thường diễn giải nhiều câu.
- **Thử FinBERT-tone** hoặc **FinLLaMA** (model tài chính mới hơn) làm encoder.
- **Task-specific auxiliary pre-training**: MLM thêm vài epoch trên FOMC trước khi fine-tune.

---

## 16. Tham Chiếu Code Đầy Đủ

Phần này đi qua từng file source, liệt kê hàm/class chính kèm giải thích ngắn.

### 16.1 `config.py`

File tập trung mọi hằng số. Không chứa hàm. Các khối:
- `PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR` — đường dẫn.
- `DEVICE` — auto-detect MPS/CUDA/CPU.
- `SEED=42`.
- `FPB_DATASET_NAME = "gtfintechlab/financial_phrasebank_sentences_allagree"`, `FPB_SUBSET = "5768"`.
- `FOMC_DATASET_NAME = "gtfintechlab/fomc_communication"`.
- `SENTIMENT_LABELS`, `STANCE_LABELS` + các dict `ID2LABEL / LABEL2ID`.
- `FINBERT_MODEL, BERT_BASE_MODEL, ROBERTA_BASE_MODEL`.
- Siêu tham số train: `MAX_SEQ_LENGTH, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, FINETUNE_EPOCHS, MULTITASK_EPOCHS, FEW_SHOT_K, WARMUP_RATIO`.
- Split sizes: `TEST_SIZE=0.2, VAL_SIZE=0.1`.

### 16.2 `src/data_loader.py`

- `load_financial_phrasebank()` — Tải FPB allagree, gộp các split, rename `sentence → text`, stratified split 70/10/20, in stats, trả về `DatasetDict`.
- `load_fomc_dataset()` — Giống trên cho FOMC; có fallback `_load_fomc_local()` đọc CSV nếu HuggingFace fail.
- `_load_fomc_local()` — Đọc `data/fomc_hawkish_dovish.csv`.
- `_process_fomc_df(df)` — Chuẩn hoá cột, map nhãn string → int nếu cần, stratified split.
- `get_few_shot_subset(dataset_split, k=FEW_SHOT_K)` — Sample `k` ví dụ mỗi lớp (seeded), trả về `Dataset`.
- `compute_class_weights(dataset_split, num_classes=3)` — Công thức inverse-frequency: `wₖ = N / (K·countₖ)`.
- `_print_split_stats(name, splits, label_names)` — In bảng kích thước + phân bố từng split.

### 16.3 `src/baseline.py`

- `build_baseline_pipeline()` — Pipeline TF-IDF (1-2 gram, max 50k) + LR.
- `train_and_evaluate_baseline(train_split, test_split, label_names, task_name)` — Fit pipeline, predict test, gọi `compute_metrics`, `plot_confusion_matrix`, `save_results`.
- `build_tfidf_svm_pipeline()` — Pipeline TF-IDF + LinearSVC(C=1, class_weight=balanced).
- `build_tfidf_trigram_lr_pipeline()` — TF-IDF (1-3 gram, max 80k, min_df=2) + LR.
- `_run_and_eval(pipeline, splits, label_names, task_name, model_name, file_prefix)` — Helper chung huấn luyện + đánh giá.
- `run_alternative_baselines(fomc_splits, fpb_splits)` — Chạy SVM và trigram LR trên cả hai task.

### 16.4 `src/lexicon.py`

- Hằng số: `LM_POSITIVE, LM_NEGATIVE, LM_UNCERTAINTY, HAWKISH_WORDS, DOVISH_WORDS`.
- `_tokenize(text)` — Regex `\b[a-z]+\b` sau `.lower()`.
- `extract_lexicon_features(texts)` — Trả về `(n_texts, 8)` với `[pos, neg, unc, hawk, dove, net_sent, net_stance, total_words]`.
- `lexicon_rule_based(test_split, label_names, task_name)` — Rule-based classifier; ngưỡng `net_sentiment > 0.02` (pos) / `< −0.02` (neg); stance dùng `hawk vs dove`.
- `lexicon_plus_tfidf(train_split, test_split, label_names, task_name)` — Hybrid: `hstack([TF-IDF, StandardScaler(lexicon_features)])` + `LogisticRegression(class_weight="balanced")`.
- `run_lexicon_experiments(fomc_splits, fpb_splits)` — Chạy cả hai biến thể trên cả hai task.

### 16.5 `src/pretrained_eval.py`

- `evaluate_finbert_native(test_split, task_name)` — Zero-shot: `pipeline("text-classification", FINBERT_MODEL, device=-1, top_k=None)`; với sentiment map trực tiếp `neg/neu/pos → 0/1/2`; với stance proxy `neg→dovish, pos→hawkish, neu→neutral`.
- `FewShotClassifier(nn.Module)` — `Sequential(Dropout(0.1), Linear(768, num_labels))`.
- `_encode_texts(tokenizer, model, texts, device)` — Encode batch bằng model đóng băng, trả về tensor `[CLS]` shape `(N, 768)`.
- `evaluate_few_shot(model_name, train_split, test_split, label_names, task_name, k=16)` — Few-shot linear probe: sample k/class, encode với encoder đóng băng, train linear head trên CPU với Adam lr=1e-3, 200 epochs, predict test.
- `run_all_pretrained_evaluations(fomc_splits, fpb_splits)` — Orchestrator chạy zero-shot + 3×2 few-shot.

### 16.6 `src/finetune_fineBert.py`

- `TextClassificationDataset(torch.utils.data.Dataset)` — wrap texts + labels; `__getitem__` trả `{input_ids, attention_mask, labels}`.
- `finetune_finbert(train_split, val_split, test_split, label_names, task_name, use_weighted_loss=True)` — Load FinBERT + head mới, build DataLoaders, AdamW + linear warmup, CE (optionally weighted), 5 epochs, track best val F1, test + save model.
- `_evaluate_model(model, dataloader, label_names, device)` — Gọi `_get_predictions` rồi `compute_metrics`.
- `_get_predictions(model, dataloader, device)` — Duyệt dataloader, trả `(y_true, y_pred)`.

### 16.7 `src/finetune_bert.py`

- Hằng số module: `LLRD_BASE_LR=2e-5, LLRD_DECAY=0.9, LABEL_SMOOTHING=0.1, BERT_EPOCHS=10, NUM_BERT_LAYERS=12`.
- `TextClassificationDataset` — identical với bản trong `finetune_fineBert.py` (tokenize on-the-fly, max_length=128).
- `_build_layer_groups(model)` — Tạo 14 group từ top (head+pooler) xuống bottom (embeddings), mỗi group có `{name, params, lr}`, LR decay 0.9 mỗi depth.
- `_build_optimizer(model, epoch)` — Freeze all → unfreeze `n_active=min(epoch, 14)` group đầu → AdamW với `param_groups` (weight_decay=0 cho head, 0.01 cho các group khác); trả `(optimizer, active_names)`.
- `_train_one_epoch(model, loader, criterion, optimizer, device)` — Training loop với gradient clipping `max_norm=1.0`.
- `_get_predictions(model, loader, device)` — Inference, trả `(y_true, y_pred)`.
- `finetune_bert_llrd(train_split, val_split, test_split, label_names, task_name)` — Entry point: load `bert-base-uncased`, loss = weighted label-smoothed CE (stance) hoặc label-smoothed CE (sentiment), 10 epochs với rebuild optimizer mỗi epoch, track best val F1, test + error_analysis + save.

### 16.8 `src/multitask.py`

- `MultiTaskFinBERT(nn.Module)` — Shared FinBERT encoder (110M params) + `nn.Dropout(0.1)` + `stance_head: Linear(768, 3)` + `sentiment_head: Linear(768, 3)`; `forward(..., task="stance"|"sentiment")` chọn head tương ứng.
- `train_multitask(fomc_splits, fpb_splits)` — Entry point: build 6 dataloader (train/val/test × 2 task), weighted CE cho stance, plain CE cho sentiment, AdamW + linear warmup 10%, 8 epochs alternating batches, track best avg_f1, test + confusion matrices + error_analysis, save `model.pt`.
- `_train_step(model, batch, criterion, optimizer, scheduler, device, task)` — One-batch step với gradient clip + scheduler step.
- `_evaluate_multitask(model, dataloader, label_names, device, task)` — Gọi `_get_multitask_predictions` + `compute_metrics`.
- `_get_multitask_predictions(model, dataloader, device, task)` — Inference trên một task.

### 16.9 `src/evaluate.py`

- `compute_metrics(y_true, y_pred, label_names)` — Trả dict `{accuracy, macro_f1, per_class_f1, report}`, round 4 chữ số.
- `print_classification_report(metrics, model_name, task_name)` — In đẹp ra console.
- `plot_confusion_matrix(y_true, y_pred, label_names, model_name, task_name, save_dir=None)` — seaborn heatmap, lưu PNG 150 dpi.
- `error_analysis(texts, y_true, y_pred, label_names, top_n=20)` — Lọc sai, đếm error-type, trả top-N theo độ dài.
- `save_results(results_dict, filename)` — Ghi JSON vào `RESULTS_DIR`.

### 16.10 `run_experiments.py`

- `step1_load_data()` — Tải FPB + FOMC.
- `step2_baseline(fomc, fpb)` — TF-IDF+LR + `run_alternative_baselines` (SVM, trigram LR).
- `step2b_lexicon(fomc, fpb)` — `run_lexicon_experiments`.
- `step3_pretrained(fomc, fpb)` — `run_all_pretrained_evaluations`.
- `step4_finetune(fomc, fpb)` — FinBERT stance (weighted) + sentiment (plain).
- `step5_multitask(fomc, fpb)` — `train_multitask`.
- `step6_finetune_bert_llrd(fomc, fpb)` — BERT-base LLRD trên cả hai task.
- `print_summary(all_results)` — In bảng tổng kết.
- `main()` — Parse `--step 0|1|2|3|4|5|6`, chạy tuần tự, gộp vào `results/all_results_summary.json`.

### 16.11 `cli.py`

- `load_multitask_model()` — Load `MultiTaskFinBERT` từ `models/multitask_finbert/model.pt`.
- `load_finetune_models()` — Load hai `AutoModelForSequenceClassification` từ `models/finbert_{stance, sentiment}/`.
- `predict_multitask(text, model, tokenizer)` — Gọi model 2 lần (task=stance, task=sentiment), softmax, trả dict label + confidence + all_scores.
- `predict_finetune(text, models)` — Gọi 2 model riêng cho 2 task.
- `format_prediction(results)` — In ASCII histogram với `█`.
- `main()` — Parse flags `--text, --file, --model`, chạy interactive / single / file mode.

### 16.12 `demo.py`

- `load_model()` — Load multi-task (fallback raise nếu không có checkpoint).
- `predict(text, model, tokenizer)` — Trả `(stance_scores, sentiment_scores)` cho `gr.Label`.
- `main()` — Build Gradio Blocks với 1 textbox + 2 Label outputs, launch trên port `7860`.

### 16.13 `data_analysis.py`

- `load_datasets()` — Tải FPB + FOMC.
- `dataset_statistics(fpb, fomc)` — Tính thống kê cơ bản (size, dist, length, top words).
- `plot_class_distributions(stats)` — Plot #1.
- `plot_text_lengths(stats)` — Plot #2.
- `plot_top_words(stats)` — Plot #3.
- `load_all_results()` — Đọc `results/all_results_summary.json`.
- `plot_model_comparison()` — Plot #4.
- `plot_per_class_f1_heatmap()` — Plot #5.
- `plot_progression()` — Plot #6.
- `plot_domain_pretraining_gap()` — Plot #7.
- `plot_multitask_improvement()` — Plot #8.
- `plot_task_difficulty()` — Plot #9.
- `plot_lexicon_coverage(stats)` — Plot #10.
- `main()` — Gọi cả 10 plot, lưu PNG vào `analysis/`.

### 16.14 `push_to_hf.py`

- `push_hf_format_model(local_name, repo_name)` — Đẩy các model `AutoModelForSequenceClassification` chuẩn (FinBERT stance/sentiment, BERT LLRD stance/sentiment).
- `push_multitask_model(local_name, repo_name)` — Đẩy multi-task model (custom state_dict + config).
- `main()` — Đẩy 5 repo tổng.

---

## Lời Kết

Tài liệu này đã trình bày đầy đủ mọi khía cạnh kỹ thuật của đồ án: từ dataset đến lexicon, từ TF-IDF đến multi-task FinBERT, từ zero-shot đến LLRD. Nếu bạn muốn tái hiện, hãy làm theo thứ tự:

```bash
git clone <repo>
cd Project
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_experiments.py               # chạy cả 6 bước
python data_analysis.py                 # sinh 10 biểu đồ
python demo.py                          # web demo
```

Mọi con số trong bảng kết quả đều khớp với `results/all_results_summary.json`. Nếu con số chạy ra không khớp, kiểm tra `SEED=42`, `random_state=42` trong split, và thứ tự chạy các bước.

Chúc bạn học vui với NLP tài chính.
