# Dự Án NLP Tài Chính — Tài Liệu Kỹ Thuật Toàn Diện

Tài liệu này là hướng dẫn chi tiết về mọi khía cạnh của dự án.
Nó được viết cho người hoàn toàn mới trong lĩnh vực NLP và học máy.
Sau khi đọc xong, bạn sẽ hiểu **tại sao** mọi quyết định được đưa ra,
**cách** mọi thuật toán hoạt động, và **mỗi** dòng code làm gì.

---

## Mục Lục

1. [Dự Án Này Làm Gì](#1-dự-án-này-làm-gì)
2. [Lý Thuyết Nền Tảng](#2-lý-thuyết-nền-tảng)
3. [Môi Trường & Thư Viện](#3-môi-trường--thư-viện)
4. [Kiến Trúc Dự Án](#4-kiến-trúc-dự-án)
5. [Chi Tiết Về Bộ Dữ Liệu](#5-chi-tiết-về-bộ-dữ-liệu)
6. [Từ Điển Loughran-McDonald](#6-từ-điển-loughran-mcdonald)
7. [Mô Hình Cơ Sở: TF-IDF + Hồi Quy Logistic (và các biến thể)](#7-mô-hình-cơ-sở-tf-idf--hồi-quy-logistic)
8. [Các Mô Hình Transformer Huấn Luyện Sẵn](#8-các-mô-hình-transformer-huấn-luyện-sẵn)
9. [Tinh Chỉnh FinBERT và BERT-base](#9-tinh-chỉnh-finbert)
10. [Học Đa Nhiệm](#10-học-đa-nhiệm)
11. [Phương Pháp Đánh Giá](#11-phương-pháp-đánh-giá)
12. [CLI, Demo, và Xuất Bản Lên HuggingFace](#12-cli-và-demo)
13. [Các Lỗi Gặp Phải và Cách Khắc Phục](#13-các-lỗi-gặp-phải-và-cách-khắc-phục)
14. [Phân Tích Kết Quả](#14-phân-tích-kết-quả)
15. [Bài Học Rút Ra](#15-bài-học-rút-ra)
16. [Tham Chiếu Code Đầy Đủ](#16-tham-chiếu-code-đầy-đủ--giải-thích-từng-hàm)

---

## 1. Dự Án Này Làm Gì

### 1.1 Hai Bài Toán

Dự án này giải quyết hai bài toán **phân loại văn bản** trong lĩnh vực tài chính:

**Bài toán 1 — Phân Loại Lập Trường (Stance Classification) trên Tập Dữ Liệu FOMC**

Cho một câu từ cuộc họp Cục Dự Trữ Liên Bang Mỹ (FOMC), phân loại nó thành:
- **Diều hâu (Hawkish)**: Văn bản ám chỉ việc thắt chặt chính sách tiền tệ
  (tăng lãi suất, giảm cung tiền). Ví dụ: *"Lạm phát vẫn ở mức cao và ủy ban
  tin rằng việc tăng lãi suất thêm là cần thiết."*
- **Bồ câu (Dovish)**: Văn bản ám chỉ việc nới lỏng chính sách tiền tệ
  (cắt giảm lãi suất, kích thích kinh tế). Ví dụ: *"Sự yếu kém của nền kinh tế
  cho thấy cần tiếp tục chính sách nới lỏng."*
- **Trung lập (Neutral)**: Văn bản không nghiêng rõ ràng về phía nào.
  Ví dụ: *"Chỉ số giá cổ phiếu rộng đã giảm mạnh trong kỳ họp."*

**Tại sao điều này quan trọng?** Các thông báo từ ngân hàng trung ương có thể
làm biến động thị trường tài chính. Nếu một nhà giao dịch có thể tự động phân
loại các tuyên bố của Fed là diều hâu hay bồ câu, họ có thể phản ứng nhanh hơn.
Đây là một lĩnh vực nghiên cứu đang rất sôi động trong NLP tài chính.

**Bài toán 2 — Phân Loại Cảm Xúc (Sentiment Classification) trên Financial PhraseBank**

Cho một câu từ bài báo tin tức tài chính, phân loại nó thành:
- **Tích cực (Positive)**: Tin tốt cho công ty/thị trường. Ví dụ: *"Doanh thu tăng 15% so với cùng kỳ."*
- **Tiêu cực (Negative)**: Tin xấu. Ví dụ: *"Công ty báo cáo lỗ ròng 50 triệu đô la."*
- **Trung lập (Neutral)**: Mang tính thông tin, không có cảm xúc rõ ràng.
  Ví dụ: *"Trụ sở công ty đặt tại Helsinki."*

### 1.2 Tại Sao Chọn Hai Bài Toán?

Chúng ta cố ý chọn hai bài toán để:
1. So sánh cách các mô hình xử lý các loại ngôn ngữ tài chính khác nhau
2. Cho phép **học đa nhiệm (multi-task learning)** — huấn luyện một mô hình
   trên cả hai bài toán đồng thời
3. Chứng minh rằng huấn luyện sẵn theo miền cụ thể (FinBERT) giúp ích cho
   các bài toán NLP tài chính

### 1.3 Hệ Thống Tín Chỉ

Môn học yêu cầu tối thiểu 80 tín chỉ trên bốn phần:

| Phần | Tối thiểu | Những gì chúng ta làm | Tín chỉ |
|------|-----------|------------------------|---------|
| A: Định Nghĩa Bài Toán | 10 | 2 bài toán NLP × 5 + 2 nguồn văn bản × 5 | 20 |
| B: Chọn Bộ Dữ Liệu | 20 | 2 bộ dữ liệu có sẵn (10) + 1 từ điển (10) | 20 |
| C: Mô Hình Hóa | 30 | Cơ sở + 3 biến thể cơ sở + 3 mô hình sẵn + tinh chỉnh FinBERT + BERT LLRD + đa nhiệm | 80+ |
| D: Đánh Giá | 20 | Định lượng + định tính + CLI + demo + publish lên HF Hub | 35 |
| **Tổng** | **80** | | **155+** |

---

## 2. Lý Thuyết Nền Tảng

### 2.1 Phân Loại Văn Bản

Phân loại văn bản là bài toán gán một **nhãn danh mục** cho một đoạn văn bản.
Đây là một trong những bài toán cơ bản nhất trong NLP.

Quy trình luôn là:
```
Văn bản thô → Trích xuất đặc trưng → Mô hình phân loại → Nhãn dự đoán
```

Câu hỏi chính là: **làm sao biến văn bản thành số?** Các phương pháp khác nhau:

| Phương pháp | Cách hoạt động | Thời kỳ |
|-------------|----------------|---------|
| Túi từ (Bag of Words) | Đếm tần suất từ | Thập niên 1990 |
| TF-IDF | Tần suất từ có trọng số | Thập niên 2000 |
| Word Embeddings | Vectơ dày đặc cho mỗi từ (Word2Vec, GloVe) | 2013+ |
| Transformers | Biểu diễn ngữ cảnh (BERT, GPT) | 2018+ |

### 2.2 TF-IDF (Tần Suất Thuật Ngữ – Tần Suất Tài Liệu Nghịch Đảo)

TF-IDF là một thống kê số phản ánh mức độ quan trọng của một từ đối với
một tài liệu trong một tập hợp (corpus).

**Tần Suất Thuật Ngữ (TF - Term Frequency)**: Từ xuất hiện bao nhiêu lần
trong tài liệu.
```
TF(từ, tài_liệu) = số_lần_xuất_hiện(từ trong tài_liệu) / tổng_số_từ(tài_liệu)
```

**Tần Suất Tài Liệu Nghịch Đảo (IDF - Inverse Document Frequency)**: Từ
hiếm như thế nào trong toàn bộ tập tài liệu.
```
IDF(từ) = log(tổng_số_tài_liệu / số_tài_liệu_chứa(từ))
```

**TF-IDF = TF × IDF**

Trực giác: Một từ xuất hiện thường xuyên trong một tài liệu nhưng hiếm
trong các tài liệu khác có lẽ là quan trọng cho tài liệu đó. Các từ phổ biến
như "the" (tiếng Anh) hoặc "của" (tiếng Việt) nhận điểm thấp (TF cao nhưng
IDF thấp), trong khi các từ đặc trưng nhận điểm cao.

**Trong code của chúng ta** (`src/baseline.py`):
```python
TfidfVectorizer(
    max_features=50_000,      # giữ 50K đặc trưng hàng đầu để tiết kiệm bộ nhớ
    ngram_range=(1, 2),       # dùng cả unigram VÀ bigram
    sublinear_tf=True,        # dùng log(1 + TF) thay vì TF thô
    strip_accents="unicode",  # chuẩn hóa ký tự có dấu
)
```

- `ngram_range=(1, 2)` nghĩa là chúng ta nắm bắt cả từ đơn ("inflation")
  và cụm hai từ ("rate hike"). Bigram thường mang nhiều ý nghĩa hơn unigram.
- `sublinear_tf=True` áp dụng chia tỷ lệ logarit: `1 + log(TF)`. Điều này
  ngăn các từ xuất hiện rất thường xuyên chi phối. Nếu không có log, một từ
  xuất hiện 100 lần sẽ được đánh trọng số 100× so với từ xuất hiện 1 lần;
  với log, chỉ khoảng 5.6× hơn.

### 2.3 Hồi Quy Logistic (Logistic Regression)

Hồi quy Logistic là một bộ phân loại tuyến tính mô hình hóa xác suất
của từng lớp.

Đối với phân loại nhị phân:
```
P(lớp=1 | x) = sigmoid(w·x + b) = 1 / (1 + exp(-(w·x + b)))
```

Đối với đa lớp (trường hợp của chúng ta với 3 lớp), nó dùng hàm **softmax**:
```
P(lớp=k | x) = exp(w_k·x + b_k) / Σ_j exp(w_j·x + b_j)
```

Mô hình học các trọng số `w` cho biết những đặc trưng nào (giá trị TF-IDF)
có khả năng dự đoán cao nhất cho từng lớp.

**Trong code của chúng ta**:
```python
LogisticRegression(
    max_iter=1000,             # cho phép tối đa 1000 vòng lặp tối ưu
    class_weight="balanced",   # tăng trọng số cho các lớp thiểu số
    random_state=SEED,         # đảm bảo tái tạo được kết quả
    solver="lbfgs",            # thuật toán tối ưu L-BFGS
)
```

- `class_weight="balanced"` rất quan trọng. Nếu không có, mô hình sẽ thiên
  về lớp đa số (trung lập). Với trọng số cân bằng, mỗi lớp được đánh trọng
  số tỷ lệ nghịch với tần suất:
  `trọng_số_k = n_mẫu / (n_lớp × n_mẫu_trong_lớp_k)`

### 2.4 Transformers và BERT

**Kiến trúc Transformer** (Vaswani và cộng sự, 2017) là nền tảng của NLP
hiện đại. Đổi mới chính của nó là **cơ chế tự chú ý (self-attention)**.

**Tự Chú Ý (Self-Attention)**: Với mỗi từ trong câu, mô hình tính toán mức
độ nó nên "chú ý" đến mọi từ khác. Điều này nắm bắt được các phụ thuộc
tầm xa mà các kiến trúc trước đó (RNN, LSTM) gặp khó khăn.

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

Trong đó Q (truy vấn - query), K (khóa - key), V (giá trị - value) là các
phép chiếu tuyến tính của đầu vào. Hệ số chia `√d_k` ngăn tích vô hướng
tăng quá lớn.

**BERT** (Biểu Diễn Mã Hóa Hai Chiều từ Transformers, Devlin và cộng sự, 2019)
là một Transformer được huấn luyện trên hai mục tiêu:
1. **Mô Hình Ngôn Ngữ Che Giấu (MLM - Masked Language Modeling)**: Che ngẫu
   nhiên 15% token, dự đoán chúng. Ví dụ: "Con mèo [MASK] trên thảm" → dự
   đoán "ngồi".
2. **Dự Đoán Câu Tiếp Theo (NSP - Next Sentence Prediction)**: Cho hai câu,
   dự đoán câu thứ hai có theo sau câu đầu tiên trong văn bản gốc không.

BERT là **hai chiều (bidirectional)** — nó đọc văn bản theo cả hai hướng
đồng thời, khác với GPT chỉ đọc từ trái sang phải.

**Token [CLS]**: BERT thêm một token đặc biệt `[CLS]` vào đầu mỗi đầu vào.
Sau khi xử lý qua tất cả các lớp, trạng thái ẩn của `[CLS]` đóng vai trò
như một **biểu diễn cấp câu** — một vectơ 768 chiều duy nhất nắm bắt ý nghĩa
của toàn bộ đầu vào. Đây là thứ chúng ta dùng để phân loại.

### 2.5 FinBERT

**FinBERT** (ProsusAI/finbert) là BERT được huấn luyện thêm trên văn bản
tài chính:
- Bắt đầu từ mô hình BERT-base tiêu chuẩn (110 triệu tham số)
- Được huấn luyện thêm trên kho dữ liệu lớn gồm tin tức tài chính, báo cáo
  thu nhập, và phân tích của nhà phân tích
- Sau đó được tinh chỉnh cho phân tích cảm xúc tài chính (tích cực/tiêu
  cực/trung lập)

**Tại sao FinBERT quan trọng**: BERT đa năng không hiểu tốt thuật ngữ tài
chính. Ví dụ, "cổ phiếu biến động mạnh" là tiêu cực trong tài chính nhưng
trung lập trong tiếng Anh thông thường. FinBERT học được những ý nghĩa
đặc thù theo lĩnh vực này.

### 2.6 Zero-Shot vs Few-Shot vs Tinh Chỉnh (Fine-Tuning)

Đây là các cách khác nhau để sử dụng mô hình huấn luyện sẵn:

**Zero-Shot (Không có mẫu)**: Sử dụng mô hình nguyên trạng, không huấn luyện
trên bài toán đích. FinBERT đã được huấn luyện cho cảm xúc tài chính, nên
chúng ta có thể dùng trực tiếp lớp phân loại có sẵn của nó trên Financial
PhraseBank.

**Few-Shot (Ít mẫu, k=16)**: Cho mô hình một lượng nhỏ dữ liệu có nhãn
(16 mẫu mỗi lớp = tổng cộng 48). Chúng ta đóng băng trọng số mô hình và
chỉ huấn luyện một bộ phân loại tuyến tính nhỏ trên các biểu diễn embedding
của nó. Điều này kiểm tra chất lượng biểu diễn đã học của mô hình.

**Tinh Chỉnh (Fine-Tuning)**: Cập nhật TẤT CẢ trọng số mô hình trên toàn
bộ tập huấn luyện. Đây là phương pháp mạnh nhất nhưng cần nhiều dữ liệu và
tài nguyên tính toán hơn. Mô hình điều chỉnh các biểu diễn bên trong cụ thể
cho bài toán đích.

### 2.7 Học Đa Nhiệm (Multi-Task Learning)

Học đa nhiệm huấn luyện một mô hình trên nhiều bài toán đồng thời.

**Kiến trúc**:
```
Văn bản đầu vào → Bộ Mã Hóa FinBERT Dùng Chung → Embedding [CLS]
                                                      ├─→ Đầu Lập Trường → diều hâu/bồ câu/trung lập
                                                      └─→ Đầu Cảm Xúc    → tích cực/tiêu cực/trung lập
```

**Tại sao nó giúp ích**:
1. **Biểu diễn dùng chung**: Cả hai bài toán đều liên quan đến hiểu ngôn ngữ
   tài chính. Huấn luyện trên cảm xúc giúp mô hình hiểu lập trường, và
   ngược lại.
2. **Chính quy hóa (Regularization)**: Học nhiều bài toán đóng vai trò như
   một dạng chính quy hóa, ngăn quá khớp (overfitting) vào bất kỳ bài toán
   đơn lẻ nào.
3. **Hiệu quả dữ liệu**: Tập dữ liệu lập trường nhỏ (~1700 mẫu huấn luyện).
   Bằng cách cũng huấn luyện trên dữ liệu cảm xúc, bộ mã hóa dùng chung
   nhìn thấy nhiều văn bản tài chính hơn.

**Quy trình huấn luyện**: Chúng ta luân phiên các batch từ hai bộ dữ liệu.
Một batch huấn luyện lập trường (cập nhật bộ mã hóa dùng chung + đầu lập
trường), rồi một batch huấn luyện cảm xúc (cập nhật bộ mã hóa dùng chung
+ đầu cảm xúc).

### 2.8 Hàm Mất Mát Cross-Entropy Có Trọng Số

Hàm cross-entropy tiêu chuẩn đối xử bình đẳng với tất cả các lớp:
```
L = -Σ y_thực × log(y_dự_đoán)
```

Khi các lớp mất cân bằng (ví dụ: FOMC có gấp ~2 lần trung lập so với diều
hâu), mô hình học cách dự đoán lớp đa số. **Cross-entropy có trọng số** gán
mất mát cao hơn cho các lớp thiểu số:
```
L = -Σ w_k × y_thực × log(y_dự_đoán)
```

Trong đó `w_k = N / (C × n_k)`:
- N = tổng số mẫu
- C = số lớp
- n_k = số mẫu trong lớp k

Với FOMC: trọng số bồ câu ≈ 1.27, diều hâu ≈ 1.37, trung lập ≈ 0.68.
Nghĩa là phân loại sai một câu diều hâu tốn gấp ~2 lần so với phân loại
sai một câu trung lập, buộc mô hình phải chú ý nhiều hơn đến các lớp thiểu số.

---

## 3. Môi Trường & Thư Viện

### 3.1 Phiên Bản Python

Chúng ta dùng **Python 3.12** (không phải bản mặc định hệ thống 3.14) vì
PyTorch chưa hỗ trợ Python 3.14. Môi trường ảo được tạo bằng:
```bash
/opt/homebrew/bin/python3.12 -m venv venv
```

### 3.2 Phần Cứng

**Apple M3 Max** với backend MPS (Metal Performance Shaders). MPS là framework
tính toán GPU của Apple cho PyTorch, tương tự CUDA của NVIDIA nhưng cho chip
Apple Silicon.

```python
# Trong config.py:
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
```

MPS cung cấp tốc độ nhanh hơn đáng kể so với CPU cho các phép tính ma trận
(nền tảng của huấn luyện mạng nơ-ron). Tinh chỉnh FinBERT chạy ở ~1.5-2
batch/giây trên MPS so với ~0.3 trên CPU.

### 3.3 Chi Tiết Từng Thư Viện

**`torch` (PyTorch 2.10+)**
Framework học sâu. Cung cấp:
- Phép toán tensor (giống NumPy nhưng hỗ trợ GPU)
- Vi phân tự động (tính gradient cho lan truyền ngược - backpropagation)
- Các module mạng nơ-ron (`nn.Module`, `nn.Linear`, v.v.)
- Bộ tối ưu (`AdamW`, `SGD`)

**`transformers` (HuggingFace Transformers 5.3+)**
Thư viện chuẩn cho các mô hình NLP huấn luyện sẵn. Cung cấp:
- `AutoTokenizer`: Chuyển đổi văn bản thành ID token mà mô hình hiểu được
- `AutoModel`: Tải trọng số mô hình đã huấn luyện sẵn
- `AutoModelForSequenceClassification`: Mô hình với đầu phân loại
- `pipeline()`: API suy luận cấp cao
- `get_linear_schedule_with_warmup()`: Bộ lập lịch tốc độ học

**`datasets` (HuggingFace Datasets 4.7+)**
Tải và xử lý bộ dữ liệu hiệu quả:
- `load_dataset()`: Tải bộ dữ liệu từ HuggingFace Hub
- `Dataset`, `DatasetDict`: Lưu trữ theo cột hiệu quả (backend Apache Arrow)
- Xử lý chia tập train/test, xáo trộn, tạo batch

**`scikit-learn` (1.8+)**
Thư viện học máy cổ điển:
- `TfidfVectorizer`: Tính đặc trưng TF-IDF
- `LogisticRegression`: Phân loại tuyến tính
- `train_test_split()`: Chia dữ liệu có phân tầng (stratified)
- `classification_report()`, `confusion_matrix()`: Các chỉ số đánh giá

**`pandas` (3.0+)**
Thư viện xử lý dữ liệu. Chúng ta dùng để:
- Chuyển đổi bộ dữ liệu HuggingFace sang DataFrame để thao tác dễ hơn
- Phép groupby để lấy mẫu (few-shot)
- Phân tích lỗi (sắp xếp các mẫu bị phân loại sai)

**`numpy` (2.4+)**
Tính toán số. Dùng cho các phép toán mảng, đặc biệt trong trích xuất đặc
trưng từ điển.

**`matplotlib` + `seaborn`**
Thư viện trực quan hóa. Chúng ta dùng để tạo biểu đồ nhiệt ma trận nhầm
lẫn (confusion matrix). `seaborn.heatmap()` tạo ra các ma trận nhầm lẫn
có chú thích được lưu trong `results/`.

**`gradio` (4.19+)**
Framework giao diện web cho demo ML. Tạo giao diện web tương tác với code
tối thiểu. Demo của chúng ta có ô nhập văn bản, hai đầu ra nhãn (lập trường
+ cảm xúc), và các câu ví dụ.

**`accelerate` (0.27+)**
Thư viện HuggingFace cho huấn luyện phân tán. Được `transformers` yêu cầu
để tải mô hình, ngay cả khi chỉ dùng một thiết bị.

**`tqdm` (4.66+)**
Thư viện thanh tiến trình. Bọc các vòng lặp để hiển thị tiến trình huấn luyện:
```
Epoch 1/5: 45%|████▌     | 25/55 [00:13<00:16, 1.83it/s]
```

### 3.4 Các Tham Số Cấu Hình Chính (config.py)

```python
MAX_SEQ_LENGTH = 128    # Số token tối đa cho mỗi đầu vào (BERT tối đa 512,
                        # nhưng câu của chúng ta ngắn — 128 tiết kiệm bộ nhớ)
BATCH_SIZE = 32         # Số mẫu được xử lý cùng lúc trên GPU
LEARNING_RATE = 2e-5    # Chuẩn cho tinh chỉnh BERT (từ bài báo gốc)
WEIGHT_DECAY = 0.01     # Chính quy hóa L2 để ngăn quá khớp
FINETUNE_EPOCHS = 5     # Số lần duyệt qua toàn bộ tập dữ liệu
MULTITASK_EPOCHS = 8    # Nhiều epoch hơn cho đa nhiệm (hai bộ dữ liệu)
FEW_SHOT_K = 16         # Số mẫu mỗi lớp cho học ít mẫu
WARMUP_RATIO = 0.1      # Khởi động tốc độ học cho 10% đầu huấn luyện
SEED = 42               # Hạt giống ngẫu nhiên để tái tạo kết quả
TEST_SIZE = 0.2          # 20% dữ liệu cho kiểm thử
VAL_SIZE = 0.1           # 10% dữ liệu cho xác nhận
```

**Tại sao dùng các giá trị này?**
- `LEARNING_RATE = 2e-5`: Đây là chuẩn từ bài báo gốc BERT (Devlin và cộng
  sự, 2019). Quá cao (ví dụ 1e-3) phá hủy trọng số đã huấn luyện sẵn; quá
  thấp (ví dụ 1e-6) học quá chậm.
- `WARMUP_RATIO = 0.1`: Dần dần tăng tốc độ học từ 0 đến 2e-5 trong 10%
  bước huấn luyện đầu tiên. Điều này ngăn mô hình thực hiện các cập nhật lớn,
  phá hủy trọng số sớm trong huấn luyện khi gradient còn nhiễu.
- `MAX_SEQ_LENGTH = 128`: Các câu của chúng ta thường dài 10-50 từ (≈15-70
  token). 128 cho không gian dự phòng mà không lãng phí bộ nhớ vào padding.

---

## 4. Kiến Trúc Dự Án

### 4.1 Hướng Dẫn Từng File

```
Project/
├── config.py                  # TẤT CẢ cài đặt ở một nơi
├── src/
│   ├── __init__.py            # Biến src/ thành package Python
│   ├── data_loader.py         # Tải và chia bộ dữ liệu
│   ├── baseline.py            # TF-IDF + LR + các baseline khác (SVM, trigram)
│   ├── lexicon.py             # Phương pháp từ điển Loughran-McDonald
│   ├── pretrained_eval.py     # Thí nghiệm zero-shot + few-shot
│   ├── finetune_fineBert.py   # Tinh chỉnh FinBERT đơn nhiệm
│   ├── finetune_bert.py       # BERT-base-uncased với LLRD + gradual unfreezing
│   ├── multitask.py           # Kiến trúc + huấn luyện mô hình đa nhiệm
│   └── evaluate.py            # Chỉ số, biểu đồ, phân tích lỗi
├── run_experiments.py         # Điều phối tất cả thí nghiệm (Bước 1-6)
├── cli.py                     # Công cụ dự đoán dòng lệnh
├── demo.py                    # Giao diện web Gradio
├── push_to_hf.py              # Upload mô hình đã huấn luyện lên HuggingFace Hub
├── data_analysis.py           # Sinh 10 biểu đồ phân tích + thống kê dữ liệu
├── create_presentation.py     # Tạo slide / báo cáo cuối
├── requirements.txt           # Các thư viện Python phụ thuộc
├── analysis/                  # 10 biểu đồ PNG do data_analysis.py tạo ra
├── models/                    # Trọng số mô hình đã lưu sau huấn luyện
│   ├── finbert_stance/        # FinBERT tinh chỉnh cho stance
│   ├── finbert_sentiment/     # FinBERT tinh chỉnh cho sentiment
│   ├── bert_llrd_stance/      # BERT-base LLRD cho stance (tạo ra khi chạy step 6)
│   ├── bert_llrd_sentiment/   # BERT-base LLRD cho sentiment (tạo ra khi chạy step 6)
│   └── multitask_finbert/     # Mô hình đa nhiệm (encoder dùng chung + 2 head)
│   # Chỉ finbert_* và multitask_finbert được commit; bert_llrd_* được
│   # tạo lại local qua `python run_experiments.py --step 6`.
└── results/                   # Chỉ số JSON + Ma trận nhầm lẫn PNG
```

### 4.2 Luồng Dữ Liệu

```
run_experiments.py
    │
    ├── Bước 1: data_loader.py       → Tải bộ dữ liệu FOMC + FPB
    │                                   Chia thành train/val/test
    │
    ├── Bước 2: baseline.py          → TF-IDF + LR (baseline gốc)
    │                                   + TF-IDF + SVM (biến thể)
    │                                   + TF-IDF (trigram) + LR (biến thể)
    │           lexicon.py           → Phân loại dựa trên từ điển LM + TF-IDF+lexicon
    │
    ├── Bước 3: pretrained_eval.py   → FinBERT zero-shot
    │                                   Few-shot (FinBERT, BERT, RoBERTa)
    │
    ├── Bước 4: finetune_fineBert.py → FinBERT tinh chỉnh trên FOMC
    │                                   FinBERT tinh chỉnh trên FPB
    │
    ├── Bước 5: multitask.py         → Huấn luyện đồng thời trên cả hai bộ dữ liệu
    │
    └── Bước 6: finetune_bert.py     → BERT-base + LLRD + gradual unfreezing
                                       (stance + sentiment, mô hình tách riêng)
```

Mỗi bước dùng `evaluate.py` để tính chỉ số và lưu kết quả.

### 4.3 Tại Sao Cấu Trúc Này?

- **Phân tách trách nhiệm**: Mỗi loại mô hình nằm trong file riêng. Bạn có
  thể hiểu `baseline.py` mà không cần đọc `multitask.py`.
- **Cấu hình tập trung**: Tất cả siêu tham số trong `config.py` nghĩa là bạn
  thay đổi cài đặt ở một nơi, không phải rải rác khắp các file.
- **Chạy theo bước**: `run_experiments.py --step N` cho phép bạn chạy lại
  từng thí nghiệm riêng lẻ mà không phải chạy lại tất cả.

---

## 5. Chi Tiết Về Bộ Dữ Liệu

### 5.1 Bộ Dữ Liệu FOMC Diều Hâu-Bồ Câu

**Nguồn**: Georgia Tech Financial Technology Lab (gtfintechlab)
**Bài báo**: "Trillion Dollar Words" (Shah và cộng sự, ACL 2023)
**ID trên HuggingFace**: `gtfintechlab/fomc_communication`

**Nội dung**: Các câu được trích xuất từ biên bản và tuyên bố cuộc họp FOMC
(2000-2023). Mỗi câu được gán nhãn bởi chuyên gia tài chính.

**Phân bố nhãn** (sau khi chia tập):
```
Train: 1736 mẫu (bồ câu: 455, diều hâu: 424, trung lập: 857)
Val:    248 mẫu (bồ câu:  65, diều hâu:  61, trung lập: 122)
Test:   496 mẫu (bồ câu: 130, diều hâu: 121, trung lập: 245)
```

Lưu ý **mất cân bằng lớp**: trung lập (~49%) gần gấp đôi diều hâu (~24%)
hay bồ câu (~26%). Đây là lý do chúng ta dùng hàm mất mát cross-entropy
có trọng số.

**Ví dụ các câu**:
- Bồ câu: *"Low readings on overall and core consumer price inflation in recent
  months, as well as the weakened economic outlook..."*
  (Tạm dịch: Các chỉ số lạm phát thấp và triển vọng kinh tế suy yếu...)
- Diều hâu: *"Our new statement explicitly acknowledges the challenges posed by
  the proximity of interest rates to..."*
  (Tạm dịch: Tuyên bố mới thừa nhận thách thức từ mức lãi suất gần...)
- Trung lập: *"Broad equity price indexes fell sharply over the intermeeting period."*
  (Tạm dịch: Chỉ số giá cổ phiếu rộng giảm mạnh trong kỳ họp.)

### 5.2 Financial PhraseBank

**Nguồn**: Malo và cộng sự (2014), "Good Debt or Bad Debt"
**ID trên HuggingFace**: `gtfintechlab/financial_phrasebank_sentences_allagree`

**Nội dung**: 2,264 câu từ tin tức tài chính tiếng Anh, được chú thích bởi
16 chuyên gia tài chính. Tập con `sentences_allagree` nghĩa là TẤT CẢ người
chú thích đồng ý về nhãn — chất lượng chú thích cao nhất.

**Tại sao dùng "allagree"?** Bộ dữ liệu đầy đủ có 4,846 câu với các mức
đồng thuận khác nhau (50%, 66%, 75%, 100%). Dùng đồng thuận 100% cho chúng
ta nhãn sạch nhất, ít mơ hồ nhất. Đánh đổi là ít mẫu hơn.

**Phân bố nhãn** (sau khi chia tập):
```
Train: 1584 mẫu (tiêu cực: 212, trung lập: 973, tích cực: 399)
Val:    227 mẫu (tiêu cực:  30, trung lập: 140, tích cực:  57)
Test:   453 mẫu (tiêu cực:  61, trung lập: 278, tích cực: 114)
```

Một lần nữa, **mất cân bằng lớp**: trung lập chiếm ưu thế (~61%), tiêu cực
là thiểu số (~13%).

### 5.3 Chiến Lược Chia Dữ Liệu

Chúng ta dùng **chia phân tầng (stratified splitting)** để đảm bảo mỗi tập
con có cùng tỷ lệ lớp như toàn bộ bộ dữ liệu:

```python
train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
```

Các tập chia: **70% huấn luyện / 10% xác nhận / 20% kiểm thử**.

- **Huấn luyện (Train)**: Mô hình học từ dữ liệu này
- **Xác nhận (Validation)**: Dùng để chọn mô hình tốt nhất (dừng sớm)
- **Kiểm thử (Test)**: Đánh giá cuối cùng — KHÔNG BAO GIỜ được dùng trong
  quá trình huấn luyện

`random_state=42` đảm bảo cùng cách chia mỗi khi chạy code.

### 5.4 Thư Viện HuggingFace Datasets

Ban đầu, các bộ dữ liệu này dùng script tải Python trên HuggingFace. Nhưng
thư viện `datasets` (v4.0+) đã loại bỏ hỗ trợ script tải để ưu tiên file
Parquet. Đây là lý do chúng ta dùng phiên bản `gtfintechlab/` thay vì
`takala/financial_phrasebank` gốc — chúng đã được chuyển đổi sang định dạng
Parquet. Dữ liệu giống hệt nhau; chỉ khác định dạng lưu trữ.

```python
# Cách này KHÔNG CÒN hoạt động (script tải cũ):
load_dataset("takala/financial_phrasebank", "sentences_allagree")

# Cách này HOẠT ĐỘNG (phiên bản Parquet):
load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", "5768")
```

`"5768"` là tên cấu hình được yêu cầu bởi phiên bản bộ dữ liệu cụ thể này
(nó tham chiếu đến một cấu hình chia dữ liệu cụ thể).

---

## 6. Từ Điển Loughran-McDonald

### 6.1 Từ Điển (Lexicon) Là Gì?

Một **từ điển (lexicon)** trong NLP là danh sách từ được xác định trước, gắn
liền với các danh mục cụ thể. Khác với phương pháp học máy học từ dữ liệu,
phương pháp dựa trên từ điển sử dụng kiến thức do con người tổng hợp.

### 6.2 Tại Sao Loughran-McDonald?

Các từ điển cảm xúc tiêu chuẩn (như Harvard General Inquirer hay VADER) hoạt
động kém trên văn bản tài chính vì nhiều từ có **ý nghĩa khác nhau** trong
tài chính:

| Từ | Cảm xúc chung | Cảm xúc tài chính |
|----|---------------|-------------------|
| "liability" (nợ phải trả) | Tiêu cực | Trung lập (thuật ngữ kế toán) |
| "tax" (thuế) | Tiêu cực | Trung lập (kinh doanh bình thường) |
| "capital" (vốn) | Trung lập | Tích cực (dấu hiệu mạnh mẽ) |
| "crude" (thô) | Tiêu cực | Trung lập (dầu thô) |

Loughran và McDonald (2011) đã tạo từ điển đặc biệt cho văn bản tài chính
bằng cách phân tích hơn 50,000 báo cáo 10-K. Danh sách từ của họ là tiêu
chuẩn vàng trong nghiên cứu NLP tài chính.

### 6.3 Các Nhóm Từ Của Chúng Ta

Chúng ta dùng sáu danh sách từ trong `src/lexicon.py`:

1. **LM_POSITIVE** (~110 từ): Từ chỉ kết quả tốt —
   "achieve", "benefit", "profit", "recovery", "strength"

2. **LM_NEGATIVE** (~230 từ): Từ chỉ kết quả xấu —
   "bankruptcy", "decline", "default", "loss", "recession"

3. **LM_UNCERTAINTY** (~90 từ): Từ chỉ sự mơ hồ —
   "approximate", "contingent", "maybe", "uncertain", "variable"

4. **HAWKISH_WORDS** (~40 từ): Thắt chặt chính sách tiền tệ —
   "hike", "tighten", "inflation", "restrictive", "tapering"

5. **DOVISH_WORDS** (~50 từ): Nới lỏng chính sách tiền tệ —
   "cut", "ease", "accommodate", "stimulus", "patient"

Các danh sách diều hâu/bồ câu là phần bổ sung tùy chỉnh dành riêng cho bài
toán phân loại lập trường, được tổng hợp từ tài liệu ngân hàng trung ương.

### 6.4 Trích Xuất Đặc Trưng

Với mỗi văn bản, chúng ta tính 8 đặc trưng số:

```python
[positive_count, negative_count, uncertainty_count,
 hawkish_count, dovish_count,
 net_sentiment,   # (tích_cực - tiêu_cực) / tổng_từ
 net_stance,      # (diều_hâu - bồ_câu) / tổng_từ
 total_words]
```

**Chuẩn hóa** theo tổng số từ rất quan trọng. Một câu 50 từ với 3 từ tiêu
cực là tiêu cực hơn một câu 200 từ với 3 từ tiêu cực.

### 6.5 Hai Cách Sử Dụng Từ Điển

**Bộ phân loại dựa trên quy tắc** (không cần huấn luyện):
- Cho cảm xúc: nếu net_sentiment > 0.02 → tích cực; < -0.02 → tiêu cực; nếu không → trung lập
- Cho lập trường: nếu hawkish_count > dovish_count → diều hâu; v.v.

**TF-IDF + Đặc trưng từ điển** (có huấn luyện):
- Nối đặc trưng TF-IDF với 8 đặc trưng từ điển
- Huấn luyện Hồi Quy Logistic trên tập đặc trưng kết hợp
- Điều này cho phép mô hình sử dụng cả mẫu cấp từ (TF-IDF) và kiến thức
  miền (từ điển)

### 6.6 Kết Quả

Bộ phân loại dựa trên quy tắc từ điển thuần túy hoạt động kém nhất (41.5%
lập trường, 69.3% cảm xúc), điều này được kỳ vọng — nó không có khả năng học.
Nhưng nó chứng minh rằng từ điển chứa **tín hiệu có ý nghĩa**.

Sự kết hợp TF-IDF + Từ điển cho thấy cải thiện nhẹ so với TF-IDF thuần cho
lập trường (61.1% so với 60.9%) nhưng hơi thấp hơn cho cảm xúc (85.4% so
với 87.2%). Điều này gợi ý TF-IDF đã nắm bắt phần lớn tín hiệu của từ điển.

---

## 7. Mô Hình Cơ Sở: TF-IDF + Hồi Quy Logistic

### 7.1 Mục Đích

Mọi dự án ML đều cần một **mô hình cơ sở (baseline)** — một mô hình đơn giản
đặt mức sàn. Nếu một mô hình phức tạp không thắng được baseline, nó không
đáng để phức tạp hóa.

TF-IDF + Hồi Quy Logistic là baseline phi nơ-ron tiêu chuẩn cho phân loại
văn bản. Nó nhanh, dễ giải thích, và thường cạnh tranh đáng ngạc nhiên.

### 7.2 Chi Tiết Pipeline (`src/baseline.py`)

```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(...)),
    ("clf", LogisticRegression(...)),
])
```

`Pipeline` của scikit-learn nối bộ vector hóa và bộ phân loại, đảm bảo
cùng biến đổi được áp dụng nhất quán cho dữ liệu train và test.

**Bước 1: Vector hóa TF-IDF**
```
"The Fed raised rates" → [0, 0, 0.42, 0, 0.71, ..., 0, 0.38]
                          (vectơ thưa 50,000 chiều)
```

**Bước 2: Hồi Quy Logistic**
```
[0, 0, 0.42, ..., 0.38] → softmax(W·x + b) → [0.1, 0.7, 0.2]
                                                 (bồ câu, diều hâu, trung lập)
```

### 7.3 Kết Quả

| Bài toán | Accuracy | Macro-F1 |
|----------|----------|----------|
| Lập trường | 0.6089 | 0.5873 |
| Cảm xúc | 0.8720 | 0.8232 |

Baseline mạnh đáng ngạc nhiên cho cảm xúc (87.2%!) vì cảm xúc tài chính
thường tương quan với các từ cụ thể ("profit" → tích cực, "loss" → tiêu cực).
Lập trường khó hơn vì cùng các từ có thể xuất hiện trong cả ngữ cảnh diều
hâu và bồ câu — nó phụ thuộc vào **ý nghĩa toàn câu**.

### 7.4 Các Baseline Phi-Nơ-ron Khác

Ngoài baseline gốc TF-IDF + LR, `src/baseline.py` hiện còn chạy thêm hai
pipeline phi-nơ-ron khác thường mạnh hơn trong thực tế:

**TF-IDF (1-2 gram) + Linear SVM** — `build_tfidf_svm_pipeline()`

```python
LinearSVC(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)
```

Linear SVM tối đa hóa **biên (margin)** giữa các lớp trong không gian đặc
trưng TF-IDF. Với dữ liệu thưa chiều cao (50k đặc trưng), LinearSVC thường
mạnh hơn Hồi Quy Logistic một chút vì nó tối ưu hàm mất mát hinge, chỉ
quan tâm đến các vector hỗ trợ (các điểm gần ranh giới).

**TF-IDF (1-3 gram) + Hồi Quy Logistic** — `build_tfidf_trigram_lr_pipeline()`

```python
TfidfVectorizer(max_features=80_000, ngram_range=(1, 3),
                sublinear_tf=True, min_df=2)
```

Mở rộng n-gram lên 3 giúp mô hình bắt được các cụm ngắn như
*"rate hike cycle"* hay *"below zero lower bound"*. `min_df=2` loại bỏ các
n-gram quá hiếm để kiểm soát số đặc trưng; `max_features=80_000` nâng trần
từ vựng để chứa được không gian n-gram lớn hơn.

**Kết quả trên tập test**

| Baseline | Stance Acc | Stance F1 | Sent. Acc | Sent. F1 |
|----------|-----------|-----------|-----------|----------|
| TF-IDF + LR (gốc)              | 0.6089 | 0.5873 | 0.8720 | 0.8232 |
| TF-IDF + SVM                   | **0.6331** | **0.6061** | **0.8940** | **0.8534** |
| TF-IDF (trigram) + LR          | 0.6109 | 0.5914 | 0.8786 | 0.8310 |

SVM là baseline phi-nơ-ron mạnh nhất trên **cả hai** bài toán. Đặc trưng
trigram cho cải thiện nhỏ nhưng ổn định so với baseline bigram cho cảm xúc.
Các mô hình này được tạo bởi `run_alternative_baselines(fomc, fpb)` trong
bước 2 của `run_experiments.py`.

---

## 8. Các Mô Hình Transformer Huấn Luyện Sẵn

### 8.1 Các Mô Hình Được So Sánh

| Mô hình | Tham số | Dữ liệu huấn luyện | Tài chính? |
|---------|---------|---------------------|------------|
| FinBERT | 110 triệu | Tin tài chính + báo cáo | Có |
| BERT-base-uncased | 110 triệu | Wikipedia + BookCorpus | Không |
| RoBERTa-base | 125 triệu | Văn bản web (80GB) | Không |

**BERT-base-uncased**: "uncased" nghĩa là nó chuyển tất cả văn bản thành chữ
thường trước khi tokenize. "The" và "the" trở thành cùng một token. Điều này
giúp nhất quán nhưng mất thông tin (ví dụ: danh từ riêng).

**RoBERTa-base** (BERT Tối Ưu Mạnh Mẽ): Cùng kiến trúc với BERT nhưng
huấn luyện lâu hơn, trên nhiều dữ liệu hơn, với siêu tham số tốt hơn.
Loại bỏ mục tiêu NSP. Thường hoạt động tốt hơn BERT-base.

### 8.2 Zero-Shot: Đầu Phân Loại Gốc Của FinBERT

FinBERT đã được tinh chỉnh cho phân loại cảm xúc (tích cực/tiêu cực/trung
lập). Chúng ta có thể dùng trực tiếp — không cần huấn luyện.

```python
clf = pipeline("text-classification", model="ProsusAI/finbert")
result = clf("Revenue grew 15%")
# → [{"label": "positive", "score": 0.97}]
```

**Trên cảm xúc (Financial PhraseBank)**: 97.4% accuracy — gần như hoàn hảo
vì FinBERT được huấn luyện cho chính loại bài toán này.

**Trên lập trường (FOMC)**: 49.8% accuracy — kém, vì cảm xúc ≠ lập trường.
Một câu có thể tiêu cực về cảm xúc nhưng diều hâu về lập trường
(ví dụ: "lạm phát vẫn ở mức nguy hiểm cao" — tin xấu, nhưng diều hâu
vì nó gợi ý tăng lãi suất).

### 8.3 Few-Shot: Thăm Dò Tuyến Tính (Linear Probe)

Cho đánh giá few-shot, chúng ta:

1. **Đóng băng** mô hình huấn luyện sẵn (không cập nhật trọng số)
2. Đưa tất cả văn bản qua mô hình để lấy embedding [CLS] (vectơ 768 chiều)
3. Huấn luyện bộ phân loại tuyến tính nhỏ trên chỉ 48 mẫu (16 mỗi lớp)

```python
class FewShotClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),                        # ngăn quá khớp
            nn.Linear(hidden_size, num_labels),     # 768 → 3
        )
```

Điều này kiểm tra **chất lượng biểu diễn của mô hình**. Một mô hình hiểu
tốt tài chính sẽ tạo ra embedding mà các văn bản cùng lớp gom lại gần nhau,
giúp phân loại dễ dàng ngay cả với rất ít mẫu.

### 8.4 Quá Trình Mã Hóa (Encoding)

```python
def _encode_texts(tokenizer, model, texts, device):
    for batch in batches:
        inputs = tokenizer(batch, padding=True, truncation=True, ...)
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # token [CLS]
```

- `tokenizer(...)` chuyển văn bản thành ID token + mặt nạ chú ý (attention mask)
- `model(**inputs)` chạy forward pass qua tất cả 12 lớp Transformer
- `outputs.last_hidden_state[:, 0, :]` trích xuất embedding [CLS]
  (token đầu tiên, tất cả 768 chiều)

### 8.5 So Sánh Kết Quả

| Mô hình | F1 Lập trường | F1 Cảm xúc |
|---------|---------------|------------|
| FinBERT (few-shot) | 0.4534 | **0.9670** |
| BERT-base (few-shot) | 0.3744 | 0.6500 |
| RoBERTa-base (few-shot) | 0.3600 | 0.6722 |

**Nhận xét chính**: FinBERT vượt trội hoàn toàn so với các mô hình đa năng
trên bài toán tài chính, ngay cả chỉ với 16 mẫu mỗi lớp. Điều này chứng
minh **huấn luyện sẵn theo miền cụ thể có giá trị rất cao**. BERT và RoBERTa
tạo ra embedding chung chung không nắm bắt tốt ý nghĩa tài chính.

---

## 9. Tinh Chỉnh FinBERT và BERT-base

Dự án hiện có **hai** chiến lược tinh chỉnh:

1. **Tinh chỉnh FinBERT đơn nhiệm** (`src/finetune_fineBert.py`) —
   công thức chuẩn: unfreeze tất cả các lớp từ epoch 1, dùng một learning
   rate đồng nhất, schedule linear warmup + decay.
2. **BERT-base với LLRD + Gradual Unfreezing** (`src/finetune_bert.py`) —
   chiến lược thận trọng hơn, áp dụng trên mô hình nền *không chuyên ngành*
   để chứng minh rằng tinh chỉnh có kỷ luật có thể thu hẹp khoảng cách với
   FinBERT (mô hình đã pre-train theo miền).

Các mục 9.1–9.6 mô tả chiến lược #1. Mục 9.7 mô tả chiến lược #2.

### 9.1 Gì Thay Đổi Trong Quá Trình Tinh Chỉnh

Khác với few-shot (mô hình đóng băng + bộ phân loại nhỏ), tinh chỉnh cập
nhật **tất cả 110 triệu tham số** của FinBERT để tối ưu cho bài toán cụ thể.

Mô hình bắt đầu từ trọng số huấn luyện sẵn và dần dần thích ứng:
- **Các lớp đầu** (embedding, vài khối Transformer đầu) học đặc trưng ngôn
  ngữ tổng quát và thay đổi ít
- **Các lớp sau** ngày càng chuyên biệt hóa cho bài toán đích
- **Đầu phân loại (classification head)** (mới, khởi tạo ngẫu nhiên) học
  ánh xạ bài toán

### 9.2 Vòng Lặp Huấn Luyện (`src/finetune_fineBert.py`)

```python
for epoch in range(FINETUNE_EPOCHS):
    model.train()
    for batch in train_loader:
        # Forward pass (truyền xuôi)
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs.logits, labels)

        # Backward pass (truyền ngược)
        optimizer.zero_grad()       # xóa gradient cũ
        loss.backward()             # tính gradient mới
        clip_grad_norm_(model.parameters(), 1.0)  # ngăn gradient bùng nổ
        optimizer.step()            # cập nhật trọng số
        scheduler.step()            # cập nhật tốc độ học
```

**Cắt gradient (Gradient clipping)** (`clip_grad_norm_(..., 1.0)`) ngăn
gradient trở nên quá lớn, điều có thể làm mất ổn định huấn luyện. Nó thay
đổi tỷ lệ gradient sao cho tổng norm không vượt quá 1.0.

### 9.3 Lịch Biểu Tốc Độ Học

Chúng ta dùng **khởi động tuyến tính + suy giảm tuyến tính**:

```
Tốc Độ Học
    ^
2e-5|           /‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
    |          /                  \
    |         /                    \
    |        /                      \
  0 |_______/________________________\___→ Bước Huấn Luyện
    0    10%                         100%
         ↑ kết thúc khởi động
```

**Tại sao cần khởi động?** Khi bắt đầu huấn luyện, đầu phân loại có trọng
số ngẫu nhiên. Nếu tốc độ học cao ngay lập tức, gradient từ đầu ngẫu nhiên
lan truyền ngược và phá hủy các trọng số bộ mã hóa đã được huấn luyện cẩn
thận. Khởi động cho đầu phân loại ổn định trước khi bộ mã hóa bắt đầu thay
đổi đáng kể.

### 9.4 Chọn Mô Hình Tốt Nhất

Chúng ta theo dõi F1 xác nhận sau mỗi epoch và lưu mô hình tốt nhất:

```python
if val_metrics["macro_f1"] > best_val_f1:
    best_val_f1 = val_metrics["macro_f1"]
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
```

Điều này ngăn **quá khớp (overfitting)**: mô hình có thể đạt hiệu suất huấn
luyện tuyệt vời ở các epoch sau nhưng hiệu suất test kém hơn. Bằng cách lưu
checkpoint xác nhận tốt nhất, chúng ta có được mô hình tổng quát hóa tốt nhất.

### 9.5 Lớp `TextClassificationDataset`

```python
class TextClassificationDataset(TorchDataset):
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
```

Lớp này bọc dữ liệu cho `DataLoader` của PyTorch. Với mỗi mẫu:
- **input_ids**: ID token (số nguyên). Ví dụ: "The Fed" → [101, 1996, 5765, 102]
- **attention_mask**: Mặt nạ nhị phân. 1 = token thật, 0 = padding.
  Ví dụ: [1, 1, 1, 1, 0, 0, ..., 0] cho câu ngắn có padding
- **labels**: Nhãn lớp số nguyên (0, 1, hoặc 2)

`padding="max_length"` đệm tất cả chuỗi đến 128 token để chúng có thể được
gộp thành một tensor duy nhất. `truncation=True` cắt các chuỗi dài hơn 128.

### 9.6 Kết Quả

| Bài toán | Accuracy | Macro-F1 |
|----------|----------|----------|
| Lập trường | 0.6129 | 0.5988 |
| Cảm xúc | 0.9669 | 0.9467 |

Tinh chỉnh cải thiện so với baseline:
- Lập trường: +0.4% accuracy, +1.2% F1 (khiêm tốn — lập trường là bài toán khó)
- Cảm xúc: +9.5% accuracy, +12.4% F1 (cải thiện đáng kể)

### 9.7 BERT-base-uncased với LLRD + Gradual Unfreezing

File: `src/finetune_bert.py`. Được gọi từ bước 6 của `run_experiments.py`.

**Động lực.** `src/finetune_fineBert.py` tinh chỉnh một mô hình *đã
pre-train theo miền* (FinBERT) một cách đồng nhất. Điều gì xảy ra nếu
chúng ta dùng một BERT đa dụng thay thế, nhưng với công thức tinh chỉnh
cẩn thận hơn nhiều? Đó là mục đích của nhánh này — nó kết hợp hai kỹ
thuật nổi tiếng:

1. **Gradual Unfreezing** (Howard & Ruder, 2018 — ULMFiT):
   Unfreeze từng nhóm lớp một, bắt đầu từ đầu phân loại và đi xuống
   embeddings. Điều này ngăn gradient ngẫu nhiên từ đầu chưa huấn luyện
   phá hủy ngay lập tức các lớp dưới đã được pre-train.

2. **Layer-wise Learning Rate Decay (LLRD)** (Sun và cộng sự, 2019):
   Mỗi lớp đang hoạt động có learning rate khác nhau. Đầu (head) dùng toàn
   bộ `base_lr`; mỗi lớp bên dưới được nhân với `decay^depth`. Các lớp
   thấp hơn (tổng quát hơn) thay đổi rất chậm, trong khi các lớp cao hơn
   (chuyên biệt hơn) thay đổi nhanh hơn.

**Các nhóm lớp** (trên → dưới):
```
index  0 : đầu phân loại + BERT pooler       lr = 2e-5
index  1 : encoder layer 11                  lr = 2e-5 × 0.9^1 = 1.80e-5
index  2 : encoder layer 10                  lr = 2e-5 × 0.9^2 = 1.62e-5
...
index 12 : encoder layer  0                  lr = 2e-5 × 0.9^12 ≈ 5.6e-6
index 13 : token / position / type embeddings lr = 2e-5 × 0.9^13 ≈ 5.1e-6
```

**Lịch trình unfreeze** (epoch đánh số từ 1):
```
epoch  1 → chỉ nhóm 0 hoạt động (chỉ head)
epoch  2 → nhóm 0–1 hoạt động  (head + layer 11)
epoch  3 → nhóm 0–2 hoạt động
...
epoch 10 → nhóm 0–9 hoạt động
(Huấn luyện dừng ở epoch 10; 4 nhóm thấp nhất vẫn đóng băng.)
```

Ở mỗi epoch chúng ta **xây lại optimizer** để chỉ tham số có
`requires_grad=True` nhận gradient. Tham số trong các nhóm còn đóng băng
bị loại hoàn toàn khỏi optimizer, giúp tiết kiệm tính toán và bộ nhớ.

**Siêu tham số** (đặt ở đầu file `src/finetune_bert.py`):
```python
LLRD_BASE_LR    = 2e-5     # LR của head
LLRD_DECAY      = 0.9      # hệ số decay theo độ sâu
LABEL_SMOOTHING = 0.1      # chống quá tự tin trên tập nhỏ
BERT_EPOCHS     = 10       # thêm 1 epoch mỗi nhóm unfreeze
NUM_BERT_LAYERS = 12       # bert-base-uncased có 12 encoder layer
```

**Hàm mất mát.** Đối với **stance**, class weights được kết hợp với
label smoothing:

```python
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights).to(device),
    label_smoothing=LABEL_SMOOTHING,
)
```

Đối với **sentiment**, dùng CE với label smoothing đơn thuần (FPB cân
bằng hơn). Label smoothing thay nhãn one-hot `[0, 1, 0]` bằng một nhãn
mềm hơn `[0.033, 0.933, 0.033]` (với ε=0.1, 3 lớp). Điều này ngăn mô
hình trở nên quá tự tin một cách bệnh lý trên tập dữ liệu nhỏ.

**Chi tiết khác.** Không cần LR scheduler — gradual unfreezing đóng vai
trò warmup ngầm. Gradient clipping (`max_norm=1.0`) bảo vệ khỏi nổ
gradient khi nhiều lớp đột ngột hoạt động. Head có `weight_decay=0.0`
(để L2 không co bớt logits thiên về bias); các nhóm khác dùng
`WEIGHT_DECAY = 0.01`.

**Kết quả.**

| Bài toán | Accuracy | Macro-F1 |
|----------|----------|----------|
| Lập trường | 0.6512 | 0.6371 |
| Cảm xúc    | 0.9691 | 0.9533 |

Với **lập trường**, BERT-base + LLRD thực sự **vượt** FinBERT đơn nhiệm
(0.6371 vs 0.5988 macro-F1). Đây là phát hiện then chốt của nhánh này:
một công thức tinh chỉnh có kỷ luật có thể bù lại việc thiếu pre-training
theo miền, ở một bài toán mà lợi thế miền của FinBERT không rõ (stance
không nằm trong mục tiêu huấn luyện của FinBERT). Với **cảm xúc**, FinBERT
vẫn nhỉnh hơn một chút (0.9467 single-task vs 0.9533 LLRD — BERT LLRD
cạnh tranh tốt nhưng head cảm xúc có sẵn của FinBERT vẫn khó đánh bại).

Được lưu vào `models/bert_llrd_stance/` và `models/bert_llrd_sentiment/`.

---

## 10. Học Đa Nhiệm

### 10.1 Kiến Trúc Mô Hình (`src/multitask.py`)

```python
class MultiTaskFinBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("ProsusAI/finbert")  # dùng chung
        self.dropout = nn.Dropout(0.1)
        self.stance_head = nn.Linear(768, 3)     # riêng cho bài toán
        self.sentiment_head = nn.Linear(768, 3)  # riêng cho bài toán

    def forward(self, input_ids, attention_mask, task="stance"):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS]
        pooled = self.dropout(pooled)

        if task == "stance":
            return self.stance_head(pooled)
        else:
            return self.sentiment_head(pooled)
```

Tham số `task` hoạt động như một công tắc: cùng bộ mã hóa, đầu ra khác nhau.

### 10.2 Huấn Luyện Luân Phiên Batch

```python
while not (stance_done and sentiment_done):
    # Batch lập trường
    if not stance_done:
        batch = next(stance_iter)
        loss = _train_step(model, batch, stance_criterion, ..., task="stance")

    # Batch cảm xúc
    if not sentiment_done:
        batch = next(sentiment_iter)
        loss = _train_step(model, batch, sentiment_criterion, ..., task="sentiment")
```

Chúng ta xen kẽ batch: lập trường, cảm xúc, lập trường, cảm xúc, ...

Điều này đảm bảo bộ mã hóa dùng chung liên tục học từ cả hai loại văn bản
tài chính. Nếu huấn luyện tất cả batch lập trường trước, rồi tất cả cảm xúc,
mô hình có thể "quên" các mẫu lập trường khi học cảm xúc (gọi là **quên
thảm khốc - catastrophic forgetting**).

### 10.3 Hàm Mất Mát Riêng Biệt

```python
# Mất mát có trọng số cho lập trường (xử lý mất cân bằng lớp)
stance_criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.272, 1.365, 0.675])
)
# Mất mát tiêu chuẩn cho cảm xúc (các lớp cân bằng hơn)
sentiment_criterion = nn.CrossEntropyLoss()
```

Chúng ta chỉ dùng mất mát có trọng số cho lập trường vì bộ dữ liệu FOMC
có mất cân bằng lớp đáng kể. Financial PhraseBank cân bằng hơn.

### 10.4 Chọn Mô Hình

Cho đa nhiệm, chúng ta lưu mô hình với F1 **trung bình** tốt nhất trên cả
hai bài toán:

```python
avg_f1 = (val_stance_f1 + val_sentiment_f1) / 2
if avg_f1 > best_avg_f1:
    best_model_state = ...
```

Điều này ngăn tối ưu cho một bài toán mà hy sinh bài toán kia.

### 10.5 Kết Quả

| Bài toán | F1 Đơn nhiệm | F1 Đa nhiệm | Cải thiện |
|----------|-------------|-------------|-----------|
| Lập trường | 0.5988 | **0.6684** | +0.0696 |
| Cảm xúc | 0.9467 | **0.9772** | +0.0305 |

**Đa nhiệm cải thiện cả hai bài toán.** Bộ mã hóa dùng chung được hưởng lợi
từ việc nhìn thấy văn bản tài chính đa dạng hơn. Bài toán lập trường đặc
biệt được hưởng lợi vì nó có ít mẫu huấn luyện hơn — dữ liệu cảm xúc cung
cấp tín hiệu phụ trợ hữu ích, và mức nhảy ~7 điểm F1 so với FinBERT đơn
nhiệm là mức cải thiện lớn nhất trong toàn bộ chuỗi mô hình của dự án.

### 10.6 Phân Tích Lỗi

Về stance, các lỗi còn lại chủ yếu là các câu mơ hồ kiểu
"trung-lập-nhưng-nghiêng-nhẹ" (trích từ
`results/all_results_summary.json → stance_errors`):

- *"In light of increased uncertainties and muted inflation pressures..."*
  → dự đoán **hawkish**, thực ra là **dovish** (mô hình bám vào từ
  "uncertainties" thay vì "muted inflation pressures").
- *"With an increase in the target range at this meeting..."*
  → dự đoán **neutral**, thực ra là **hawkish** (ngôn ngữ có vẻ sự kiện
  che đi hàm ý hawkish của hành động được mô tả).
- *"Looking ahead, reports from retailer contacts ..."*
  → dự đoán **dovish**, thực ra là **neutral** (báo cáo dữ liệu vs. quan
  điểm chính sách).

Những lỗi này dễ hiểu. Các câu trung lập FOMC thường chứa **tín hiệu hỗn
hợp** — cả yếu tố diều hâu và bồ câu — khiến việc phân loại thực sự mơ hồ
ngay cả đối với chuyên gia con người.

Về cảm xúc, chỉ có 7 lỗi tổng cộng trên tập test 453 câu; phần lớn là
confusion "tích cực → tiêu cực" hoặc "trung lập → tích cực" ở các câu so
sánh lãi/lỗ mà người đọc cũng có thể diễn giải theo cả hai cách.

---

## 11. Phương Pháp Đánh Giá

### 11.1 Tại Sao Dùng Macro-F1?

Chúng ta dùng **Macro-F1** làm chỉ số chính thay vì accuracy vì bộ dữ liệu
có mất cân bằng lớp.

**Accuracy** có thể gây hiểu lầm: nếu 60% câu FOMC là trung lập, mô hình
dự đoán "trung lập" cho mọi thứ đạt 60% accuracy nhưng vô dụng.

**Điểm F1** cân bằng precision và recall:
```
Precision (Độ chính xác) = TP / (TP + FP)
Recall (Độ phủ)          = TP / (TP + FN)
F1                       = 2 × Precision × Recall / (Precision + Recall)
```

Trong đó:
- TP = True Positive (dự đoán đúng là dương)
- FP = False Positive (dự đoán sai là dương)
- FN = False Negative (bỏ sót dương)

**Macro-F1** lấy trung bình F1 trên tất cả các lớp một cách bình đẳng:
```
Macro-F1 = (F1_bồ_câu + F1_diều_hâu + F1_trung_lập) / 3
```

Điều này cho trọng số bằng nhau cho tất cả các lớp, bất kể tần suất.
Mô hình phải hoạt động tốt trên TẤT CẢ các lớp để đạt Macro-F1 cao.

### 11.2 Ma Trận Nhầm Lẫn (Confusion Matrix)

Ma trận nhầm lẫn cho thấy mô hình mắc lỗi ở đâu. Mỗi ô (i, j) chứa số
lượng mẫu có nhãn thực i được dự đoán là j.

```
                     Dự Đoán
              Bồ câu  Diều hâu  Trung lập
Thực  Bồ câu  [ 85      6       39 ]    ← 85 đúng, 6 nhầm với diều hâu
     Diều hâu [  5     85       31 ]    ← 31 dự đoán là trung lập
    Trung lập [ 55     70      120 ]    ← nhiều lỗi nhất: trung lập → bồ câu/diều hâu
```

Đường chéo cho thấy dự đoán đúng. Các ô ngoài đường chéo là lỗi.

Chúng ta tạo biểu đồ nhiệt dùng `seaborn.heatmap()` và lưu dưới dạng
file PNG trong `results/`.

### 11.3 Phân Tích Lỗi (`evaluate.py`)

```python
def error_analysis(texts, y_true, y_pred, label_names, top_n=20):
    errors = []
    for text, true, pred in zip(texts, y_true, y_pred):
        if true != pred:
            errors.append({
                "text": text,
                "true_label": label_names[true],
                "pred_label": label_names[pred],
                "error_type": f"{label_names[true]} → {label_names[pred]}",
            })
```

Hàm này thu thập tất cả mẫu bị phân loại sai và phân loại theo loại lỗi
(ví dụ: "bồ câu → diều hâu"). Kiểm tra các lỗi này cho thấy:
- **Câu mơ hồ** mà ngay cả con người cũng có thể bất đồng
- **Không khớp miền** khi mô hình thiếu kiến thức cụ thể
- **Thiên lệch hệ thống** (ví dụ: luôn dự đoán trung lập cho câu dài)

---

## 12. CLI, Demo, và Xuất Bản Lên HuggingFace

### 12.1 CLI (`cli.py`)

CLI cung cấp ba chế độ:

**Chế độ tương tác** (`python cli.py`):
```
>>> The Fed raised rates by 75 basis points
  LẬP TRƯỜNG: diều hâu (0.9000)
  CẢM XÚC: tích cực (0.9245)
```

**Một câu** (`python cli.py --text "..."`)

**Chế độ file** (`python cli.py --file input.txt`): Xử lý từng dòng.

Bên trong, nó:
1. Tải mô hình đa nhiệm từ `models/multitask_finbert/`
2. Tokenize văn bản đầu vào
3. Chạy qua bộ mã hóa dùng chung
4. Lấy dự đoán từ cả hai đầu bài toán
5. Áp dụng softmax để có điểm tin cậy

```python
logits = model(input_ids, attention_mask, task="stance")
probs = F.softmax(logits, dim=-1)  # chuyển logits thành xác suất
pred_idx = probs.argmax().item()   # chọn lớp có xác suất cao nhất
```

### 12.2 Demo Gradio (`demo.py`)

Gradio tạo giao diện web với code tối thiểu:

```python
with gr.Blocks() as demo:
    text_input = gr.Textbox(label="Văn Bản Tài Chính")
    stance_output = gr.Label(label="Lập Trường", num_top_classes=3)
    sentiment_output = gr.Label(label="Cảm Xúc", num_top_classes=3)

    submit_btn.click(fn=classify, inputs=text_input,
                     outputs=[stance_output, sentiment_output])
```

Khi người dùng nhập văn bản và nhấn "Phân Loại", Gradio gọi hàm `classify()`
và hiển thị kết quả dưới dạng thanh xác suất có nhãn.

Demo bao gồm 8 câu ví dụ mà người dùng có thể nhấp để thử.

### 12.3 Phân Tích Dữ Liệu (`data_analysis.py`)

Chạy `python data_analysis.py` tạo lại 10 biểu đồ phân tích trong thư mục
`analysis/` cộng với file JSON thống kê mức dataset. Các biểu đồ bao gồm:

- `class_distribution.png` — phân bố nhãn từng dataset.
- `text_length_distribution.png` — histogram độ dài câu theo task.
- `top_words_per_class.png` — top-k từ đặc trưng cho mỗi nhãn.
- `lexicon_coverage.png` — tỷ lệ kích hoạt từ điển LM theo lớp.
- `model_comparison.png` — macro-F1 so sánh tất cả mô hình × task.
- `per_class_f1_heatmap.png` — ma trận F1 theo lớp, trên tất cả mô hình.
- `performance_progression.png` — câu chuyện từ lexicon → TF-IDF → FinBERT → đa nhiệm.
- `multitask_improvement.png` — chênh lệch đơn nhiệm vs đa nhiệm FinBERT.
- `domain_pretraining_gap.png` — few-shot FinBERT vs BERT vs RoBERTa.
- `task_difficulty_gap.png` — macro-F1 stance-vs-sentiment theo họ mô hình.

Các biểu đồ này được dùng trực tiếp trong báo cáo và slide.

### 12.4 Xuất Bản Mô Hình Lên HuggingFace Hub (`push_to_hf.py`)

Sau khi huấn luyện xong, `python push_to_hf.py` upload các mô hình lên
HuggingFace Hub dưới namespace `Louisnguyen/*`. Nó push:

| Thư mục local | Repo remote |
|---------------|-------------|
| `models/finbert_stance/`      | `Louisnguyen/finbert-financial-stance` |
| `models/finbert_sentiment/`   | `Louisnguyen/finbert-financial-sentiment` |
| `models/bert_llrd_stance/`    | `Louisnguyen/bert-llrd-financial-stance` |
| `models/bert_llrd_sentiment/` | `Louisnguyen/bert-llrd-financial-sentiment` |
| `models/multitask_finbert/`   | `Louisnguyen/multitask-finbert-financial` |

Script cũng đính kèm file `results/*.json` tương ứng vào mỗi repo (đổi tên
thành `results.json`) để model card có thể trích dẫn đúng các chỉ số.

**Yêu cầu** biến môi trường `HF_TOKEN`:
```bash
export HF_TOKEN=hf_...
python push_to_hf.py
```

### 12.5 Tạo Báo Cáo / Slide (`create_presentation.py`)

Tạo các tài liệu báo cáo cuối bằng cách kết hợp các biểu đồ trong `analysis/`,
các chỉ số trong `results/all_results_summary.json`, và phần văn bản tường
thuật. Có thể chạy lại nhiều lần sau mỗi lần thử nghiệm mới.

---

## 13. Các Lỗi Gặp Phải và Cách Khắc Phục

### Lỗi 1: `trust_remote_code` Không Còn Được Hỗ Trợ

**Thông báo lỗi**:
```
RuntimeError: Dataset scripts are no longer supported, but found financial_phrasebank.py
```

**Nguyên nhân**: HuggingFace `datasets` v4.0+ đã loại bỏ hỗ trợ cho các
script tải Python. Bộ dữ liệu `takala/financial_phrasebank` gốc sử dụng
script tải tùy chỉnh.

**Cách sửa**: Chuyển sang phiên bản dựa trên Parquet:
```python
# Trước (lỗi):
load_dataset("takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True)

# Sau (hoạt động):
load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", "5768")
```

**Bài học**: Luôn dùng định dạng bộ dữ liệu mới nhất. HuggingFace đang
chuyển mọi thứ sang Parquet vì bảo mật (script tải có thể thực thi code
tùy ý).

### Lỗi 2: Tham Số `multi_class` Bị Loại Bỏ Khỏi LogisticRegression

**Thông báo lỗi**:
```
TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'
```

**Nguyên nhân**: scikit-learn 1.8+ đã loại bỏ tham số `multi_class`. Bộ giải
(solver) giờ tự động xác định chiến lược đa lớp.

**Cách sửa**: Loại bỏ tham số:
```python
# Trước:
LogisticRegression(multi_class="multinomial", solver="lbfgs")

# Sau:
LogisticRegression(solver="lbfgs")
```

**Bài học**: Phiên bản thư viện mới hơn có thể deprecated và loại bỏ tham số.
Luôn kiểm tra ghi chú phát hành khi dùng phiên bản mới nhất.

### Lỗi 3: Lấy Mẫu Few-Shot Mất Cột `label`

**Thông báo lỗi**:
```
ValueError: Column 'label' doesn't exist.
```

**Nguyên nhân**: Phép `groupby("label").apply(lambda x: x.sample(...))` trong
pandas 3.0+ tiêu thụ cột groupby, loại bỏ nó khỏi kết quả. Tham số
`include_groups=False` (thử đầu tiên) cũng loại bỏ nó.

**Cách sửa**: Thay thế groupby bằng vòng lặp tường minh:
```python
# Trước (lỗi trong pandas 3.0):
df.groupby("label").apply(lambda x: x.sample(n=k, random_state=SEED))

# Sau (hoạt động trong tất cả phiên bản pandas):
pieces = []
for label_val in sorted(df["label"].unique()):
    subset = df[df["label"] == label_val]
    pieces.append(subset.sample(n=min(k, len(subset)), random_state=SEED))
sampled = pd.concat(pieces, ignore_index=True)
```

**Bài học**: Hành vi `groupby().apply()` thay đổi theo phiên bản pandas.
Vòng lặp đơn giản đáng tin cậy hơn và dễ gỡ lỗi hơn.

### Lỗi 4: Vấn Đề Thiết Bị MPS Với HuggingFace Pipeline

**Vấn đề**: Hàm `pipeline()` của HuggingFace đôi khi lỗi trên MPS (GPU Apple
Silicon) do các phép toán không được hỗ trợ.

**Cách sửa**: Dùng `device=-1` (CPU) cho suy luận pipeline, chỉ dùng MPS cho
huấn luyện PyTorch trực tiếp:
```python
clf = pipeline("text-classification", model=FINBERT_MODEL, device=-1)
```

Đây là đánh đổi hiệu suất nhỏ — suy luận pipeline trên CPU đủ nhanh cho
các tập test nhỏ.

### Lỗi 5: Python 3.14 Không Tương Thích Với PyTorch

**Vấn đề**: Python hệ thống là 3.14.2, nhưng PyTorch không hỗ trợ Python 3.14.

**Cách sửa**: Tạo venv với Python 3.12:
```bash
/opt/homebrew/bin/python3.12 -m venv venv
```

**Bài học**: Luôn kiểm tra tính tương thích framework trước khi bắt đầu.
Dùng `venv` để cách ly dự án khỏi Python hệ thống.

---

## 14. Phân Tích Kết Quả

### 14.1 Bảng Kết Quả Đầy Đủ

Các số liệu dưới đây lấy trực tiếp từ `results/all_results_summary.json`.

| Mô hình | Acc Lập trường | F1 Lập trường | Acc Cảm xúc | F1 Cảm xúc |
|---------|---------------|---------------|-------------|------------|
| Từ điển LM (quy tắc)           | 0.4153 | 0.3885 | 0.6932 | 0.5315 |
| TF-IDF + LR                    | 0.6089 | 0.5873 | 0.8720 | 0.8232 |
| TF-IDF + SVM                   | 0.6331 | 0.6061 | 0.8940 | 0.8534 |
| TF-IDF (trigram) + LR          | 0.6109 | 0.5914 | 0.8786 | 0.8310 |
| TF-IDF + Từ điển LM            | 0.6109 | 0.5863 | 0.8543 | 0.8050 |
| FinBERT (zero-shot)            | 0.4980 | 0.4874 | 0.9735 | 0.9650 |
| FinBERT (few-shot k=16)        | 0.4859 | 0.4552 | 0.9801 | 0.9690 |
| BERT-base (few-shot k=16)      | 0.3790 | 0.3694 | 0.7461 | 0.6599 |
| RoBERTa-base (few-shot k=16)   | 0.3589 | 0.3489 | 0.7572 | 0.6439 |
| FinBERT (tinh chỉnh)           | 0.6129 | 0.5988 | 0.9669 | 0.9467 |
| BERT-base LLRD + Gradual UF    | 0.6512 | 0.6371 | 0.9691 | 0.9533 |
| **FinBERT Đa nhiệm**           | **0.6774** | **0.6684** | **0.9845** | **0.9772** |

### 14.2 Các Quan Sát Chính

**1. Huấn luyện sẵn theo miền là yếu tố quan trọng nhất cho cảm xúc.**
FinBERT (few-shot, chỉ 48 mẫu) đạt 96.9% F1 cho cảm xúc, đánh bại BERT-base
(66.0%) và RoBERTa-base (64.4%) với cùng dữ liệu. Đó là chênh lệch hơn 30
điểm phần trăm chỉ từ huấn luyện sẵn theo miền.

**2. Lập trường về cơ bản khó hơn cảm xúc.**
Ngay cả mô hình tốt nhất cũng chỉ đạt 66.8% F1 cho lập trường so với 97.7%
cho cảm xúc. Lập trường yêu cầu hiểu **lập trường chính sách tiền tệ ngầm
ẩn** — suy luận tinh tế vượt ra ngoài ý nghĩa từ bề mặt.

**3. Học đa nhiệm giúp cả hai bài toán — đặc biệt nhiều với stance.**
Mô hình đa nhiệm đánh bại tinh chỉnh đơn nhiệm trên cả hai bài toán:
- Lập trường: +7.0% F1 (0.5988 → 0.6684)
- Cảm xúc: +3.1% F1 (0.9467 → 0.9772)

Điều này xác nhận lập trường và cảm xúc tài chính là các bài toán liên
quan có lợi từ biểu diễn dùng chung. Mức tăng cho stance đặc biệt lớn vì
FOMC có ít dữ liệu gán nhãn hơn — task cảm xúc đóng vai trò cung cấp tín
hiệu giám sát phụ cho bộ encoder dùng chung.

**4. LLRD + Gradual Unfreezing là phương án thay thế thật sự cho
pre-training theo miền.**
Với **stance**, BERT-base dùng LLRD (F1 = 0.6371) thực sự vượt FinBERT
đơn nhiệm (F1 = 0.5988). Một mô hình đa dụng với công thức tinh chỉnh
đúng có thể đánh bại mô hình chuyên ngành dùng công thức tinh chỉnh
thông thường, ít nhất ở các task mà domain pretraining không trực tiếp
thấy phân bố nhãn.

**5. TF-IDF + SVM là baseline phi-nơ-ron tốt nhất, và cũng rất mạnh.**
89.4% accuracy cho sentiment với pipeline TF-IDF+SVM cho thấy cảm xúc
tài chính thường quy về sự hiện diện từ khóa. Mô hình nơ-ron thêm giá
trị chủ yếu cho các trường hợp mơ hồ.

**6. FinBERT zero-shot xuất sắc cho cảm xúc nhưng thất bại cho lập trường.**
FinBERT được huấn luyện cho cảm xúc, không phải lập trường. Dùng nhãn cảm
xúc làm proxy cho lập trường (tích cực→diều hâu) chỉ đạt 49.8% — gần như
ngẫu nhiên (33.3%). Điều này chứng minh lập trường và cảm xúc là hai bài
toán riêng biệt.

**7. Few-shot BERT/RoBERTa hoạt động kém hơn baseline TF-IDF.**
Với chỉ 48 mẫu huấn luyện, embedding transformer đóng băng không nắm bắt
đủ tín hiệu đặc thù bài toán. Baseline TF-IDF+SVM, huấn luyện trên 1700+
mẫu, thắng dễ dàng. Điều này cho thấy nhiều dữ liệu huấn luyện hơn có
thể bù đắp cho mô hình đơn giản hơn.

### 14.3 Câu Chuyện Tiến Triển Mô Hình

Kết quả kể một câu chuyện tiến triển rõ ràng:

```
Chỉ quy tắc (từ điển)            → 0.39 / 0.53  (stance / sentiment F1)
Baseline thống kê (LR)           → 0.59 / 0.82  (học từ dữ liệu)
Classical mạnh hơn (SVM)         → 0.61 / 0.85  (mô hình margin lớn hơn)
Mô hình miền, ít dữ liệu         → 0.46 / 0.97  (đúng mô hình cho sentiment,
                                                thiếu dữ liệu cho stance)
Mô hình miền, tinh chỉnh đầy đủ  → 0.60 / 0.95  (đúng mô hình, đúng dữ liệu)
Mô hình đa dụng + LLRD+Unfreeze  → 0.64 / 0.95  (công thức cẩn thận thu hẹp gap)
Mô hình miền đa nhiệm            → 0.67 / 0.98  (học chia sẻ ngôi đầu)
```

Mỗi bước thêm một thứ: khả năng học → kiến thức miền → thêm dữ liệu →
tinh chỉnh có kỷ luật → tín hiệu đa nhiệm chia sẻ.

---

## 15. Bài Học Rút Ra

### Cho Người Làm NLP

1. **Luôn bắt đầu với baseline.** TF-IDF + LR mất 5 giây để huấn luyện và
   cho biết bài toán khó như thế nào. TF-IDF + SVM thường mạnh hơn miễn
   phí và nên thử trước khi chuyển sang mô hình nơ-ron.

2. **Pre-training theo miền > kích thước mô hình, *với các task mà mô hình
   miền đã được huấn luyện*.** FinBERT thống trị sentiment nhưng chỉ vượt
   BERT-base khiêm tốn trên stance — lợi thế miền co lại khi task mục tiêu
   nằm ngoài mục tiêu pre-training.

3. **Tinh chỉnh có kỷ luật có thể bù đắp việc thiếu prior miền.** BERT-base
   với LLRD + Gradual Unfreezing đánh bại FinBERT đơn nhiệm trên stance,
   đơn giản bằng cách tinh chỉnh cẩn thận hơn.

4. **Học đa nhiệm là hiệu suất miễn phí.** Nếu bạn có các bài toán liên
   quan, huấn luyện chúng đồng thời không tốn thêm chi phí và cải thiện cả
   hai, với mức tăng lớn hơn đi vào task có ít dữ liệu hơn.

5. **Mất cân bằng lớp phải được xử lý.** Nếu không có mất mát có trọng số,
   mô hình lập trường dự đoán "trung lập" cho hầu hết đầu vào (dễ đạt 49%
   accuracy, khó đạt F1 cao).

6. **Chỉ số đánh giá quan trọng.** Accuracy có thể gây hiểu lầm với lớp
   mất cân bằng. Luôn báo cáo F1 theo lớp và Macro-F1.

### Cho Môn Học Này

- Code được tổ chức để chạy theo bước: `python run_experiments.py --step N`
- Tất cả kết quả được lưu dưới dạng JSON trong `results/` để phân tích dễ dàng
- CLI và demo Gradio cung cấp cách nhanh để test mô hình đã huấn luyện
- Ma trận nhầm lẫn trực quan hóa nơi mỗi mô hình gặp khó khăn

### Cho Công Việc Tương Lai

- **Thêm dữ liệu**: Bộ dữ liệu FOMC nhỏ (2,480 câu). Thu thập thêm thông
  báo Fed có thể cải thiện đáng kể phân loại lập trường.
- **Mô hình lớn hơn**: GPT-4 hoặc Claude có thể được đánh giá zero-shot để
  so sánh với FinBERT đã tinh chỉnh.
- **Chuyển giao xuyên miền**: Huấn luyện trên lập trường FOMC có giúp phân
  loại thông báo ECB (Ngân Hàng Trung Ương Châu Âu) hay BOJ (Ngân Hàng Nhật
  Bản) không?
- **Phân tích theo thời gian**: Các mẫu diều hâu/bồ câu có thay đổi theo
  thời gian không? Có thể huấn luyện trên dữ liệu FOMC cũ và test trên dữ
  liệu mới hơn.

---

## 16. Tham Chiếu Code Đầy Đủ — Giải Thích Từng Hàm

Phần này giải thích **mọi hàm và lớp** trong dự án, theo từng file.
Với mỗi hàm, chúng ta giải thích: mục đích, tham số đầu vào, giá trị trả
về, và logic bên trong.

---

### 16.1 `config.py` — Cấu Hình Trung Tâm

File này không chứa hàm nào — chỉ có các hằng số. Nó được import bởi
mọi file khác. Mọi thứ đều tập trung ở đây để khi bạn muốn thay đổi
(ví dụ: tăng batch size), bạn chỉ sửa MỘT chỗ.

```python
# Phát hiện thiết bị tính toán theo thứ tự ưu tiên
if torch.backends.mps.is_available():    # GPU Apple Silicon
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():          # GPU NVIDIA
    DEVICE = torch.device("cuda")
else:                                    # CPU nếu không có GPU
    DEVICE = torch.device("cpu")
```

**Tại sao kiểm tra MPS trước CUDA?** Vì chúng ta chạy trên MacBook M3 Max.
Nếu chạy trên cluster H200, nó sẽ tự động chọn CUDA.

```python
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_ID2LABEL = {i: l for i, l in enumerate(SENTIMENT_LABELS)}
# → {0: "negative", 1: "neutral", 2: "positive"}
SENTIMENT_LABEL2ID = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
# → {"negative": 0, "neutral": 1, "positive": 2}
```

Hai dict `ID2LABEL` và `LABEL2ID` giúp chuyển đổi qua lại giữa số nguyên
(mô hình dùng) và chuỗi (con người đọc).

---

### 16.2 `src/data_loader.py` — Tải Và Xử Lý Dữ Liệu

#### `load_financial_phrasebank()`

**Mục đích**: Tải bộ dữ liệu Financial PhraseBank từ HuggingFace, chuẩn
hóa cột, và chia thành train/val/test.

**Tham số**: Không có (dùng hằng số từ config).

**Giá trị trả về**: `DatasetDict` với 3 khóa: `"train"`, `"val"`, `"test"`.
Mỗi split có các cột: `text` (str), `label` (int), `label_name` (str).

**Logic bên trong**:
```python
# 1. Tải từ HuggingFace
ds = load_dataset(FPB_DATASET_NAME, FPB_SUBSET)

# 2. Gộp tất cả split có sẵn thành một DataFrame
#    (bộ dữ liệu gốc chỉ có train/test, không có val)
frames = []
for split_name in ds:
    frames.append(ds[split_name].to_pandas())
df = pd.concat(frames, ignore_index=True)

# 3. Đổi tên cột "sentence" → "text" cho nhất quán
df = df.rename(columns={"sentence": "text"})

# 4. Thêm cột label_name để dễ đọc
df["label_name"] = df["label"].map({0: "negative", 1: "neutral", 2: "positive"})

# 5. Chia phân tầng: đảm bảo tỷ lệ lớp giống nhau trong mỗi split
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"])
train_df, val_df = train_test_split(train_df, test_size=0.125, stratify=...)
# 0.125 = 0.1 / 0.8 → vì train đã là 80%, lấy 12.5% của 80% = 10% tổng
```

**Tại sao gộp rồi chia lại?** Vì bộ dữ liệu gốc không có tập validation.
Chúng ta cần val để chọn mô hình tốt nhất trong quá trình huấn luyện.

---

#### `load_fomc_dataset()`

**Mục đích**: Tải bộ dữ liệu FOMC Hawkish-Dovish. Có cơ chế dự phòng: thử
HuggingFace trước, nếu thất bại thì tải từ file CSV cục bộ.

**Logic**:
```python
try:
    ds = load_dataset(FOMC_DATASET_NAME)    # thử HuggingFace
except Exception as e:
    return _load_fomc_local()               # dự phòng: file cục bộ
```

Tại sao cần dự phòng? Vì mạng có thể bị lỗi hoặc HuggingFace có thể thay
đổi tên bộ dữ liệu trong tương lai.

---

#### `_load_fomc_local()`

**Mục đích**: Hàm dự phòng — tải FOMC từ file CSV trong thư mục `data/`.
Tên bắt đầu bằng `_` theo quy ước Python nghĩa là "hàm nội bộ, không nên
gọi trực tiếp từ bên ngoài module".

---

#### `_process_fomc_df(df)`

**Mục đích**: Chuẩn hóa DataFrame FOMC — tìm đúng cột text và label,
chuyển nhãn chuỗi thành số, chia tập dữ liệu.

**Logic đáng chú ý**:
```python
# Tìm cột text — có thể có tên khác nhau tùy phiên bản dữ liệu
if "sentence" in df.columns:
    df = df.rename(columns={"sentence": "text"})

# Chuyển nhãn chuỗi sang số nếu cần
if df["label"].dtype == object:           # object = kiểu chuỗi trong pandas
    label_map = {"dovish": 0, "hawkish": 1, "neutral": 2}
    df["label"] = df["label"].str.strip().str.lower().map(label_map)
    # .strip() loại bỏ khoảng trắng thừa
    # .lower() chuyển thành chữ thường
    # .map() áp dụng dict ánh xạ
```

---

#### `get_few_shot_subset(dataset_split, k=16)`

**Mục đích**: Lấy mẫu k ví dụ từ MỖI lớp cho học few-shot.
Với k=16 và 3 lớp → trả về 48 mẫu.

**Tham số**:
- `dataset_split`: Một split của HuggingFace Dataset (ví dụ: `fomc["train"]`)
- `k`: Số mẫu mỗi lớp (mặc định 16)

**Logic**:
```python
pieces = []
for label_val in sorted(df["label"].unique()):  # lặp qua: 0, 1, 2
    subset = df[df["label"] == label_val]        # lọc mẫu của lớp này
    pieces.append(subset.sample(n=min(k, len(subset)), random_state=SEED))
    # .sample(n=k) lấy ngẫu nhiên k mẫu
    # min(k, len(subset)) phòng trường hợp lớp có ít hơn k mẫu
sampled = pd.concat(pieces, ignore_index=True)  # gộp lại thành một DataFrame
```

**Tại sao dùng vòng lặp thay vì groupby?** Vì `groupby().apply()` trong
pandas 3.0 có bug làm mất cột label (xem mục 13, Lỗi 3).

---

#### `compute_class_weights(dataset_split, num_classes=3)`

**Mục đích**: Tính trọng số lớp cho hàm mất mát có trọng số.
Lớp ít mẫu → trọng số cao hơn → mô hình chú ý nhiều hơn.

**Công thức**: `weight_k = N / (C × n_k)`
- N = tổng mẫu, C = số lớp, n_k = mẫu trong lớp k

**Ví dụ** với FOMC (1736 mẫu train):
```
dovish (455 mẫu):  1736 / (3 × 455) = 1.272
hawkish (424 mẫu): 1736 / (3 × 424) = 1.365
neutral (857 mẫu): 1736 / (3 × 857) = 0.675
```

Trung lập có trọng số thấp nhất vì nó có nhiều mẫu nhất.

---

#### `_print_split_stats(name, splits, label_names)`

**Mục đích**: In thống kê bộ dữ liệu ra console — kích thước mỗi split
và phân bố lớp. Hàm tiện ích để kiểm tra dữ liệu nhanh.

---

### 16.3 `src/evaluate.py` — Chỉ Số Đánh Giá

#### `compute_metrics(y_true, y_pred, label_names)`

**Mục đích**: Tính tất cả chỉ số đánh giá cùng lúc.

**Tham số**:
- `y_true`: Danh sách nhãn thực (số nguyên)
- `y_pred`: Danh sách nhãn dự đoán (số nguyên)
- `label_names`: Danh sách tên nhãn (chuỗi)

**Giá trị trả về**: Dict chứa:
```python
{
    "accuracy": 0.6593,
    "macro_f1": 0.6478,
    "per_class_f1": {"dovish": 0.6182, "hawkish": 0.6050, "neutral": 0.7202},
    "report": "... bảng classification_report đầy đủ ..."
}
```

**Bên trong**:
```python
# f1_score với average=None trả về F1 RIÊNG cho từng lớp
per_class_f1 = f1_score(y_true, y_pred, average=None, ...)
# → [0.6182, 0.6050, 0.7202]

# f1_score với average="macro" lấy trung bình bình đẳng
macro_f1 = f1_score(y_true, y_pred, average="macro", ...)
# → (0.6182 + 0.6050 + 0.7202) / 3 = 0.6478

# zero_division=0: nếu một lớp không có dự đoán nào, F1 = 0 thay vì lỗi
```

---

#### `print_classification_report(metrics, model_name, task_name)`

**Mục đích**: In kết quả đánh giá đẹp ra console. Hàm thuần hiển thị,
không tính toán gì.

---

#### `plot_confusion_matrix(y_true, y_pred, label_names, model_name, task_name, save_dir=None)`

**Mục đích**: Tạo và lưu biểu đồ nhiệt ma trận nhầm lẫn.

**Logic**:
```python
# 1. Tính ma trận nhầm lẫn bằng scikit-learn
cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
# cm là mảng 2D, ví dụ:
# [[85,  6, 39],     ← dovish: 85 đúng, 6 nhầm hawk, 39 nhầm neutral
#  [ 5, 85, 31],     ← hawkish: 85 đúng
#  [55, 70, 120]]    ← neutral: 120 đúng

# 2. Vẽ bằng seaborn
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ...)
# annot=True: hiện số trong mỗi ô
# fmt="d": định dạng số nguyên (không có chữ số thập phân)
# cmap="Blues": bảng màu xanh (ô tối = giá trị cao)

# 3. Lưu file
fig.savefig(path, dpi=150)     # dpi=150 cho độ phân giải tốt
plt.close(fig)                  # đóng figure để giải phóng bộ nhớ
```

**Tại sao `matplotlib.use("Agg")`?** Ở đầu file, chúng ta đặt backend là
"Agg" (Anti-Grain Geometry) — đây là backend không tương tác, chỉ lưu file.
Nếu không đặt, matplotlib có thể cố mở cửa sổ đồ họa, gây lỗi trên server
hoặc khi chạy qua SSH.

---

#### `error_analysis(texts, y_true, y_pred, label_names, top_n=20)`

**Mục đích**: Thu thập và phân tích các mẫu bị phân loại sai.

**Giá trị trả về**:
- `error_df`: DataFrame chứa các mẫu sai, sắp xếp theo độ dài văn bản
  (câu dài thường mơ hồ hơn → thú vị hơn để phân tích)
- `error_counts`: Dict đếm từng loại lỗi, ví dụ `{"neutral → hawkish": 49}`

---

#### `save_results(results_dict, filename)`

**Mục đích**: Lưu dict kết quả thành file JSON trong thư mục `results/`.
`default=str` trong `json.dump` xử lý các kiểu dữ liệu không serialize
được (ví dụ: numpy float → string).

---

### 16.4 `src/baseline.py` — Mô Hình Cơ Sở

#### `build_baseline_pipeline()`

**Mục đích**: Tạo pipeline scikit-learn gồm TF-IDF vectorizer + Logistic
Regression. Trả về pipeline chưa huấn luyện.

**Tại sao dùng Pipeline?** Pipeline đảm bảo:
1. `fit()` chỉ gọi trên dữ liệu train (tránh rò rỉ dữ liệu)
2. `transform()` áp dụng cùng biến đổi cho test
3. Một object duy nhất quản lý toàn bộ quy trình

---

#### `train_and_evaluate_baseline(train_split, test_split, label_names, task_name)`

**Mục đích**: Huấn luyện baseline và đánh giá trên tập test. Đây là hàm
"một nút" — gọi nó là xong.

**Luồng xử lý**:
```python
# 1. Trích xuất text và label từ HuggingFace Dataset
train_texts = train_split["text"]     # → list các chuỗi
train_labels = train_split["label"]   # → list các số nguyên

# 2. Xây dựng, huấn luyện, dự đoán
pipeline = build_baseline_pipeline()
pipeline.fit(train_texts, train_labels)      # TF-IDF fit + LR fit
predictions = pipeline.predict(test_texts)   # TF-IDF transform + LR predict

# 3. Đánh giá
metrics = compute_metrics(test_labels, predictions, label_names)
plot_confusion_matrix(...)
save_results(...)
```

**Giá trị trả về**: `(metrics_dict, pipeline)` — cả chỉ số lẫn pipeline
đã huấn luyện (để dùng lại nếu cần).

---

### 16.5 `src/lexicon.py` — Phân Loại Dựa Trên Từ Điển

#### `_tokenize(text)`

**Mục đích**: Tách văn bản thành các token (từ) đơn giản.

```python
def _tokenize(text):
    return re.findall(r'\b[a-z]+\b', text.lower())
```

- `text.lower()`: chuyển thành chữ thường
- `re.findall(r'\b[a-z]+\b', ...)`: tìm tất cả các từ chỉ chứa chữ cái
- `\b` = ranh giới từ (word boundary)
- `[a-z]+` = một hoặc nhiều chữ cái thường
- Kết quả: `"The Fed raised rates!"` → `["the", "fed", "raised", "rates"]`

**Tại sao không dùng tokenizer phức tạp hơn?** Vì lexicon chỉ cần so khớp
từ đơn. Tokenizer đơn giản nhanh hơn và đủ chính xác cho mục đích này.

---

#### `extract_lexicon_features(texts)`

**Mục đích**: Trích xuất 8 đặc trưng số từ mỗi văn bản dựa trên từ điển.

**Tham số**: `texts` — danh sách các chuỗi văn bản.

**Giá trị trả về**: Mảng numpy kích thước `(n_texts, 8)`.

**Logic cho mỗi văn bản**:
```python
tokens = _tokenize(text)          # tách từ
total = max(len(tokens), 1)       # tránh chia cho 0

# Đếm từ thuộc mỗi nhóm bằng set lookup (O(1) cho mỗi tra cứu)
pos_count = sum(1 for t in tokens if t in LM_POSITIVE)
neg_count = sum(1 for t in tokens if t in LM_NEGATIVE)
# ... tương tự cho uncertainty, hawkish, dovish

# Tính chỉ số chuẩn hóa
net_sentiment = (pos_count - neg_count) / total
net_stance = (hawk_count - dove_count) / total
```

**Tại sao dùng `set` cho danh sách từ?** Tra cứu `t in set` là O(1)
(hằng số), trong khi `t in list` là O(n) (tuyến tính). Với ~230 từ
tiêu cực và hàng nghìn token, set nhanh hơn nhiều.

---

#### `lexicon_rule_based(test_split, label_names, task_name)`

**Mục đích**: Phân loại dựa trên quy tắc thuần túy — KHÔNG có huấn luyện.

**Logic phân loại**:
```python
if task_name == "sentiment":
    # Dùng net_sentiment (tích cực - tiêu cực, chuẩn hóa)
    if ns > 0.02:   y_pred = 2   # positive
    elif ns < -0.02: y_pred = 0   # negative
    else:           y_pred = 1   # neutral
else:
    # Dùng số từ hawkish vs dovish
    if hawk > dove:  y_pred = 1   # hawkish
    elif dove > hawk: y_pred = 0   # dovish
    else:            y_pred = 2   # neutral
```

**Ngưỡng 0.02** cho sentiment được chọn thực nghiệm. Quá thấp → quá
nhạy, quá cao → mọi thứ thành trung lập.

---

#### `lexicon_plus_tfidf(train_split, test_split, label_names, task_name)`

**Mục đích**: Kết hợp TF-IDF + 8 đặc trưng từ điển → huấn luyện LR.

**Bước quan trọng — Ghép đặc trưng**:
```python
# TF-IDF cho vectơ thưa 50,000 chiều
train_tfidf = tfidf.fit_transform(train_texts)  # (1736, 50000) sparse

# Lexicon cho mảng dày đặc 8 chiều
train_lex = extract_lexicon_features(train_texts)  # (1736, 8) dense

# Chuẩn hóa lexicon để có cùng tỷ lệ với TF-IDF
scaler = StandardScaler()
train_lex_scaled = scaler.fit_transform(train_lex)
# StandardScaler: (x - mean) / std → trung bình 0, độ lệch chuẩn 1

# Ghép hai ma trận: 50,000 + 8 = 50,008 chiều
train_combined = hstack([train_tfidf, csr_matrix(train_lex_scaled)])
# hstack = horizontal stack (ghép ngang)
# csr_matrix() chuyển mảng dày đặc thành ma trận thưa
```

**Tại sao cần StandardScaler?** Giá trị TF-IDF nằm trong khoảng [0, 1].
Giá trị lexicon (ví dụ: word count) có thể là 0-20. Nếu không chuẩn hóa,
LR sẽ bỏ qua đặc trưng lexicon vì chúng có cùng tỷ lệ với TF-IDF.

---

#### `run_lexicon_experiments(fomc_splits, fpb_splits)`

**Mục đích**: Chạy tất cả thí nghiệm lexicon — gọi 4 hàm trên.
Hàm "điều phối" — không có logic riêng, chỉ gọi các hàm khác.

---

### 16.6 `src/pretrained_eval.py` — Đánh Giá Mô Hình Sẵn

#### `evaluate_finbert_native(test_split, task_name="sentiment")`

**Mục đích**: Đánh giá FinBERT dùng lớp phân loại có sẵn (zero-shot).

**Logic đáng chú ý**:
```python
clf = pipeline(
    "text-classification",
    model=FINBERT_MODEL,
    device=-1,           # CPU — pipeline không ổn định trên MPS
    top_k=None,          # trả về điểm cho TẤT CẢ lớp, không chỉ top-1
)

# Xử lý batch để nhanh hơn xử lý từng câu
for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]
    results = clf(batch)         # → list of list of dicts
    for r in results:
        best = max(r, key=lambda x: x["score"])  # chọn lớp có điểm cao nhất
        pred_label = best["label"].lower()       # "Positive" → "positive"
```

**Ánh xạ nhãn cho stance**:
Khi dùng FinBERT (vốn phân loại cảm xúc) cho lập trường, chúng ta ánh xạ:
- "negative" → dovish (tin tiêu cực thường liên quan nới lỏng)
- "positive" → hawkish (tin tích cực thường liên quan thắt chặt)
- "neutral" → neutral

Đây chỉ là proxy thô — và kết quả kém (49.8%) chứng minh cảm xúc ≠ lập trường.

---

#### `class FewShotClassifier(nn.Module)`

**Mục đích**: Bộ phân loại tuyến tính đơn giản cho few-shot learning.

```python
class FewShotClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),                    # tắt 10% nơ-ron ngẫu nhiên
            nn.Linear(hidden_size, num_labels), # 768 → 3
        )

    def forward(self, x):
        return self.classifier(x)   # x: (batch, 768) → logits: (batch, 3)
```

**nn.Sequential**: Container xếp các layer nối tiếp. Đầu ra layer trước
là đầu vào layer sau.

**nn.Dropout(0.1)**: Trong quá trình huấn luyện, ngẫu nhiên đặt 10% giá
trị về 0. Ngăn mô hình phụ thuộc quá mức vào bất kỳ feature nào. Tự động
tắt khi `model.eval()`.

**nn.Linear(768, 3)**: Phép biến đổi tuyến tính `y = Wx + b` với
W: (3, 768), b: (3,). Học cách ánh xạ từ không gian embedding 768 chiều
sang 3 lớp.

---

#### `_encode_texts(tokenizer, model, texts, device)`

**Mục đích**: Chuyển danh sách văn bản thành embedding [CLS] dùng mô hình
đã đóng băng.

**Logic từng bước**:
```python
model.eval()    # tắt dropout, batch normalization chế độ inference
model.to(device)

for i in range(0, len(texts), BATCH_SIZE):
    batch = texts[i : i + BATCH_SIZE]

    # Tokenize: văn bản → token IDs + attention mask
    inputs = tokenizer(
        batch,
        padding=True,         # đệm câu ngắn cho bằng câu dài nhất trong batch
        truncation=True,      # cắt câu dài hơn max_length
        max_length=128,
        return_tensors="pt",  # trả về PyTorch tensors (không phải list)
    ).to(device)

    # Forward pass không tính gradient (tiết kiệm bộ nhớ, nhanh hơn)
    with torch.no_grad():
        outputs = model(**inputs)
        # outputs.last_hidden_state: (batch, seq_len, 768)
        # Lấy token đầu tiên ([CLS]) của mỗi câu:
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        all_embeddings.append(cls_emb.cpu())  # chuyển về CPU để ghép

return torch.cat(all_embeddings, dim=0)  # ghép tất cả batch: (n_texts, 768)
```

**`torch.no_grad()`**: Tắt hệ thống autograd. Bình thường PyTorch ghi nhớ
mọi phép tính để tính gradient (backprop). Khi chỉ suy luận, không cần
gradient → tiết kiệm ~50% bộ nhớ GPU và nhanh hơn ~20%.

---

#### `evaluate_few_shot(model_name, train_split, test_split, label_names, task_name, k=16)`

**Mục đích**: Đánh giá few-shot — huấn luyện linear probe trên embedding
đóng băng.

**Luồng đầy đủ**:
```python
# 1. Tải mô hình base (không có lớp phân loại)
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

# 2. Lấy 48 mẫu (16 × 3 lớp)
few_shot_data = get_few_shot_subset(train_split, k=k)

# 3. Mã hóa thành embedding
train_embs = _encode_texts(tokenizer, base_model, few_shot_data["text"], device)
# train_embs: (48, 768)
test_embs = _encode_texts(tokenizer, base_model, test_split["text"], device)
# test_embs: (496, 768) cho FOMC

# 4. Giải phóng bộ nhớ GPU (mô hình lớn ~440MB)
base_model.cpu()
del base_model
torch.mps.empty_cache()

# 5. Huấn luyện linear probe 200 epochs (trên CPU, nhanh vì chỉ 48 mẫu)
classifier = FewShotClassifier(768, 3)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
# Adam: tốc độ học thích ứng, lr=1e-3 là mặc định phổ biến

for epoch in range(200):
    logits = classifier(train_embs)          # (48, 3)
    loss = criterion(logits, train_labels)   # cross-entropy loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 6. Dự đoán
test_logits = classifier(test_embs)           # (496, 3)
y_pred = test_logits.argmax(dim=1).numpy()    # chọn lớp có logit cao nhất
```

**Tại sao 200 epochs?** Với chỉ 48 mẫu và 1 layer tuyến tính, huấn luyện
cực nhanh (~0.1 giây cho 200 epochs). Nhiều epoch đảm bảo hội tụ.

**Tại sao lr=1e-3 cho probe nhưng 2e-5 cho fine-tuning?** Probe chỉ có
~2,300 tham số (768×3 + 3) → cần tốc độ học cao để hội tụ nhanh.
Fine-tuning có 110 triệu tham số đã huấn luyện sẵn → cần tốc độ học
thấp để không phá hủy chúng.

---

#### `run_all_pretrained_evaluations(fomc_splits, fpb_splits)`

**Mục đích**: Điều phối tất cả đánh giá pretrained — zero-shot FinBERT
trên cả hai bộ dữ liệu, và few-shot cho 3 mô hình × 2 bài toán = 8 thí
nghiệm.

---

### 16.7 `src/finetune_fineBert.py` — Tinh Chỉnh FinBERT Đơn Nhiệm

#### `class TextClassificationDataset(TorchDataset)`

**Mục đích**: Bọc dữ liệu text để PyTorch DataLoader có thể tạo batch.

```python
def __len__(self):
    return len(self.texts)     # DataLoader cần biết tổng số mẫu

def __getitem__(self, idx):    # DataLoader gọi khi cần mẫu thứ idx
    encoding = self.tokenizer(
        self.texts[idx],
        padding="max_length",   # LUÔN đệm đến 128 token
        truncation=True,
        max_length=self.max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        # squeeze(0): loại bỏ chiều batch thừa
        # tokenizer trả về (1, seq_len), squeeze → (seq_len,)
        # DataLoader sẽ tự thêm chiều batch khi gộp
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        # torch.long = int64, cần cho cross-entropy loss
    }
```

**Tại sao `padding="max_length"` thay vì `padding=True`?**
- `padding=True` đệm đến câu dài nhất trong batch → mỗi batch có độ dài
  khác nhau → không thể cache/optimize
- `padding="max_length"` luôn đệm đến 128 → mọi mẫu cùng kích thước →
  DataLoader hiệu quả hơn

---

#### `finetune_finbert(train_split, val_split, test_split, label_names, task_name, use_weighted_loss=True)`

**Mục đích**: Tinh chỉnh toàn bộ FinBERT trên một bài toán.

**Đây là hàm lớn nhất — giải thích từng phần**:

```python
# --- Tải mô hình ---
model = AutoModelForSequenceClassification.from_pretrained(
    FINBERT_MODEL,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,  # ← QUAN TRỌNG
)
```

`ignore_mismatched_sizes=True`: FinBERT gốc có lớp phân loại cho 3 nhãn
sentiment. Chúng ta cũng cần 3 nhãn nhưng cho bài toán khác. Flag này
bảo model loader tạo lớp phân loại MỚI với trọng số ngẫu nhiên, giữ
nguyên bộ mã hóa.

```python
# --- Tạo DataLoaders ---
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
# shuffle=True: xáo trộn dữ liệu mỗi epoch → mô hình không ghi nhớ thứ tự

val_loader = DataLoader(val_ds, batch_size=32)
# KHÔNG shuffle val/test → kết quả nhất quán giữa các lần chạy
```

```python
# --- Optimizer ---
optimizer = torch.optim.AdamW(
    model.parameters(), lr=2e-5, weight_decay=0.01
)
```

**AdamW** (Adam with Weight Decay): Cải tiến của Adam optimizer.
- Adam: tốc độ học thích ứng cho mỗi tham số, dùng momentum
- W (weight decay): thêm chính quy hóa L2, giúp ngăn trọng số quá lớn
- `weight_decay=0.01`: mỗi bước, trọng số được nhân với (1 - 0.01 × lr)

```python
# --- Learning Rate Scheduler ---
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),   # 10% bước đầu
    num_training_steps=total_steps,
)
```

Scheduler thay đổi tốc độ học MỖI BƯỚC (không phải mỗi epoch):
- Bước 1→10%: tăng tuyến tính từ 0 → 2e-5 (khởi động)
- Bước 10%→100%: giảm tuyến tính từ 2e-5 → 0 (suy giảm)

```python
# --- Vòng lặp huấn luyện ---
for epoch in range(5):
    model.train()      # bật dropout, batch norm chế độ huấn luyện

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)       # chuyển lên GPU
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.logits: (batch_size, 3) — điểm thô chưa softmax

        loss = criterion(outputs.logits, labels)
        # Cross-entropy tự áp dụng softmax bên trong

        optimizer.zero_grad()    # xóa gradient cũ (PyTorch tích lũy gradient!)
        loss.backward()          # tính gradient qua backpropagation

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # Cắt gradient: nếu ‖g‖ > 1.0, scale g ← g × 1.0/‖g‖
        # Ngăn "gradient explosion" khi loss đột ngột tăng

        optimizer.step()         # cập nhật trọng số: w ← w - lr × gradient
        scheduler.step()         # điều chỉnh learning rate
```

```python
# --- Lưu mô hình tốt nhất ---
if val_metrics["macro_f1"] > best_val_f1:
    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # .cpu(): chuyển tensor về CPU (để không chiếm GPU memory)
    # .clone(): tạo bản sao (không phải reference)
    # state_dict(): dict chứa tất cả tham số, ví dụ:
    # {"bert.encoder.layer.0.attention.self.query.weight": tensor(...), ...}
```

---

#### `_evaluate_model(model, dataloader, label_names, device)`

**Mục đích**: Wrapper — gọi `_get_predictions` rồi `compute_metrics`.

#### `_get_predictions(model, dataloader, device)`

**Mục đích**: Chạy mô hình trên DataLoader, trả về `(y_true, y_pred)`.

```python
model.eval()       # tắt dropout
with torch.no_grad():   # tắt gradient computation
    for batch in dataloader:
        outputs = model(input_ids=..., attention_mask=...)
        preds = outputs.logits.argmax(dim=-1).cpu().tolist()
        # argmax(dim=-1): chọn chỉ số lớp có logit cao nhất
        # dim=-1 = chiều cuối cùng = chiều 3 lớp
        # .cpu(): chuyển về CPU
        # .tolist(): tensor → Python list
```

---

### 16.8 `src/finetune_bert.py` — BERT-base LLRD + Gradual Unfreezing

#### `_build_layer_groups(model)`

**Mục đích**: Chia tham số của `model` thành 14 nhóm có thứ tự, trên →
dưới. Index 0 là đầu phân loại + BERT pooler; index 1–12 là các encoder
layer 11..0; index 13 là token/position/type embeddings.

Với mỗi nhóm nó tính LR đã scale theo LLRD:
```python
lr = LLRD_BASE_LR * (LLRD_DECAY ** (depth + 1))
```
trong đó `depth` là khoảng cách nhóm đó so với head. Head nhận toàn bộ
`LLRD_BASE_LR = 2e-5`; mỗi layer bên dưới nhân với 0.9.

#### `_build_optimizer(model, epoch)`

**Mục đích**: Xây dựng optimizer AdamW mới mỗi epoch để thực hiện
**gradual unfreezing**. Đóng băng tất cả trước, rồi unfreeze
`min(epoch, 14)` nhóm trên cùng. Chỉ nhóm đang hoạt động được đưa vào
`param_groups` của optimizer.

Mẹo: không weight decay cho head (`wd = 0.0`) để logits thiên về bias
không bị co bởi L2; các nhóm khác dùng `WEIGHT_DECAY = 0.01`.

#### `_train_one_epoch(model, loader, criterion, optimizer, device)`

**Mục đích**: Vòng huấn luyện chuẩn cho một epoch, có gradient clipping
ở `max_norm=1.0` để bảo vệ khỏi nổ gradient khi một nhóm layer vừa được
unfreeze bắt đầu cập nhật.

#### `finetune_bert_llrd(train_split, val_split, test_split, label_names, task_name)`

**Mục đích**: Entry point chính. Khi `task_name == "stance"` nó dùng
cross-entropy có *class weights* + label smoothing; khi sentiment nó
dùng CE với label smoothing đơn thuần.

Log huấn luyện in mỗi epoch một dòng gồm: số epoch, tên nhóm sâu nhất
đang hoạt động, số tham số trainable, train loss, val accuracy, val
macro-F1. Checkpoint tốt nhất (theo val macro-F1) được giữ trong bộ nhớ
và khôi phục trước đánh giá test.

Kết thúc hàm lưu:
- `models/bert_llrd_{task}/` — mô hình + tokenizer định dạng HuggingFace
- `results/finetune_bert_llrd_{task}.json` — chỉ số test
- `results/BERT_base_LLRD_{task}_cm.png` — ma trận nhầm lẫn

### 16.9 `src/multitask.py` — Học Đa Nhiệm

#### `class MultiTaskFinBERT(nn.Module)`

Đã giải thích chi tiết ở [Mục 10.1](#101-kiến-trúc-mô-hình-srcmultitaskpy).
Bổ sung:

```python
self.encoder = AutoModel.from_pretrained(FINBERT_MODEL)
hidden_size = self.encoder.config.hidden_size  # 768
```

`self.encoder.config` là đối tượng cấu hình chứa thông tin kiến trúc:
hidden_size=768, num_attention_heads=12, num_hidden_layers=12, v.v.

---

#### `train_multitask(fomc_splits, fpb_splits)`

**Mục đích**: Huấn luyện mô hình đa nhiệm. Hàm lớn nhất trong dự án.

**Phần luân phiên batch** (giải thích chi tiết):
```python
stance_iter = iter(stance_train_loader)      # tạo iterator
sentiment_iter = iter(sentiment_train_loader)
stance_done = False
sentiment_done = False

while not (stance_done and sentiment_done):
    # Thử lấy batch lập trường
    if not stance_done:
        try:
            batch = next(stance_iter)   # lấy batch tiếp theo
            loss = _train_step(..., task="stance")
        except StopIteration:           # hết dữ liệu lập trường
            stance_done = True

    # Thử lấy batch cảm xúc
    if not sentiment_done:
        try:
            batch = next(sentiment_iter)
            loss = _train_step(..., task="sentiment")
        except StopIteration:
            sentiment_done = True
```

**StopIteration**: Python raise ngoại lệ này khi iterator hết phần tử.
`next()` cố lấy phần tử tiếp, nếu hết → StopIteration → đánh dấu done.

Hai bộ dữ liệu có kích thước khác nhau (FOMC: 55 batch, FPB: 50 batch).
Khi một cái hết trước, cái kia tiếp tục chạy đến hết.

---

#### `_train_step(model, batch, criterion, optimizer, scheduler, device, task)`

**Mục đích**: Thực hiện MỘT bước huấn luyện trên MỘT batch.
Tách riêng để dùng chung cho cả hai bài toán.

```python
logits = model(input_ids=..., attention_mask=..., task=task)
# task="stance" → dùng stance_head
# task="sentiment" → dùng sentiment_head
loss = criterion(logits, labels)
optimizer.zero_grad()
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()
return loss.item()  # .item() chuyển tensor 0-chiều → Python float
```

---

#### `_evaluate_multitask(model, dataloader, label_names, device, task)`

Giống `_evaluate_model` nhưng thêm tham số `task` để chọn đúng head.

#### `_get_multitask_predictions(model, dataloader, device, task)`

Giống `_get_predictions` nhưng truyền `task` vào `model(...)`.

---

### 16.10 `run_experiments.py` — Điều Phối Thí Nghiệm

#### `step1_load_data()` đến `step5_multitask()`

Mỗi hàm step gọi đúng module tương ứng. Không có logic phức tạp.

#### `print_summary(all_results)`

In bảng tổng hợp kết quả. Logic suy luận tên bài toán từ key:
```python
task = "stance" if "stance" in key else "sentiment"
model = key.replace(f"_{task}", "").replace("_", " ").title()
# "baseline_stance" → task="stance", model="Baseline"
```

#### `main()`

```python
parser = argparse.ArgumentParser(...)
parser.add_argument("--step", type=int, default=0)
# --step 0 (mặc định) = chạy tất cả
# --step 2 = chỉ chạy baseline + lexicon
```

---

### 16.11 `cli.py` — Giao Diện Dòng Lệnh

#### `load_multitask_model()`

```python
model = MultiTaskFinBERT()    # tạo mô hình với kiến trúc đúng
model.load_state_dict(torch.load(
    os.path.join(model_path, "model.pt"),
    map_location=DEVICE,       # tải lên đúng thiết bị (MPS/CPU/CUDA)
    weights_only=True,         # bảo mật: chỉ tải tensor, không tải code
))
model.to(DEVICE)
model.eval()                   # chế độ suy luận
```

**`weights_only=True`**: Khi `torch.load()` tải file pickle, nó có thể
thực thi code tùy ý (rủi ro bảo mật). `weights_only=True` chỉ cho phép
tải dữ liệu tensor.

---

#### `load_finetune_models()`

Tải HAI mô hình riêng biệt (stance + sentiment). Khác với multitask
(một mô hình cho cả hai).

```python
model = AutoModelForSequenceClassification.from_pretrained(model_path)
# from_pretrained tải cả kiến trúc + trọng số từ thư mục đã lưu
```

---

#### `predict_multitask(text, model, tokenizer)` và `predict_finetune(text, models)`

**Logic chung**:
```python
inputs = tokenizer(text, return_tensors="pt", truncation=True, ...)
inputs = inputs.to(DEVICE)

with torch.no_grad():
    logits = model(...)
    probs = F.softmax(logits, dim=-1)    # logits → xác suất (tổng = 1.0)
    pred_idx = probs.argmax().item()     # chỉ số lớp cao nhất
    confidence = probs[pred_idx].item()  # xác suất của lớp đó
```

**`F.softmax`**: Chuyển logits (có thể âm, không giới hạn) thành xác suất
(0-1, tổng bằng 1):
```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```
Ví dụ: logits [2.1, -0.5, 0.3] → softmax → [0.72, 0.05, 0.12]

---

#### `format_prediction(results)`

Tạo chuỗi hiển thị kết quả với thanh trực quan:
```python
bar = "█" * int(score * 30)
# score=0.9 → 27 ký tự █
# score=0.05 → 1 ký tự █
```

---

### 16.12 `demo.py` — Demo Gradio

#### `load_model()`

Giống `load_multitask_model()` trong cli.py, nhưng trả về thêm tên
mô hình để hiển thị trên giao diện.

#### `predict(text, model, tokenizer)`

Trả về HAI dict (stance_scores, sentiment_scores) thay vì dict lồng nhau.
Gradio cần mỗi output là một dict `{label: score}` riêng.

```python
stance_scores = {}
for i, label in enumerate(STANCE_LABELS):
    stance_scores[label] = float(stance_probs[i])
# → {"dovish": 0.036, "hawkish": 0.340, "neutral": 0.624}
```

`float()` chuyển tensor PyTorch → Python float (Gradio không hiểu tensor).

#### `create_demo()`

```python
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # gr.Blocks: container linh hoạt (so với gr.Interface đơn giản hơn)
    # theme=Soft(): giao diện nhẹ nhàng, chuyên nghiệp

    with gr.Row():        # bố cục hàng ngang
        with gr.Column(scale=2):  # cột trái, rộng gấp 2
            text_input = gr.Textbox(...)
        with gr.Column(scale=1):  # cột phải, hẹp hơn
            stance_output = gr.Label(num_top_classes=3)
            # gr.Label: hiển thị nhãn với thanh xác suất

    gr.Examples(
        examples=examples,
        inputs=text_input,
        fn=classify,
        cache_examples=False,  # không cache — mỗi lần nhấp chạy lại
    )

    # Hai cách kích hoạt: nút bấm hoặc Enter
    submit_btn.click(fn=classify, inputs=text_input, outputs=[...])
    text_input.submit(fn=classify, inputs=text_input, outputs=[...])
```

**`demo.launch(share=False, server_name="0.0.0.0", server_port=7860)`**:
- `share=False`: không tạo public URL (chỉ truy cập cục bộ)
- `server_name="0.0.0.0"`: lắng nghe trên mọi network interface
- `server_port=7860`: cổng mặc định của Gradio

---

### 16.13 `push_to_hf.py` — Xuất Bản Lên HuggingFace Hub

#### `push_hf_format_model(local_name, repo_name)`

Upload thư mục mô hình định dạng HuggingFace (đã lưu qua
`model.save_pretrained(...)`) lên `Louisnguyen/{repo_name}`. Cũng upload
file `results/*.json` tương ứng (đổi tên thành `results.json`) để model
card có thể tham chiếu đúng các chỉ số đánh giá.

Bao gồm: `finbert_stance`, `finbert_sentiment`,
`bert_llrd_stance`, `bert_llrd_sentiment`.

#### `push_multitask_model(local_name, repo_name)`

Mô hình đa nhiệm được lưu dạng custom `torch.save(model.state_dict(),
"model.pt")` (xem `src/multitask.py`), không phải định dạng HF, nên cần
đường upload riêng. Upload `model.pt` thô + tokenizer + file
`results/multitask_*.json`.

#### Biến môi trường

`HF_TOKEN` phải được set; script đọc qua `os.environ.get("HF_TOKEN")`.

---

### 16.14 `data_analysis.py` — Biểu Đồ Phân Tích Dữ Liệu & Kết Quả

Tạo 10 biểu đồ PNG trong `analysis/` (xem danh sách ở Mục 12.3) cộng với
một file JSON thống kê mức dataset. Nó đọc kết quả từ
`results/all_results_summary.json` và dataset qua `src.data_loader`, nên
giả định huấn luyện đã chạy xong.

### 16.15 `create_presentation.py` — Tạo Báo Cáo / Slide

Ghép các tài liệu báo cáo cuối từ biểu đồ trong `analysis/`, số liệu
trong `results/`, và phần văn bản tường thuật. Có thể chạy lại sau mỗi
lần thí nghiệm mới; kết quả rơi vào `presentation/` và `report/`.

---

*Tài liệu được cập nhật lại cho COMP6713 2026 T1. Tất cả code, thí nghiệm,
và phân tích đều có thể tái tạo bằng `python run_experiments.py` (bước
1–6), theo sau là `python data_analysis.py` và tuỳ chọn
`python push_to_hf.py`.*
