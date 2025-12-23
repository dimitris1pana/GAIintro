# PDF Text Preprocessing Guide 

This guide explains how to extract, clean, and analyze text from PDF documents using Python. 
---

## 📚 Table of Contents

1. [What This Code Does](#what-this-code-does)
2. [Required Libraries](#required-libraries)
3. [Step-by-Step Explanation](#step-by-step-explanation)
4. [How to Use](#how-to-use)
5. [Understanding the Output](#understanding-the-output)
6. [Common Issues & Solutions](#common-issues--solutions)

---

## What This Code Does

This notebook helps you:
- **Extract text** from PDF files
- **Clean and process** the text (remove unnecessary words, standardize format)
- **Analyze** the most important words and phrases
- **Visualize** the results with charts
- **Prepare text** for a downstream task like emotion detection etc 

Think of it like organizing a messy drawer: you take everything out (extraction), throw away what you don't need (preprocessing), organize what's left (analysis), and display it nicely (visualization).

---

## Required Libraries

Before running the code, install these Python libraries:

```bash
pip install pypdf pandas scikit-learn nltk matplotlib seaborn transformers torch
```

**What each library does:**
- `pypdf`: Reads PDF files
- `pandas`: Organizes data in tables
- `scikit-learn`: Provides text analysis tools
- `nltk`: Natural Language Toolkit for text processing
- `matplotlib` & `seaborn`: Create charts and graphs
- `transformers` & `torch`: For AI-powered text summarization

---

## Step-by-Step Explanation

### 🔍 Step 1: Extract Text from PDF

**Function:** `extract_text_from_pdf(pdf_path)`

**What it does:** 
- Opens your PDF file
- Reads each page one by one
- Combines all pages into one big text

**Example:**
```python
data = extract_text_from_pdf("my_document.pdf")
full_text = data["full_text"]  # All text combined
pages = data["pages_text"]      # Text from each page separately
```

**Why it matters:** You can't analyze a PDF directly - you need to convert it to plain text first!

---

### ✅ Step 2: Quality Check

**Function:** `extraction_report(text)`

**What it does:**
- Counts characters and words
- Checks if the extraction worked properly
- Warns you if something looks wrong

**Red flags to watch for:**
- Very few words (< 50 words means extraction probably failed)
- Low printable ratio (< 0.8 means weird characters or encoding issues)

**Example output:**
```python
{
    "chars": 45000,
    "words": 8500,
    "printable_ratio": 0.98,
    "has_many_empty": False  # Good! Extraction worked
}
```

---

### 🧹 Step 3: Text Preprocessing

This is where we **clean up** the text to make it easier to analyze.

#### A. Tokenization
**Function:** `basic_tokenize(text)`

**What it does:** Splits text into individual words (called "tokens")

**Example:**
```
Input:  "The cat's sitting on the mat!"
Output: ["the", "cat's", "sitting", "on", "the", "mat"]
```

#### B. Remove Stopwords
**Function:** `get_stopwords()`

**What it does:** Removes common words that don't add much meaning (like "the", "is", "a", "an")

**Why?** Words like "the" appear everywhere but tell us nothing about the content!

**Example:**
```
Before: ["the", "cat", "is", "sitting", "on", "the", "mat"]
After:  ["cat", "sitting", "mat"]
```

#### C. Stemming vs Lemmatization

**Stemming** (`stem_tokens`): Chops words down to their root
- running → run
- studies → studi
- Faster but rougher

**Lemmatization** (`lemmatize_tokens`): Finds the dictionary form
- running → run
- studies → study
- Slower but more accurate

**Full preprocessing function:**
```python
result = preprocess(full_text, mode="stem")  # or mode="lemma"
clean_text = result["text_processed"]
```

---

### 📊 Step 4: Find Important Words & Phrases

**Function:** `top_ngrams(texts, method, ngram_range, top_k)`

**What are n-grams?**
- **Unigram** (1-gram): Single word → "machine"
- **Bigram** (2-gram): Two words → "machine learning"
- **Trigram** (3-gram): Three words → "deep neural network"

**Two methods to find important terms:**

#### TF-IDF (Term Frequency-Inverse Document Frequency)
- Shows words that are **important but not too common**
- Good for finding **unique topics** in your document

#### BoW (Bag of Words)
- Simply counts how often words appear
- Good for finding **most mentioned terms**

**Example usage:**
```python
# Find top 20 most important bigrams using TF-IDF
tfidf_df = top_ngrams([text], method="tfidf", ngram_range=(1,2), top_k=20)
```

---

### 📈 Step 5: Visualize Results

**Function:** `plot_ngrams(df, title)`

**What it does:** Creates a horizontal bar chart showing the most important words/phrases

**Reading the chart:**
- Longer bars = more important/frequent
- Y-axis = the words/phrases
- X-axis = importance score

---

### ✂️ Step 6: Chunk Text for Summarization

**Function:** `chunk_by_words(text, words_per_chunk, overlap)`

**Why chunk?** 
- AI models can only read a limited amount of text at once
- Long documents need to be split into smaller pieces

**Parameters explained:**
- `words_per_chunk=800`: Each chunk has ~800 words
- `overlap=80`: Chunks overlap by 80 words (prevents cutting sentences awkwardly)

**Example:**
```
Document: 5000 words
↓
Chunk 1: words 1-800
Chunk 2: words 720-1520 (overlaps with chunk 1)
Chunk 3: words 1440-2240
... and so on
```

---

### 🤖 Step 7: AI Summarization

**Function:** `summarize_chunks(chunks, model_name)`

**What it does:** 
- Uses an AI model to create short summaries of each chunk
- Automatically handles text length limits

**Parameters:**
- `model_name`: Which AI model to use (e.g., "facebook/bart-large-cnn")
- `max_new_tokens=180`: Maximum length of summary
- `device=-1`: Use CPU (change to 0 for GPU if available)

---

## How to Use

### Basic Workflow

1. **Set your PDF path:**
```python
PDF_PATH = "your_document.pdf"
```

2. **Run the extraction:**
```python
data = extract_text_from_pdf(PDF_PATH)
full_text = data["full_text"]
```

3. **Check if extraction worked:**
```python
rep = extraction_report(full_text)
print(rep)
```

4. **Clean the text:**
```python
pp = preprocess(full_text, mode="stem")
processed_text = pp["text_processed"]
```

5. **Analyze and visualize:**
```python
# Find important terms
tfidf_df = top_ngrams([processed_text], method="tfidf", ngram_range=(1,2), top_k=20)

# Plot results
plot_ngrams(tfidf_df, title="Most Important Terms")
```

6. **Optional - Summarize:**
```python
chunks = chunk_by_words(full_text, words_per_chunk=800)
summaries = summarize_chunks(chunks, model_name="facebook/bart-large-cnn")
```

---

## Understanding the Output

### Token Counts
```
raw: 10000          # Total words before cleaning
no_stop: 5000       # After removing stopwords
processed: 4800     # After stemming/lemmatization
```

**What this tells you:**
- If `raw` is very low → PDF extraction might have failed
- If `no_stop` is ~50% of `raw` → Normal (stopwords are ~50% of text)
- If `processed` is close to `no_stop` → Good preprocessing

### N-gram Tables

| ngram | score |
|-------|-------|
| machine learning | 0.45 |
| neural network | 0.38 |
| deep learning | 0.32 |

**How to read:**
- Higher score = more important/frequent
- Look for domain-specific terms (tells you what the document is about)
- Repeated terms across methods = definitely important!

---

## Common Issues & Solutions

### ❌ Problem: "No text extracted" or very few words

**Possible causes:**
1. PDF is image-based (scanned document)
2. PDF has security restrictions
3. Unusual PDF encoding

**Solutions:**
1. Use OCR (Optical Character Recognition) for scanned PDFs
2. Try a different PDF extraction library (e.g., `pdfminer.six`)
3. Check if PDF opens correctly in a PDF reader

---

### ❌ Problem: "NLTK data not found"

**Error message:** `Resource stopwords not found`

**Solution:**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

Run this once to download required data.

---

### ❌ Problem: Memory error during summarization

**Error:** `CUDA out of memory` or system freezes

**Solutions:**
1. Use CPU instead: `device=-1`
2. Reduce chunk size: `words_per_chunk=500`
3. Use a smaller model: `model_name="sshleifer/distilbart-cnn-12-6"`

---

### ❌ Problem: Weird characters in output

**Example:** `caf\u00e9` instead of `café`

**Solution:**
Already handled in the code! The extraction removes soft hyphens and normalizes text.

---

## Tips for Best Results

### ✅ DO:
- Always check the extraction report before proceeding
- Preview the text to ensure it looks correct
- Try both stemming and lemmatization to see which works better
- Save your processed text for later use
- Compare TF-IDF and BoW results

### ❌ DON'T:
- Skip the quality check (you might analyze garbage!)
- Use huge chunk sizes (models have limits)
- Forget to handle errors (always wrap in try-except for production)
- Process without removing stopwords (wastes computation)

---

## Next Steps

After preprocessing, you can:
1. **Topic Modeling:** Find themes in your document (LDA, NMF)
2. **Sentiment Analysis:** Determine if text is positive/negative
3. **Named Entity Recognition:** Find people, places, organizations
4. **Question Answering:** Build a system that answers questions about the document
5. **Document Classification:** Categorize documents automatically

---



**Remember:** Text preprocessing is often 80% of the work in NLP projects. Taking time to understand and clean your data properly will make everything else much easier! 🎯
