# Emotion Detection from PDF Documents - Beginner's Guide

This guide explains how to detect emotions in PDF documents using AI models. Learn how to analyze the emotional tone of academic papers, reports, or any text-based PDF!

---

## 📚 Table of Contents

1. [What This Code Does](#what-this-code-does)
2. [Required Libraries](#required-libraries)
3. [Understanding Emotions](#understanding-emotions)
4. [Step-by-Step Explanation](#step-by-step-explanation)
5. [How to Use](#how-to-use)
6. [Understanding the Output](#understanding-the-output)
7. [Common Issues & Solutions](#common-issues--solutions)
8. [Advanced Tips](#advanced-tips)

---

## What This Code Does

This notebook performs **emotion analysis** on PDF documents:
- **Extracts** text from PDF files
- **Splits** text into manageable chunks
- **Analyzes** each chunk using an AI emotion detection model
- **Reports** the dominant emotions found in your document
- **Saves** detailed results to a text file

**Real-world use cases:**
- Analyze customer feedback documents
- Study emotional tone in literature
- Evaluate sentiment in research papers
- Monitor emotional patterns in reports

---

## Required Libraries

Install these Python packages:

```bash
pip install pypdf transformers torch
```

**What each library does:**
- `pypdf`: Extracts text from PDF files
- `transformers`: Provides pre-trained AI models for emotion detection
- `torch`: PyTorch deep learning framework (required by transformers)

---

## Understanding Emotions

The default model (`j-hartmann/emotion-english-distilroberta-base`) detects **7 basic emotions**:

| Emotion | Description | Example Text |
|---------|-------------|--------------|
| **joy** | Happiness, pleasure, satisfaction | "We are excited to announce..." |
| **sadness** | Unhappiness, sorrow, disappointment | "Unfortunately, the results were negative..." |
| **anger** | Frustration, irritation, displeasure | "This approach is fundamentally flawed..." |
| **fear** | Anxiety, worry, concern | "There are significant risks involved..." |
| **surprise** | Astonishment, unexpectedness | "Surprisingly, the findings contradict..." |
| **disgust** | Strong disapproval, aversion | "This method is completely inappropriate..." |
| **neutral** | No strong emotion, factual tone | "The experiment was conducted using..." |

**Important note:** Academic papers often score high on "neutral" because they use objective, factual language!

---

## Step-by-Step Explanation

### 🔍 Step 1: Extract Text from PDF

**Function:** `extract_text_from_pdf(pdf_path)`

Same as the preprocessing notebook - converts PDF pages to plain text.

**What you get:**
```python
{
    "pages_text": ["page 1 text", "page 2 text", ...],
    "full_text": "all pages combined",
    "num_pages": 10
}
```

---

### ✅ Step 2: Quality Check

**Function:** `extraction_report(text)`

Quick validation to ensure extraction worked properly.

**Key metrics:**
- `chars`: Total characters extracted
- `words`: Word count
- `printable_ratio`: Should be > 0.95 (means text is clean)
- `preview`: First 800 characters

---

### 🧩 Step 3: Smart Text Chunking

**Why chunking is needed:**
- AI models have **token limits** (usually 512 tokens)
- A token ≈ 4 characters on average
- Long documents must be split into smaller pieces

**Function:** `chunk_by_tokens(text, tokenizer, max_tokens, overlap_tokens)`

**How it works:**

1. **Converts text to tokens** (model's internal units)
2. **Splits into chunks** of `max_tokens` size
3. **Overlaps chunks** by `overlap_tokens` to avoid cutting sentences

**Visual example:**
```
Full text: 2000 tokens
↓
Chunk 1: tokens 0-480
Chunk 2: tokens 400-880   (80 token overlap with chunk 1)
Chunk 3: tokens 800-1280  (80 token overlap with chunk 2)
Chunk 4: tokens 1200-1680
Chunk 5: tokens 1600-2000
```

**Why overlap?**
- Prevents splitting sentences/phrases awkwardly
- Ensures emotional context isn't lost at boundaries
- Default overlap = 80 tokens (about 20 words)

---

### 🔐 Step 4: Chunk Validation

**Function:** `validate_chunks(chunks, tokenizer, max_length)`

**What it does:**
- Double-checks that all chunks fit within model limits
- Accounts for special tokens (CLS, SEP) added automatically
- Reports any problematic chunks

**Example output:**
```
WARNING: 2 chunks exceed max length!
  Chunk 3: 524 tokens (max: 512)
  Chunk 7: 516 tokens (max: 512)
  Truncation will occur during inference.
```

**Don't panic if you see this!** The model will automatically truncate - you just might lose a few words at the end of those chunks.

---

### 🤖 Step 5: Emotion Classification

**Function:** `emotion_from_pdf(...)`

**The magic happens here!**

For each chunk:
1. **Feed text** into the emotion detection model
2. **Get predictions** for all 7 emotions (with confidence scores)
3. **Select top emotion** (highest confidence)
4. **Record results**

**Parameters explained:**

```python
emotion_from_pdf(
    pdf_path="document.pdf",           # Your PDF file
    model_name="j-hartmann/...",       # Which AI model to use
    device=-1,                          # -1 = CPU, 0 = GPU
    overlap_tokens=80,                  # Overlap between chunks
    out_txt="emotion_results.txt"      # Where to save results
)
```

**Under the hood:**
```python
# For each chunk, the model returns something like:
[
    {"label": "neutral", "score": 0.85},
    {"label": "joy", "score": 0.08},
    {"label": "surprise", "score": 0.04},
    {"label": "sadness", "score": 0.02},
    {"label": "anger", "score": 0.01},
    ...
]
# Top emotion = "neutral" with 85% confidence
```

---

### 📊 Step 6: Results Aggregation

After analyzing all chunks, the code:

1. **Counts** how many times each emotion appears as the top emotion
2. **Saves** detailed per-chunk results to a text file
3. **Returns** summary statistics

**Example counts:**
```
neutral: 45 chunks
joy: 8 chunks
surprise: 3 chunks
sadness: 2 chunks
anger: 1 chunk
fear: 1 chunk
disgust: 0 chunks
```

---

## How to Use

### Basic Usage

**Step 1:** Set your PDF path
```python
pdf_path = "my_document.pdf"
```

**Step 2:** Run the analysis
```python
results, counts = emotion_from_pdf(pdf_path)
```

**Step 3:** Check the output file
```
emotion_results.txt created!
```

### Complete Example

```python
# Analyze with custom parameters
results, counts = emotion_from_pdf(
    pdf_path="research_paper.pdf",
    model_name="j-hartmann/emotion-english-distilroberta-base",
    device=-1,              # Use CPU
    overlap_tokens=100,     # More overlap for better context
    out_txt="my_results.txt"
)

# Print summary
print(f"Analyzed {len(results)} chunks")
print("\nEmotion distribution:")
for emotion, count in counts.most_common():
    percentage = (count / len(results)) * 100
    print(f"{emotion:>10}: {count:>3} chunks ({percentage:>5.1f}%)")
```

---

## Understanding the Output

### Console Output

```
Extraction sanity: {'chars': 45820, 'words': 7234, 'printable_ratio': 0.9956, 'preview': '...'}
Chunks: 15 (max_tokens=480, overlap=80)
Saved: emotion_results.txt
```

**What this tells you:**
- **chars/words**: Amount of text extracted (should match your document roughly)
- **printable_ratio**: Close to 1.0 = clean extraction
- **Chunks**: More chunks = longer document

---

### Output File Format

```
MODEL: j-hartmann/emotion-english-distilroberta-base
PDF: research_paper.pdf
Chunks: 15 (max_tokens=480, overlap=80)

Per-chunk top emotion:
Chunk   1: neutral         score=0.8234
Chunk   2: neutral         score=0.7891
Chunk   3: joy             score=0.6543
Chunk   4: surprise        score=0.7234
...

Counts:
neutral: 10
joy: 3
surprise: 2
```

**How to interpret:**

1. **High neutral count**: Normal for academic/technical documents
2. **High specific emotion**: Document has strong emotional tone
3. **Mixed emotions**: Complex emotional content
4. **Score values**:
   - 0.7-1.0 = Very confident prediction
   - 0.5-0.7 = Moderately confident
   - 0.3-0.5 = Uncertain (mixed emotions in chunk)

---

## Common Issues & Solutions

### ❌ Problem: "Token limit exceeded" warnings

**What it means:** Some chunks are too long for the model

**Why it happens:**
- Very long paragraphs with no breaks
- Dense technical text with long sentences

**Solutions:**
1. **Reduce max_tokens:**
   ```python
   max_tokens = max(64, max_in - 64)  # More conservative margin
   ```

2. **Increase overlap:**
   ```python
   overlap_tokens=120  # Better context preservation
   ```

3. **The model will auto-truncate** - usually not a big problem!

---

### ❌ Problem: Everything is classified as "neutral"

**Why this happens:**
- Academic/technical writing is inherently neutral
- Model was trained on emotional text (social media, reviews)
- Your document uses formal, objective language

**Solutions:**
1. **This is often correct!** Academic papers should be neutral
2. **Try a different model** trained on formal text:
   ```python
   model_name="cardiffnlp/twitter-roberta-base-emotion"
   ```
3. **Analyze specific sections** (introduction, conclusion) separately
4. **Look at score distributions** - low confidence might indicate subtle emotions

---

### ❌ Problem: Slow processing / Out of memory

**Symptoms:**
- Takes forever to process
- Computer freezes
- "CUDA out of memory" error

**Solutions:**

1. **Use CPU instead of GPU:**
   ```python
   device=-1  # Always use CPU
   ```

2. **Reduce chunk size:**
   ```python
   max_tokens = 256  # Smaller chunks
   ```

3. **Process in batches:**
   ```python
   # Manually split document into sections
   section1 = extract_text_from_pdf("doc.pdf")["pages_text"][:5]
   section2 = extract_text_from_pdf("doc.pdf")["pages_text"][5:10]
   ```

4. **Close other applications** to free up RAM

---

### ❌ Problem: Inconsistent results between chunks

**Example:**
```
Chunk 1: joy (0.65)
Chunk 2: sadness (0.63)
Chunk 3: joy (0.68)
```

**Why it happens:**
- Emotional tone changes throughout document
- Chunk boundaries split emotional context
- Model is uncertain (low confidence scores)

**How to handle:**
1. **Increase overlap** for better context:
   ```python
   overlap_tokens=150
   ```

2. **Smooth results** by looking at trends, not individual chunks

3. **Focus on overall counts** rather than per-chunk details

4. **Consider sentence-level analysis** for more granular insights

---

## Advanced Tips

### 🎯 Tip 1: Analyze Specific Sections

```python
# Extract only introduction and conclusion
data = extract_text_from_pdf("paper.pdf")
intro = data["pages_text"][0]  # First page
conclusion = data["pages_text"][-1]  # Last page

# Analyze separately
intro_emotions = emotion_from_pdf(intro, out_txt="intro_emotions.txt")
concl_emotions = emotion_from_pdf(conclusion, out_txt="conclusion_emotions.txt")
```

---

### 🎯 Tip 2: Compare Documents

```python
# Analyze multiple papers
papers = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]
all_results = {}

for paper in papers:
    _, counts = emotion_from_pdf(paper, out_txt=f"{paper}_emotions.txt")
    all_results[paper] = counts

# Compare dominant emotions
for paper, counts in all_results.items():
    top_emotion = counts.most_common(1)[0]
    print(f"{paper}: {top_emotion[0]} ({top_emotion[1]} chunks)")
```

---

### 🎯 Tip 3: Visualize Results

```python
import matplotlib.pyplot as plt

results, counts = emotion_from_pdf("paper.pdf")

# Create pie chart
labels = list(counts.keys())
sizes = list(counts.values())
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Emotion Distribution")
plt.show()

# Create bar chart
plt.bar(labels, sizes, color='steelblue')
plt.xlabel("Emotion")
plt.ylabel("Number of Chunks")
plt.title("Emotion Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

### 🎯 Tip 4: Get All Emotion Scores (Not Just Top)

```python
# Modify the main function to return full predictions
clf = pipeline("text-classification", model=model_name, top_k=None)

for chunk in chunks:
    all_emotions = clf(chunk, truncation=True)[0]
    # all_emotions = list of all 7 emotions with scores
    for emotion in all_emotions:
        print(f"{emotion['label']}: {emotion['score']:.3f}")
```

---

### 🎯 Tip 5: Track Emotional Journey

```python
# Plot how emotions change through document
results, _ = emotion_from_pdf("story.pdf")

chunk_numbers = [r["chunk"] for r in results]
emotions = [r["top_emotion"] for r in results]

# Create timeline
emotion_codes = {e: i for i, e in enumerate(set(emotions))}
emotion_values = [emotion_codes[e] for e in emotions]

plt.plot(chunk_numbers, emotion_values, marker='o')
plt.yticks(range(len(emotion_codes)), list(emotion_codes.keys()))
plt.xlabel("Chunk Number")
plt.ylabel("Emotion")
plt.title("Emotional Journey Through Document")
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Model Comparison

### Default Model
**j-hartmann/emotion-english-distilroberta-base**
- ✅ 7 emotion categories
- ✅ Good for general text
- ✅ Fast (distilled model)
- ❌ May over-predict "neutral" for formal text

### Alternative Models

**cardiffnlp/twitter-roberta-base-emotion**
- More sensitive to subtle emotions
- Trained on social media data
- 4 emotions: joy, sadness, anger, optimism

**bhadresh-savani/distilbert-base-uncased-emotion**
- 6 emotions
- Good balance of speed and accuracy
- Works well with shorter texts

**To use a different model:**
```python
emotion_from_pdf(
    "document.pdf",
    model_name="cardiffnlp/twitter-roberta-base-emotion"
)
```

---

## Understanding Confidence Scores

**Score interpretation:**

| Score Range | Meaning | What to Do |
|-------------|---------|------------|
| 0.90 - 1.00 | Very confident | Trust this prediction |
| 0.70 - 0.89 | Confident | Reliable result |
| 0.50 - 0.69 | Moderate | Check context, might be mixed emotions |
| 0.30 - 0.49 | Uncertain | Text has weak emotional signal |
| < 0.30 | Very uncertain | Likely neutral or ambiguous |

**Example:**
```
Chunk 5: joy score=0.92  → Definitely joyful content
Chunk 7: neutral score=0.54  → Weakly neutral, possibly subtle emotion
Chunk 9: anger score=0.38  → Not really angry, just low scores overall
```

---

## Best Practices

### ✅ DO:
- Always check extraction quality first
- Review a few chunks manually to validate results
- Use overlap to preserve context
- Save results for later analysis
- Consider the document type (academic vs. creative writing)
- Compare multiple models if results seem off

### ❌ DON'T:
- Trust results blindly without checking samples
- Use tiny chunks (< 100 tokens) - loses context
- Ignore confidence scores
- Expect strong emotions in formal documents
- Process without validating chunk sizes

---

## Troubleshooting Checklist

Before asking for help, check:

1. ✅ PDF extraction worked (check printable_ratio)
2. ✅ Chunks are reasonable size (check validation output)
3. ✅ Model downloaded successfully (first run takes time)
4. ✅ Output file created (check file path)
5. ✅ Results make sense for your document type
6. ✅ Confidence scores are reasonable (> 0.5 on average)

---

## Next Steps

After emotion detection, you can:

1. **Combine with summarization** to understand emotional context
2. **Track emotions over time** in document series
3. **Compare emotional tones** across authors/sources
4. **Build dashboards** to visualize results
5. **Fine-tune models** on your specific domain

---

## Quick Reference

```python
# Basic usage
results, counts = emotion_from_pdf("document.pdf")

# Custom parameters
results, counts = emotion_from_pdf(
    pdf_path="doc.pdf",
    model_name="j-hartmann/emotion-english-distilroberta-base",
    device=-1,           # CPU
    overlap_tokens=80,   # Context overlap
    out_txt="results.txt"
)

# Check results
print(f"Total chunks: {len(results)}")
print(f"Dominant emotion: {counts.most_common(1)[0][0]}")
```

---

**Remember:** Emotion detection is an AI-powered guess, not absolute truth. Always validate results with human judgment, especially for important decisions! 🎭
