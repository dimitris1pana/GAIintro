# Metrics Used 

## 1️⃣ Flesch Reading Ease (FRE)

The Flesch Reading Ease score is a readability metric that assesses how easy a text is to read. It is calculated using the formula:

```
206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
```

### Interpretation

- **Higher score = easier to read**
- **Range (typical): 0–100+**

| Score Range | Reading Level          |
|-------------|------------------------|
| 90–100      | Very easy              |
| 60–70       | Standard               |
| 30–50       | Difficult              |
| 0–30        | Very difficult (academic/legal) |

---

## 2️⃣ Flesch–Kincaid Grade Level (FKGL)

### Formula

```
Flesch–Kincaid Grade =
0.39 * (Total Words / Total Sentences) +
11.8 * (Total Syllables / Total Words) -
15.59
```

### Interpretation

- **Output ≈ US school grade level**
- **Examples:**
  - FK = 8 → 8th grade
  - FK = 16 → university level

---

## 3️⃣ Key Difference

| Metric               | Higher Value Means |
|-----------------------|--------------------|
| Flesch Reading Ease   | Easier text        |
| Flesch–Kincaid Grade  | Harder text        |

- FRE is **inverted and scaled**.
- FKGL is **linearized for grade interpretation**.

---

### Python Implementation

```python
def flesch_reading_ease(words, sentences, syllables):
    return (
        206.835
        - 1.015 * (words / sentences)
        - 84.6 * (syllables / words)
    )

def flesch_kincaid_grade(words, sentences, syllables):
    return (
        0.39 * (words / sentences)
        + 11.8 * (syllables / words)
        - 15.59
    )
```

The Flesch Reading Ease and Flesch–Kincaid Grade Level metrics are computed using sentence length and syllabic complexity. While both rely on the same linguistic variables, the Reading Ease score increases with simplicity, whereas the Grade Level increases with textual complexity, making them complementary indicators of readability.

---

# Cosine Similarity

Cosine similarity measures how similar two vectors are in direction, not magnitude. 

It answers the question:

> **Do two texts point in the same semantic or lexical direction, regardless of length?**

### Ideal Use Cases:
- TF-IDF vectors
- Bag-of-Words vectors
- Sentence embeddings

---

## 1️⃣ Mathematical Definition (The Formula)

Given two vectors:

```
A = (a₁, a₂, ..., aₙ)
B = (b₁, b₂, ..., bₙ)
```

**Cosine similarity:**

```
cosine_similarity(A, B) =
(A · B) / (||A|| * ||B||)
```

Where:
- **Dot product:**  
  ```
  A · B = Σ (aᵢ * bᵢ)
  ```
- **Vector norm:**  
  ```
  ||A|| = √(Σ aᵢ²)
  ```

---

## 2️⃣ Output Range and Meaning

| Value  | Meaning                                   |
|--------|-------------------------------------------|
| 1.0    | Identical direction (very similar texts)  |
| 0.0    | Orthogonal (unrelated texts)              |
| −1.0   | Opposite direction (rare in NLP)          |

📌 **In NLP, values are usually 0 → 1 because features are non-negative.**

Cosine similarity measures the angle between two vectors, not their length.

### Key Points:
- Long text vs short text → still comparable
- Verbose vs concise → not penalized
- Focuses on content overlap, not size

Two arrows pointing in the same direction are similar, even if one is longer.

---

## 3️⃣ TF-IDF Cosine Similarity

### Measures:
- Lexical similarity
- Shared important words/phrases

### Used to Show:
- Wording overlap
- Paraphrasing vs copying

---

### Example

**Text A:**  
“The model identifies increased financial risk.”

**Text B:**  
“Higher uncertainty in financial exposure is detected.”

- **TF-IDF cosine → moderate**
- **Embedding cosine → high**

📌 Same meaning, different vocabulary.

---

### Python Implementation

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

Cosine similarity quantifies the similarity between two text representations by computing the cosine of the angle between their vector embeddings. Unlike distance-based measures, it is invariant to document length and therefore well suited for comparing TF-IDF vectors and transformer-based sentence embeddings.

Cosine similarity compares **meaning or content orientation**, not text length.