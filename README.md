# Introduction to Generative AI & LLMs
### University of Piraeus - Course Materials

---

## 📚 Course Overview

This repository contains interactive Jupyter notebooks and educational materials for an introductory course on Generative AI. The course covers fundamental concepts in natural language processing, embeddings, attention mechanisms, and retrieval-augmented generation (RAG) systems.

## 🎯 Learning Objectives

By working through these materials, students will:
- Understand how text is transformed into numerical representations (embeddings)
- Learn the fundamentals of attention mechanisms in modern AI systems
- Explore semantic similarity and vector space operations
- Implement basic RAG (Retrieval-Augmented Generation) pipelines
- Visualize and interpret embedding transformations
- Extract and process information from unstructured documents

---

## 📁 Repository Structure

### Core Notebooks

#### 1. **testing.ipynb** - Python & Jupyter Fundamentals
*Introduction to the course environment*

**Topics Covered:**
- Getting started with Jupyter notebooks
- Basic Python programming concepts
- Working with lists and data structures
- Introduction to pandas DataFrames
- Text data manipulation


---

#### 2. **transformation.ipynb** - Text Embeddings & Semantic Similarity
*Understanding how text becomes numbers*

**Topics Covered:**
- Sentence embeddings using pre-trained models (`all-MiniLM-L6-v2`)
- Converting text to high-dimensional vectors
- Dimensionality reduction with PCA
- 2D visualization of semantic relationships
- Exploring similarity between sentences

**Key Concepts:**
```python
# Example sentences demonstrating semantic clustering:
- "The cat sits on the laptop"
- "A dermatologist examines a skin lesion"
- "Skin cancer screening uses dermatoscopic images"
```

**Visualization:** Interactive 2D scatter plot showing how semantically similar sentences cluster together in embedding space.

---

#### 3. **attentionWeights.ipynb** - Attention Mechanisms & RAG Fundamentals
*How AI systems focus on relevant information*

**Topics Covered:**
- Query-document similarity using cosine distance
- Attention weight calculation via softmax
- Weighted context vector generation
- Basic RAG (Retrieval-Augmented Generation) concepts
- Word generation from context vectors

**Mathematical Concepts:**
1. **Similarity Calculation:**
   ```
   similarity = (doc · query) / (||doc|| × ||query||)
   ```

2. **Attention Weights:**
   ```
   weight_i = exp(sim_i) / Σ(exp(sim_j))
   ```

3. **Context Vector:**
   ```
   context = Σ(weight_i × doc_i)
   ```

**Practical Application:** Demonstrates how attention mechanisms prioritize relevant documents when generating responses.

---

#### 4. **contextextraction.ipynb** - Document Processing & Semantic Search
*Real-world PDF analysis and retrieval*

**Topics Covered:**
- PDF text extraction from structured and unstructured layouts
- Page-level embedding generation
- Document similarity matrices
- Comparing single-column vs. multi-column PDF formats
- Building searchable document collections

**Libraries Used:**
- `pypdf`: PDF text extraction
- `sentence-transformers`: Semantic embeddings
- `sklearn`: Cosine similarity calculations
- `pandas`: Data organization

**Use Cases:**
- Academic paper analysis
- Document retrieval systems
- Content similarity detection
- Knowledge base construction

---

#### 5. **anim/animationEmbeddings.ipynb** - 3D Embedding Visualization
*Visualizing the embedding transformation process*

**Topics Covered:**
- 3D visualization of word embeddings
- Animated transformation from random to semantic space
- PCA dimensionality reduction to 3D
- Interactive matplotlib animations
- Understanding embedding space structure

**Features:**
- Real-time animation showing tokens moving from random positions to their semantic coordinates
- Color-coded token categories
- Export animations to MP4 format (requires FFmpeg)
- Interactive HTML visualization

**Sample Text:**
```
"The cat sits on the laptop while the dermatologist examines a skin lesion.
The dog plays in the garden near the clinic."
```

**Visual Output:** 60-frame animation demonstrating how word embeddings organize themselves in 3D space based on semantic meaning.

---

### Supporting Files

- **Gpt4osim.md** - Conversation logs demonstrating AI reasoning about embeddings and analogies
- **clean.txt, clean2.txt, clean3.text** - Text datasets for experiments
- **PDF examples** (referenced in notebooks) - Structured and unstructured document samples

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### Required Libraries

Install all dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn
pip install sentence-transformers
pip install pypdf
pip install nltk
```

### Optional (for animations):
```bash
# For saving animations to video
pip install ffmpeg-python
# Or install FFmpeg system-wide:
# macOS: brew install ffmpeg
# Linux: apt-get install ffmpeg
```

### First-Time Setup

1. Clone or download this repository
2. Open terminal in the repository directory
3. Install dependencies (see above)
4. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
5. Start with `testing.ipynb` for an introduction

---

## 📖 Recommended Learning Path

### Week 1-2: Foundations
1. **testing.ipynb** - Get comfortable with Python and Jupyter
2. **transformation.ipynb** - Learn about embeddings and visualization

### Week 3-4: Core Concepts
3. **attentionWeights.ipynb** - Understand attention mechanisms
4. **contextextraction.ipynb** - Apply concepts to real documents

### Week 5: Advanced Visualization
5. **anim/animationEmbeddings.ipynb** - Explore 3D embedding space

---

## 🔑 Key Concepts Covered

### 1. **Embeddings**
Numerical representations of text that capture semantic meaning:
- Words with similar meanings have similar vector representations
- Enable mathematical operations on language
- Foundation for modern NLP systems

### 2. **Attention Mechanisms**
How AI systems determine what information is relevant:
- Query-document similarity scoring
- Softmax normalization for probability distribution
- Weighted combination of information sources

### 3. **RAG (Retrieval-Augmented Generation)**
Combining retrieval and generation for accurate AI responses:
- Retrieve relevant documents from knowledge base
- Weight documents by relevance to query
- Generate responses grounded in retrieved context

### 4. **Dimensionality Reduction**
Visualizing high-dimensional data:
- PCA (Principal Component Analysis)
- Projection from 384D to 2D/3D for visualization
- Preserving semantic relationships in lower dimensions

---

## 🎓 Assessment & Exercises

Each notebook contains:
- **Code cells** to execute and experiment with
- **Markdown explanations** of concepts
- **Visualizations** to interpret
- **Questions** to consider (see `Gpt4osim.md` for examples)

### Suggested Exercises

1. **Modify the text samples** in `transformation.ipynb` and observe how embeddings cluster
2. **Change attention weights** in `attentionWeights.ipynb` manually and see effects
3. **Add your own PDFs** to `contextextraction.ipynb` for analysis
4. **Experiment with different embedding models** (see sentence-transformers documentation)
5. **Extend the animation** in `animationEmbeddings.ipynb` with more complex text

---

## 📊 Model Information

### Sentence Transformer: `all-MiniLM-L6-v2`

**Specifications:**
- **Architecture:** MiniLM (distilled from BERT)
- **Parameters:** 22.7 million
- **Embedding Dimension:** 384
- **Max Sequence Length:** 256 tokens
- **Performance:** Fast inference, suitable for real-time applications

**Why This Model?**
- Excellent balance of speed and quality
- Pre-trained on large text corpus
- Optimized for semantic similarity tasks
- Widely used in production systems

---

## 🤝 Contributing & Questions

This is educational material for the University of Piraeus. Students are encouraged to:
- Experiment with the code
- Ask questions during lectures
- Suggest improvements or additional examples
- Share interesting findings

---

## 📚 Additional Resources

### Recommended Reading
- **Attention Is All You Need** (Vaswani et al., 2017) - Original transformer paper
- **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
- **Sentence-BERT** (Reimers & Gurevych, 2019) - Foundation for our embedding model

### Online Resources
- [Hugging Face Sentence Transformers](https://www.sbert.net/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Matplotlib Animation Tutorial](https://matplotlib.org/stable/api/animation_api.html)

### Related Topics to Explore
- Large Language Models (LLMs)
- Vector databases (Pinecone, Weaviate, FAISS)
- Prompt engineering
- Fine-tuning embedding models
- Production RAG systems

---

## 🔧 Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'sentence_transformers'`
- **Solution:** Run `pip install sentence-transformers`

**Issue:** NLTK punkt tokenizer not found
- **Solution:** The notebook automatically downloads it with `nltk.download('punkt')`

**Issue:** Animation doesn't display
- **Solution:** Use `HTML(anim.to_jshtml())` for inline display instead of `plt.show()`

**Issue:** FFmpeg not found when saving animations
- **Solution:** Install FFmpeg system-wide or use `.to_jshtml()` for HTML output

---

## 📝 License & Citation

**Educational Use:** These materials are provided for educational purposes as part of the University of Piraeus curriculum.
---

## 👨‍🏫 Course Information

**Institution:** University of Piraeus  
**Course:** Introduction to Generative AI & LLMs  
**Format:** Interactive Jupyter notebooks with hands-on exercises  
**Level:** Introductory to Intermediate  

---

**Last Updated:** December 2025  
**Repository:** [GAIintro](https://github.com/dimitris1pana/GAIintro)
---

