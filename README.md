# AI Resume & Job Fit Analyzer

An end-to-end Natural Language Processing (NLP) system that evaluates resumes against job descriptions using semantic vector embeddings, cosine similarity, skill gap extraction, and TF-IDF powered keyword analysis.

---

## ✨ Features & Architecture

1. **Semantic Text Embeddings:** Uses Hugging Face's `sentence-transformers` (`all-MiniLM-L6-v2`) to convert unconstrained resume text and job postings into 384-dimensional dense vectors.
2. **Cosine Similarity Matching:** Understands contextual meaning beyond naive keyword counting.
3. **Skill Gap Detection:** Scans 70+ common technical competencies to highlight matched versus missing requirements.
4. **TF-IDF Keyword Extraction:** Reveals high-impact terms in the job description to prioritize in resume tailoring.
5. **Interactive Streamlit UI:** Allows pdf upload or direct pasting with instant visual feedback.

---

## 💨 Project Structure

```
ai-resume-analyzer/
 ├─ app.py               # Streamlit frontend dashboard & metrics visualization
 ├─ nlp_engine.py       # CERE NLP logic, PDF parsing, embeddings & skill gap analysis
 ├─ requirements.txt     # Dependencies
 └─ README.md            # Documentation & resume guide
```

---

## 🚀 Quick Start

1. **Activate virtual environment:**
   ```powershell
   ./venv/Scripts/activate
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Run the Web App:**
   ```powershell
   streamlit run app.py
   ```
