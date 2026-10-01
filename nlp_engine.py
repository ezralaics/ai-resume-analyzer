import re
from typing import List, Dict, Tuple, Set
import pypdf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

COMMON_TECH_SKILLS = {
    # Languages
    'python', 'r', 'sql', 'java', 'c++', 'c#', 'javascript', 'typescript', 'scala', 'go', 'rust', 'bash',
    # ML/DL
    'machine learning', 'deep learning', 'nlp', 'computer vision', 'neural networks', 'transformers',
    'supervised learning', 'unsupervised learning', 'reinforcement learning', 'llm', 'large language models',
    'rag', 'retrieval augmented generation', 'prompt engineering', 'embeddings', 'vector search',
    # Frameworks
    'scikit-learn', 'sklearn', 'pytorch', 'tensorflow', 'keras', 'xgboost', 'lightgbm', 'catboost',
    'hugging face', 'huggingface', 'langchain', 'llamaindex', 'spacy', 'nltk', 'opencv',
    # Data Science
    'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'polars', 'statistics',
    'a/b testing', 'hypothesis testing', 'data analysis', 'data visualization', 'exploratory data analysis', 'eda',
    # Data Engineering & DB
    'spark', 'pyspark', 'hadoop', 'kafka', 'dbt', 'airflow', 'sql server', 'postgresql', 'mysql',
    'mongodb', 'redis', 'cassandra', 'snowflake', 'bigquery', 'redshift', 'pinecone', 'chroma', 'qdrant',
    # Cloud, DevOps & Serving
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'fastapi', 'flask', 'django', 'git', 'github',
    'ci/cd', 'mlflow', 'wandb', 'linux', 'rest api', 'microservices'
}

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def extract_text_from_pdf(pdf_file) -> str:
    try:
        reader = pypdf.PdfReader(pdf_file)
        text_parts = []
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text_parts.append(content)
        return '\n'.join(text_parts).strip()
    except Exception as e:
        return f'Error reading PDF: {str(e)}'

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def compute_similarity(resume_text: str, job_desc_text: str) -> float:
    model = get_embedding_model()
    embeddings = model.encode([clean_text(resume_text), clean_text(job_desc_text)])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(round(max(0.0, sim) * 100, 1))

def extract_skills_from_text(text: str) -> Set[str]:
    text_lower = ' ' + text.lower() + ' '
    found_skills = set()
    for skill in COMMON_TECH_SKILLS:
        pattern = r'(?<!\w)' + re.escape(skill) + r'(?!\w)'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    return found_skills

def analyze_skill_gaps(resume_text: str, job_desc_text: str) -> Dict[str, List[str]]:
    resume_skills = extract_skills_from_text(resume_text)
    job_skills = extract_skills_from_text(job_desc_text)
    
    matched = sorted(list(resume_skills.intersection(job_skills)))
    missing = sorted(list(job_skills.difference(resume_skills)))
    additional = sorted(list(resume_skills.difference(job_skills)))
    
    return {
        'matched_skills': matched,
        'missing_skills': missing,
        'additional_resume_skills': additional
    }

def get_top_keywords(text: str, top_n: int = 8) -> List[Tuple[str, float]]:
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([clean_text(text)])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        keywords = [(feature_names[i], round(float(scores[i]), 3)) for i in scores.argsort()[::-1]]
        return keywords
    except Exception:
        return []

def generate_tailored_recommendations(resume_text: str, job_desc_text: str, missing_skills: List[str]) -> List[Dict[str, str]]:
    suggestions = []
    action_verbs = {
        'docker': ('Containerized application services using Docker and multi-stage builds', 'reducing deployment discrepancies and speeding up CI/CD test runs by 40%'),
        'aws': ('Architected cloud workloads on AWS (S3, EC2, Lambda)', 'scaling inference throughput and ensuring 99.9% service availability'),
        'fastapi': ('Developed high-throughput RESTful API endpoints using FastAPI', 'reducing P95 latency to under 50ms for real-time model queries'),
        'pytorch': ('Trained and fine-tuned deep learning architectures with PyTorch', 'improving model predictive accuracy by 14% over baseline algorithms'),
        'scikit-learn': ('Engineered predictive pipelines using Scikit-Learn', 'automating feature preprocessing and cross-validation across 100K+ records'),
        'sql': ('Optimized complex analytical SQL queries and database schemas', 'cutting database execution time from 15s to under 800ms'),
        'git': ('Standardized Git version control workflows and PR review criteria', 'accelerating team release velocity across development sprint cycles'),
        'kubernetes': ('Orchestrated microservices deployment via Kubernetes clusters', 'enabling zero-downtime rolling updates and automated scaling'),
        'mlflow': ('Integrated MLflow experiment tracking and model registry', 'standardizing hyperparameter logging and reproducibility across training runs'),
        'langchain': ('Engineered Retrieval-Augmented Generation (RAG) pipelines using LangChain', 'enhancing context-aware query accuracy by 25%')
    }
    
    for skill in missing_skills[:4]:
        skill_lower = skill.lower()
        if skill_lower in action_verbs:
            action, impact = action_verbs[skill_lower]
            suggestions.append({
                'skill': skill.upper(),
                'category': 'Suggested Impact Bullet (Google XYZ Format)',
                'bullet': f'{action}, which {impact}.'
            })
        else:
            suggestions.append({
                'skill': skill.upper(),
                'category': 'Competency Integration',
                'bullet': f'Implemented scalable solutions leveraging {skill.title()}, standardizing workflows and enhancing project delivery.'
            })
            
    if len(resume_text.split()) < 200:
        suggestions.append({
            'skill': 'RESUME DEPTH & METRICS',
            'category': 'Formatting / ATS Optimization',
            'bullet': 'Your resume text is concise (< 200 words). Add quantifiable metrics (e.g., % improvement, latency, dataset size) to each project.'
        })
        
    return suggestions
