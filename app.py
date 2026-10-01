import streamlit as st
import pandas as pd
from nlp_engine import (extract_text_from_pdf, compute_similarity, analyze_skill_gaps, get_top_keywords, generate_tailored_recommendations)

st.set_page_config(page_title="AI Resume & Job Matcher", page_icon=" ", layout="wide")
st.title("📯 AI Resume & Job Fit Analyzer")
st.caption("Semantic vector matching, skill gap discovery, and AI-tailored resume bullet recommendations.")

with st.sidebar:
    st.header("project Architecture")
    st.markdown("""
    * **Embeddings**: `all-MiniLM-L6-v2` dense vectors.
    * **Cosine Similarity**: Contextual semantic overlap scoring.
    * **Taxonomy Engine**: Scans 70+ technical proficiencies.
    * **TF-IDF**: High-impact keyword extraction.
    * **Actionable Rewrites**: Generates bullet points using Google XYZ formula.
    """)

col_l, ol_r = st.columns(2)
with col_l:
    st.subheader("1. Upload or Paste Resume")
    upload_type = st.radio("Resume Source", ["Upload PDF", "Paste Plain Text"], horizontal=True)
    if upload_type == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        resume_text = extract_text_from_pdf(uploaded_file) if uploaded_file else ""
        if resume_text:
            st.success("PDF parsed successfully!")
    else:
        resume_text = st.text_area("Paste resume text here:", height=250)

with ol_r:
    st.subheader("2. Paste Job Description")
    default_jd = "We are seeking a Machine Learning Engineer proficient in Python, PyTorch, Scikit-Learn, Docker, FastAPI, and Git."
    job_description = st.text_area("Paste target job description:", value=default_jd, height=280)

if st.button("Θ Run Semantic Match & Skill Gap Analysis", type="primary", use_container_width=True):
    if not resume_text.strip():
        st.error("Please provide a resume.")
    elif not job_description.strip():
        st.error("Please provide a job description.")
    else:
        with st.spinner("Calculating semantic similarity & analyzing skill overlaps..."):
            score = compute_similarity(resume_text, job_description)
            gaps = analyze_skill_gaps(resume_text, job_description)
            keywords = get_top_keywords(job_description, top_n=8)
            recommendations = generate_tailored_recommendations(resume_text, job_description, gaps["missing_skills"])

        st.divider()
        st.subheader("🐪 Match Overview")
        m1, m2, m3 = st.columns(3)
        m1.metric("Semantic Match Score", f"{score}%", "Vector Cosine Similarity")
        m2.metric("Matched Target Skills", len(gaps["matched_skills"]), "Aligned")
        m3.metric("Missing Requirements", len(gaps["missing_skills"]), "Gaps to address")


        st.write("")
        st.subheader("✨ AI Actionable Resume Tailoring & Bullet Rewrites")
        st.markdown("Incorporate these quantifiable bullet points (crafted with the **Google XYZ formula**) into your resume:")
        if recommendations:
            for rec in recommendations:
                with st.expander(f"💫 {rec['skill']} - {rec['category']}", expanded=True):
                    st.code(rec["bullet"], language="markdown")
        else:
            st.success("Outstanding! No major skill gaps identified for this posting.")

        st.write("")
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ✅ Matched Skills in Resume")
            if gaps["matched_skills"]:
                st.write(", ".join([s.upper() for s in gaps["matched_skills"]]))
            else:
                st.info("No direct keyword matches found.")
            st.write("")
            st.markdown("#### 🔡 Additional Skills on Resume")
            if gaps["additional_resume_skills"]:
                st.write(", ".join([s.upper() for s in gaps["additional_resume_skills"]]))
        with c2:
            st.markdown("#### ⚠️ Missing Skills (Target Role)")
            if gaps["missing_skills"]:
                st.write(", ".join([s.upper() for s in gaps["missing_skills"]]))
            else:
                st.success("Zero technical skill gaps detected!")
            st.write("")
            st.markdown("#### 🔍 Top Salient Job Terms (TF-IDF)")
            if keywords:
                df_kw = pd.DataFrame(keywords, columns=["Key Term", "Relevance Score"])
                st.dataframe(df_kw, use_container_width=True, hide_index=True)
