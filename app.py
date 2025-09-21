import fitz  # PyMuPDF
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# ----------------------------
# Helper: Extract text from PDF
# ----------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.lower()

# ----------------------------
# Skills list
# ----------------------------
SKILLS = [
    "python", "java", "c++", "sql", "machine learning", "deep learning", "nlp",
    "excel", "tableau", "power bi", "django", "flask", "react", "aws", "git",
    "docker", "kubernetes", "tensorflow", "pytorch"
]

def extract_keywords(text, skill_list):
    return [skill for skill in skill_list if re.search(r"\b" + re.escape(skill) + r"\b", text)]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Automated Resume Relevance Check System")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if resume_file and jd_file:
    # Extract text
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)

    # Keyword match
    resume_skills = extract_keywords(resume_text, SKILLS)
    jd_skills = extract_keywords(jd_text, SKILLS)

    matched_skills = set(resume_skills) & set(jd_skills)
    missing_skills = set(jd_skills) - set(resume_skills)
    hard_score = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0

    # Semantic match
    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    semantic_score = util.cos_sim(resume_embedding, jd_embedding).item() * 100

    # Final score
    final_score = (0.6 * hard_score) + (0.4 * semantic_score)
    if final_score >= 70:
        verdict = " High Suitability"
    elif final_score >= 40:
        verdict = " Medium Suitability"
    else:
        verdict = " Low Suitability"

    # Display results
    st.subheader("Results")
    st.write(f"**Matched Skills:** {', '.join(matched_skills) if matched_skills else 'None'}")
    st.write(f"**Missing Skills:** {', '.join(missing_skills) if missing_skills else 'None'}")
    st.write(f"**Hard Match Score:** {hard_score:.2f}")
    st.write(f"**Semantic Score:** {semantic_score:.2f}")
    st.write(f"**Final Score:** {final_score:.2f} / 100")
    st.write(f"**Verdict:** {verdict}")
