# app_granite_streamlit.py
"""
StudyMate — Streamlit app using IBM Granite (via Hugging Face API) + local pipeline fallback,
FAISS vector store using HuggingFace embeddings, PDF/PPTX/DOCX/image ingestion, quiz generation, etc.

Requirements (suggested):
    streamlit PyPDF2 python-pptx python-docx easyocr faiss-cpu sentence-transformers transformers huggingface-hub langchain langchain-huggingface pillow requests python-dotenv

Run:
    streamlit run app_granite_streamlit.py
"""

import os
import json
import random
import re
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from io import BytesIO

import streamlit as st

HF_API_KEY = st.secrets["HF_API_KEY"]
from dotenv import load_dotenv

# Optional heavy imports - fail gracefully
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    from pptx import Presentation
except Exception:
    Presentation = None

try:
    from docx import Document
except Exception:
    Document = None

try:
    import easyocr
except Exception:
    easyocr = None

try:
    from PIL import Image
except Exception:
    Image = None

# LangChain text splitter fallback imports (different versions use different paths)
try:
    # new-ish
    from langchain.text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        # older package split
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # your original fallback
    except Exception:
        RecursiveCharacterTextSplitter = None

# Embeddings and vectorstore
# Embeddings and vectorstore (robust fallback)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except:
        HuggingFaceEmbeddings = None

try:
    from langchain_community.vectorstores import FAISS
except:
    try:
        from langchain.vectorstores import FAISS
    except:
        FAISS = None



# Transformers pipeline (optional local pipeline instead of inference API)
try:
    from transformers import pipeline, logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pipeline = None

# Load environment variables
load_dotenv()
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_KEY", "") or os.getenv("HF_API_KEY", "")
HF_MODEL = os.getenv("HF_MODEL", "ibm-granite/granite-3.3-2b-instruct")
USE_PIPELINE = os.getenv("USE_PIPELINE", "0") in ["1", "true", "True", "yes", "YES"]

# Streamlit quick guard
if not HUGGINGFACE_API_TOKEN and not USE_PIPELINE:
    st.warning("Set HUGGINGFACE_API_KEY in your environment (or set USE_PIPELINE=1 to use a local transformers pipeline).")

# ---------- Session State ----------
if 'study_history' not in st.session_state:
    st.session_state.study_history = []
if 'weak_topics' not in st.session_state:
    st.session_state.weak_topics = defaultdict(int)
if 'quiz_results' not in st.session_state:
    st.session_state.quiz_results = []
if 'uploaded_files_metadata' not in st.session_state:
    st.session_state.uploaded_files_metadata = []
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = {}
if 'faiss_index_created' not in st.session_state:
    st.session_state.faiss_index_created = False

# ---------- Pipeline initialization (lazy) ----------
_gen_pipeline = None
def get_local_pipeline():
    global _gen_pipeline
    if _gen_pipeline is not None:
        return _gen_pipeline
    if pipeline is None:
        return None
    try:
        # load text-generation pipeline; user must ensure model is available locally or via HF
        _gen_pipeline = pipeline("text-generation", model=HF_MODEL, do_sample=True)
        return _gen_pipeline
    except Exception as e:
        st.warning(f"Could not initialize local transformers pipeline: {e}")
        return None

# ========== Helper: call Granite (either local pipeline or HF Inference API) ==========
def call_granite(prompt: str, max_new_tokens: int = 256, temperature: float = 0.2, timeout: int = 120):
    """
    Call IBM Granite via either local transformers pipeline (if USE_PIPELINE)
    or the Hugging Face Router Inference API.
    Returns a string (generated text) or raises a RuntimeError with a helpful message.
    """
    # Local pipeline preferred when requested
    if USE_PIPELINE:
        p = get_local_pipeline()
        if p is None:
            raise RuntimeError("USE_PIPELINE requested but local transformers pipeline could not be initialized.")
        out = p(prompt, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=0.95, num_return_sequences=1)
        if isinstance(out, list) and len(out) > 0 and 'generated_text' in out[0]:
            return out[0]['generated_text']
        # best-effort fallback
        return " ".join([str(x) for x in out])

    # Use Hugging Face Router inference endpoint (NEW)
    if not HUGGINGFACE_API_TOKEN:
        raise RuntimeError("HUGGINGFACE_API_TOKEN not set. Set HF_API_KEY in st.secrets or environment.")

    # New router URL (must use /hf-inference/)
    url = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "repetition_penalty": 1.1
        },
        # router supports the same options structure; keep wait_for_model to True
        "options": {"wait_for_model": True}
    }

    # simple retry (useful for transient HF router errors)
    last_exc = None
    for attempt in range(1, 4):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                # tolerant extraction: many models return list/dict or nested
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    if isinstance(first, dict) and "generated_text" in first:
                        return first["generated_text"]
                    if isinstance(first, str):
                        return first
                if isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"]
                # some router outputs plain string or different shape — return best-effort
                if isinstance(data, str):
                    return data
                # fallback: try to find a 'generated_text' in nested dicts
                def find_generated(obj):
                    if isinstance(obj, dict):
                        if "generated_text" in obj:
                            return obj["generated_text"]
                        for v in obj.values():
                            r = find_generated(v)
                            if r is not None:
                                return r
                    if isinstance(obj, list):
                        for item in obj:
                            r = find_generated(item)
                            if r is not None:
                                return r
                    return None
                found = find_generated(data)
                if found:
                    return found
                # Last resort: return JSON string
                return json.dumps(data)
            else:
                # Helpful error message for common HF responses
                text = resp.text or resp.reason
                if resp.status_code == 410:
                    raise RuntimeError("Hugging Face returned 410: old API endpoint — please ensure you are using the router endpoint.")
                if resp.status_code in (401, 403):
                    raise RuntimeError(f"Hugging Face authentication error {resp.status_code}: check your HF_API_KEY/secret.")
                # transient 502/503/524 etc -> retry
                if resp.status_code in (502, 503, 504):
                    last_exc = RuntimeError(f"Transient HF router error {resp.status_code}: {text}")
                    # small backoff
                    import time; time.sleep(1.0 * attempt)
                    continue
                # otherwise raise with server body
                raise RuntimeError(f"HuggingFace API error {resp.status_code}: {text}")
        except requests.RequestException as e:
            last_exc = e
            import time; time.sleep(0.6 * attempt)
            continue

    # If we get here, all attempts failed
    if last_exc:
        raise RuntimeError(f"Failed to call Hugging Face Router after retries: {last_exc}")
    raise RuntimeError("Unknown error calling Hugging Face Router.")


# ===================== FILE PROCESSING =====================
def extract_text_from_pdf(pdf_file):
    if PdfReader is None:
        st.error("PyPDF2 not installed or unavailable.")
        return []
    text_with_pages = []
    try:
        pdf_reader = PdfReader(pdf_file)
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return []
    for page_num, page in enumerate(pdf_reader.pages, 1):
        try:
            page_text = page.extract_text()
        except Exception:
            page_text = None
        if page_text:
            text_with_pages.append({
                'text': page_text,
                'page': page_num,
                'source': getattr(pdf_file, "name", "uploaded_pdf")
            })
    return text_with_pages

def extract_text_from_pptx(pptx_file):
    if Presentation is None:
        st.error("python-pptx not installed or unavailable.")
        return []
    try:
        prs = Presentation(pptx_file)
        text_with_slides = []
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = ""
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text += shape.text + "\n"
            if slide_text.strip():
                text_with_slides.append({
                    'text': slide_text,
                    'page': slide_num,
                    'source': getattr(pptx_file, "name", "uploaded_pptx")
                })
        return text_with_slides
    except Exception as e:
        st.error(f"Error processing PPTX: {e}")
        return []

def extract_text_from_docx(docx_file):
    if Document is None:
        st.error("python-docx not installed or unavailable.")
        return []
    try:
        doc = Document(docx_file)
        full_text = ""
        for para in doc.paragraphs:
            full_text += para.text + "\n"
        return [{'text': full_text, 'page': 1, 'source': getattr(docx_file, "name", "uploaded_docx")}]
    except Exception as e:
        st.error(f"Error processing DOCX: {e}")
        return []

def extract_text_from_image(image_file):
    if easyocr is None:
        st.error("easyocr not installed or unavailable.")
        return []
    if Image is None:
        st.error("Pillow not installed; needed for image OCR.")
        return []
    try:
        # read bytes and convert to PIL then numpy array for easyocr
        image_bytes = image_file.read()
        pil_img = Image.open(BytesIO(image_bytes)).convert('RGB')
        import numpy as np
        img_arr = np.array(pil_img)
        reader = easyocr.Reader(['en'], gpu=False)  # adjust gpu if needed
        result = reader.readtext(img_arr)
        text = " ".join([d[1] for d in result])
        return [{'text': text, 'page': 1, 'source': getattr(image_file, "name", "uploaded_image")}]
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return []

def process_uploaded_files(uploaded_files):
    all_text_data = []
    for file in uploaded_files:
        file_type = file.name.split('.')[-1].lower()
        if file_type == 'pdf':
            all_text_data.extend(extract_text_from_pdf(file))
        elif file_type in ['ppt', 'pptx']:
            all_text_data.extend(extract_text_from_pptx(file))
        elif file_type == 'docx':
            all_text_data.extend(extract_text_from_docx(file))
        elif file_type in ['png', 'jpg', 'jpeg']:
            all_text_data.extend(extract_text_from_image(file))
        else:
            st.info(f"Skipping unsupported file type: {file.name}")
        st.session_state.uploaded_files_metadata.append({
            'name': file.name,
            'type': file_type,
            'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M")
        })
    return all_text_data

# ===================== KNOWLEDGE EXTRACTION =====================
def extract_knowledge_structure(text_data):
    knowledge = {'concepts': [], 'definitions': [], 'formulas': [], 'key_terms': []}
    for item in text_data:
        text = item['text']
        definition_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is defined as|refers to|means)\s+([^.]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):\s+([^.]+)'
        ]
        for pattern in definition_patterns:
            try:
                matches = re.finditer(pattern, text)
                for match in matches:
                    knowledge['definitions'].append({
                        'term': match.group(1).strip(),
                        'definition': match.group(2).strip(),
                        'source': item['source'],
                        'page': item['page']
                    })
            except Exception:
                continue
        formula_pattern = r'[A-Za-z]+\s*=\s*[^.]+[0-9+\-*/^()]+[^.]*'
        try:
            formulas = re.findall(formula_pattern, text)
            for formula in formulas:
                knowledge['formulas'].append({
                    'formula': formula.strip(),
                    'source': item['source'],
                    'page': item['page']
                })
        except Exception:
            pass
        try:
            key_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b', text)
            knowledge['key_terms'].extend(set(key_terms))
        except Exception:
            pass
    knowledge['key_terms'] = list(set(knowledge['key_terms']))[:50]
    st.session_state.knowledge_graph = knowledge
    return knowledge

# ========== Vector store functions ==========
def get_text_chunks_with_metadata(text_data):
    if RecursiveCharacterTextSplitter is None:
        st.error("Text splitter not available. Install a compatible langchain/text_splitters package.")
        return []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks_with_metadata = []
    for item in text_data:
        try:
            chunks = text_splitter.split_text(item['text'])
        except Exception:
            # fallback: naive splitter by paragraphs
            chunks = [p.strip() for p in item['text'].split('\n\n') if p.strip()]
        for chunk in chunks:
            chunks_with_metadata.append({
                'text': chunk,
                'metadata': {'source': item['source'], 'page': item['page']}
            })
    return chunks_with_metadata

def create_vector_store(chunks_with_metadata):
    if not chunks_with_metadata:
        raise ValueError("No text chunks available.")
    if HuggingFaceEmbeddings is None:
        st.error("❌ HuggingFaceEmbeddings not available. Install: pip install langchain-huggingface")
        return None

    if FAISS is None:
        st.error("❌ FAISS not available. Install: pip install faiss-cpu")
        return None

    texts = [c['text'] for c in chunks_with_metadata]
    metadatas = [c['metadata'] for c in chunks_with_metadata]
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)
    # Save to local dir
    try:
        os.makedirs("faiss_index", exist_ok=True)
        vector_store.save_local("faiss_index")
        st.session_state.faiss_index_created = True
    except Exception as e:
        st.warning(f"Warning: could not save FAISS index locally: {e}")
    return vector_store

def load_vector_store():
    if HuggingFaceEmbeddings is None or FAISS is None:
        raise RuntimeError("HuggingFaceEmbeddings or FAISS is not available in this environment.")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        return FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(f"Could not load FAISS index: {e}")

# ===================== QUIZ GENERATION =====================
def generate_quiz(difficulty="medium", num_questions=5, question_types=None):
    if question_types is None:
        question_types = ["mcq", "true_false", "fill_blank"]
    try:
        vector_store = load_vector_store()
        knowledge = st.session_state.knowledge_graph
        quiz_questions = []
        all_docs = vector_store.similarity_search("", k=20)
        sampled_docs = random.sample(all_docs, min(num_questions, len(all_docs))) if all_docs else []
        for doc in sampled_docs:
            q_type = random.choice(question_types)
            context = doc.page_content
            if q_type == "mcq":
                question = generate_mcq(context, difficulty, doc.metadata)
            elif q_type == "true_false":
                question = generate_true_false(context, doc.metadata)
            elif q_type == "fill_blank":
                question = generate_fill_blank(context, doc.metadata)
            else:
                question = None
            if question:
                quiz_questions.append(question)
        return quiz_questions[:num_questions]
    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return []

def generate_mcq(context, difficulty, metadata):
    sentences = [s.strip() for s in re.split(r'\.|\n', context) if s.strip()]
    if len(sentences) < 1:
        return None
    # pick a sentence with reasonable length
    candidate_sentences = [s for s in sentences if len(s.split()) > 6]
    answer_sentence = random.choice(candidate_sentences) if candidate_sentences else sentences[0]
    prompt = (
        "You are an instruction-following assistant. From the following sentence, generate a clear multiple-choice "
        "question (1 correct answer + 3 distractors). Return strictly a JSON object with keys: question, options (list of 4), correct_answer.\n\n"
        f"Sentence: {answer_sentence}\n\nJSON:"
    )
    try:
        generated = call_granite(prompt, max_new_tokens=180)
        # Try to parse JSON inside model output (tolerant)
        start = generated.find('{'); end = generated.rfind('}') + 1
        if start != -1 and end != -1:
            jtext = generated[start:end]
            j = json.loads(jtext)
            options = j.get('options') or j.get('choices') or []
            correct = j.get('correct_answer') or j.get('answer')
            question_text = j.get('question') or f"What is referred to by: {answer_sentence[:120]}?"
            if options and correct:
                return {
                    'type': 'mcq',
                    'question': question_text,
                    'options': options,
                    'correct_answer': correct,
                    'difficulty': difficulty,
                    'source': metadata.get('source', 'Unknown'),
                    'page': metadata.get('page', 'N/A'),
                    'explanation': f"From source: {metadata.get('source')} (Page {metadata.get('page')})"
                }
    except Exception as e:
        st.info(f"MCQ generation via model failed: {e}")
    # Fallback naive MCQ
    words = [w.strip(".,;:()[]") for w in answer_sentence.split() if len(w) > 3]
    if len(words) < 3:
        return None
    correct_answer = ' '.join(random.sample(words, min(3, len(words))))
    distractors = [
        "None of the above",
        correct_answer[::-1],
        ' '.join(random.sample(words, min(2, len(words))))
    ]
    options = [correct_answer] + distractors
    random.shuffle(options)
    return {
        'type': 'mcq',
        'question': f"What does this refer to: {answer_sentence[:100]}?",
        'options': options,
        'correct_answer': correct_answer,
        'difficulty': difficulty,
        'source': metadata.get('source', 'Unknown'),
        'page': metadata.get('page', 'N/A'),
        'explanation': f"From source: {metadata.get('source')} (Page {metadata.get('page')})"
    }

def generate_true_false(context, metadata):
    sentences = [s.strip() for s in re.split(r'\.|\n', context) if len(s.split()) > 5]
    if not sentences:
        return None
    statement = random.choice(sentences)
    prompt = (
        "Decide whether the following statement is true or false based on the sentence. "
        "Return JSON: {\"statement\":\"...\",\"is_true\":true}\n\n"
        f"Sentence: {statement}\n\nJSON:"
    )
    try:
        generated = call_granite(prompt, max_new_tokens=120)
        start = generated.find('{'); end = generated.rfind('}') + 1
        if start != -1 and end != -1:
            j = json.loads(generated[start:end])
            is_true = bool(j.get('is_true'))
            statement_text = j.get('statement') or statement
            return {
                'type': 'true_false',
                'question': statement_text,
                'correct_answer': is_true,
                'source': metadata.get('source', 'Unknown'),
                'page': metadata.get('page', 'N/A'),
                'explanation': f"From source: {metadata.get('source')} (Page {metadata.get('page')})"
            }
    except Exception:
        pass
    # fallback random
    is_true = random.choice([True, False])
    s = statement
    if not is_true:
        if " is " in s:
            s = s.replace(" is ", " is not ", 1)
        elif " are " in s:
            s = s.replace(" are ", " are not ", 1)
    return {
        'type': 'true_false',
        'question': s,
        'correct_answer': is_true,
        'source': metadata.get('source', 'Unknown'),
        'page': metadata.get('page', 'N/A'),
        'explanation': f"From source: {metadata.get('source')} (Page {metadata.get('page')})"
    }

def generate_fill_blank(context, metadata):
    sentences = [s.strip() for s in re.split(r'\.|\n', context) if len(s.split()) > 6]
    if not sentences:
        return None
    sentence = random.choice(sentences)
    words = [w.strip(".,;:()[]") for w in sentence.split()]
    meaningful_words = [w for w in words if len(w) > 4 and w.lower() not in ['that','this','with','from']]
    if not meaningful_words:
        return None
    blank_word = random.choice(meaningful_words)
    question = sentence.replace(blank_word, "______", 1)
    return {
        'type': 'fill_blank',
        'question': question,
        'correct_answer': blank_word,
        'source': metadata.get('source', 'Unknown'),
        'page': metadata.get('page', 'N/A'),
        'explanation': f"From source: {metadata.get('source')} (Page {metadata.get('page')})"
    }

# ===================== QUIZ RESULT ANALYSIS, STUDY PLAN, FLASHCARDS =====================
def analyze_quiz_results(quiz_questions, user_answers):
    score = 0
    total = len(quiz_questions)
    weak_topics_detected = []
    for i, question in enumerate(quiz_questions):
        user_answer = user_answers.get(i)
        correct = False
        if question['type'] == 'mcq':
            correct = user_answer == question['correct_answer']
        elif question['type'] == 'true_false':
            correct = user_answer == question['correct_answer']
        elif question['type'] == 'fill_blank':
            correct = user_answer and user_answer.lower().strip() == question['correct_answer'].lower().strip()
        if correct:
            score += 1
        else:
            weak_topics_detected.append({
                'question': question['question'],
                'source': question['source'],
                'page': question['page']
            })
            st.session_state.weak_topics[question['source']] += 1
    result = {
        'score': score,
        'total': total,
        'percentage': (score / total) * 100 if total > 0 else 0,
        'weak_topics': weak_topics_detected,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    st.session_state.quiz_results.append(result)
    return result

def generate_study_plan(days_until_exam, weak_topics):
    study_plan = []
    if days_until_exam <= 0:
        return [{"day": 1, "tasks": ["Review all materials immediately"]}]
    weak_sources = sorted(weak_topics.items(), key=lambda x: x[1], reverse=True)
    tasks_per_day = max(1, len(weak_sources) // days_until_exam) if weak_sources else 1
    for day in range(1, days_until_exam + 1):
        daily_tasks = []
        start_idx = (day - 1) * tasks_per_day
        end_idx = start_idx + tasks_per_day
        for source, count in weak_sources[start_idx:end_idx]:
            daily_tasks.append(f"📖 Review: {source} (Focus: {count} weak areas)")
        if day == days_until_exam:
            daily_tasks.append("✅ Final revision of all topics")
            daily_tasks.append("🧪 Take practice test")
        study_plan.append({
            'day': day,
            'date': (datetime.now() + timedelta(days=day-1)).strftime("%Y-%m-%d"),
            'tasks': daily_tasks if daily_tasks else ["📚 General review"]
        })
    return study_plan

def generate_flashcards():
    flashcards = []
    knowledge = st.session_state.knowledge_graph
    for defn in knowledge.get('definitions', [])[:20]:
        flashcards.append({
            'front': f"What is {defn['term']}?",
            'back': defn['definition'],
            'source': defn['source']
        })
    for formula in knowledge.get('formulas', [])[:10]:
        flashcards.append({
            'front': "Write the formula",
            'back': formula['formula'],
            'source': formula['source']
        })
    return flashcards
def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    
    .achievement-badge {
        display: inline-block;
        background: #ffd700;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .quiz-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .progress-ring {
        transform: rotate(-90deg);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)
# ===================== Streamlit UI =====================
def main():
    st.set_page_config(page_title="StudyMate - Granite 3.3 (2B) powered", page_icon="🎓", layout="wide")
    st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: bold; text-align:center; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">🎓 StudyMate — Granite 3.3 (2B) Instruct</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/brain.png", width=80)
        st.title("📚 Navigation")
        page = st.radio("Go to", [
            "🏠 Dashboard", "📤 Upload Materials", "🧠 Knowledge Graph", "📝 Quiz Mode",
            "📊 Analytics", "📅 Study Plan", "🃏 Flashcards", "💬 Ask Questions"
        ])

    # Default page content
    if page == "🏠 Dashboard":
        st.header("📊 Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Files Uploaded", len(st.session_state.uploaded_files_metadata))
        with col2:
            st.metric("Quizzes Taken", len(st.session_state.quiz_results))
        with col3:
            avg_score = sum([r['percentage'] for r in st.session_state.quiz_results]) / len(st.session_state.quiz_results) if st.session_state.quiz_results else 0
            st.metric("Avg Score", f"{avg_score:.1f}%")
        with col4:
            st.metric("Weak Topics", len(st.session_state.weak_topics))
        st.markdown("---")
        st.subheader("Recent Activity")
        if st.session_state.quiz_results:
            for r in st.session_state.quiz_results[-3:]:
                st.info(f"{r['score']}/{r['total']} ({r['percentage']:.1f}%) - {r['timestamp']}")
        else:
            st.info("No quizzes yet. Upload materials and generate a quiz.")

    elif page == "📤 Upload Materials":
        st.header("Upload Materials")
        uploaded_files = st.file_uploader("Upload", type=["pdf","pptx","ppt","docx","png","jpg","jpeg"], accept_multiple_files=True)
        if st.button("Process Materials"):
            if not uploaded_files:
                st.error("Please upload at least one file.")
            else:
                with st.spinner("Processing..."):
                    text_data = process_uploaded_files(uploaded_files)
                    if text_data:
                        extract_knowledge_structure(text_data)
                        chunks = get_text_chunks_with_metadata(text_data)
                        try:
                            create_vector_store(chunks)
                            st.success(f"Processed {len(uploaded_files)} file(s).")
                            st.balloons()
                        except Exception as e:
                            st.error(f"Vector store error: {e}")
                    else:
                        st.warning("No text extracted from uploaded files.")
        if st.session_state.uploaded_files_metadata:
            st.markdown("---")
            for f in st.session_state.uploaded_files_metadata:
                st.write(f"📄 **{f['name']}** ({f['type']}) - {f['uploaded_at']}")

    elif page == "🧠 Knowledge Graph":
        st.header("Knowledge Graph")
        kg = st.session_state.knowledge_graph
        if not kg:
            st.info("Process materials to extract knowledge.")
        else:
            tab1, tab2, tab3 = st.tabs(["Definitions","Formulas","Key Terms"])
            with tab1:
                for d in kg.get('definitions', []):
                    with st.expander(d['term']):
                        st.write(d['definition'])
                        st.caption(f"Source: {d['source']} (Page {d['page']})")
            with tab2:
                for f in kg.get('formulas', []):
                    st.code(f['formula'])
                    st.caption(f"Source: {f['source']} (Page {f['page']})")
            with tab3:
                st.write(", ".join(kg.get('key_terms', [])[:200]))

    elif page == "📝 Quiz Mode":
        st.header("Adaptive Quiz Generator")
        col1, col2 = st.columns(2)
        with col1:
            difficulty = st.selectbox("Difficulty", ["easy","medium","hard"])
            num_questions = st.slider("Number of Questions", 3, 20, 5)
        with col2:
            question_types = st.multiselect("Question Types", ["mcq","true_false","fill_blank"], default=["mcq","true_false"])
        if st.button("Generate Quiz"):
            with st.spinner("Generating..."):
                quiz = generate_quiz(difficulty, num_questions, question_types)
                st.session_state.current_quiz = quiz
                st.session_state.user_answers = {}
        if 'current_quiz' in st.session_state and st.session_state.current_quiz:
            quiz = st.session_state.current_quiz
            st.markdown("---")
            for i, q in enumerate(quiz):
                st.subheader(f"Q{i+1}")
                st.write(q['question'])
                st.caption(f"Source: {q['source']} (Page {q['page']})")
                if q['type'] == 'mcq':
                    ans = st.radio(f"Answer Q{i+1}", q['options'], key=f"q{i}")
                    st.session_state.user_answers[i] = ans
                elif q['type'] == 'true_false':
                    ans = st.radio(f"Answer Q{i+1}", [True, False], key=f"q{i}")
                    st.session_state.user_answers[i] = ans
                else:
                    ans = st.text_input(f"Answer Q{i+1}", key=f"q{i}")
                    st.session_state.user_answers[i] = ans
            if st.button("Submit Quiz"):
                result = analyze_quiz_results(quiz, st.session_state.user_answers)
                st.success(f"Score: {result['score']}/{result['total']} ({result['percentage']:.1f}%)")
                if result['weak_topics']:
                    st.subheader("Areas to Improve")
                    for w in result['weak_topics']:
                        st.error(f"{w['question'][:180]}... (Source: {w['source']})")

    elif page == "📊 Analytics":
        st.header("Analytics")
        if not st.session_state.quiz_results:
            st.info("No quiz data.")
        else:
            scores = [r['percentage'] for r in st.session_state.quiz_results]
            st.line_chart(scores)
            st.subheader("Weak Areas")
            if st.session_state.weak_topics:
                for topic, count in sorted(st.session_state.weak_topics.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"{topic}: {count}")

    elif page == "📅 Study Plan":
        st.header("Study Plan")
        days = st.number_input("Days until exam", 1, 90, 7)
        if st.button("Generate Plan"):
            plan = generate_study_plan(days, st.session_state.weak_topics)
            for p in plan:
                with st.expander(f"Day {p['day']} - {p['date']}"):
                    for t in p['tasks']:
                        st.write("- " + t)

    elif page == "🃏 Flashcards":
        st.header("Flashcards")
        if st.button("Generate Flashcards"):
            st.session_state.flashcards = generate_flashcards()
        if 'flashcards' in st.session_state and st.session_state.flashcards:
            for i, c in enumerate(st.session_state.flashcards):
                with st.expander(f"Card {i+1}"):
                    st.markdown(f"**Front:** {c['front']}")
                    if st.button(f"Reveal {i}", key=f"rev{i}"):
                        st.info(f"**Back:** {c['back']}")
                        st.caption(f"Source: {c['source']}")

    elif page == "💬 Ask Questions":
        st.header("Ask Questions (Granite-powered)")
        q = st.text_input("Ask about your materials")
        mode = st.selectbox("Mode", ["Smart Answer", "Search Only", "ELI5"])
        if q:
            try:
                vs = load_vector_store()
                docs = vs.similarity_search(q, k=5)
                if not docs:
                    st.warning("No relevant docs. Make sure you processed materials and created vector store.")
                else:
                    with st.expander("Relevant sections"):
                        for idx, d in enumerate(docs, 1):
                            st.markdown(f"**Section {idx}** — {d.metadata.get('source','Unknown')} (Page {d.metadata.get('page','N/A')})")
                            st.write(d.page_content[:350]+"...")
                    context = " ".join([d.page_content for d in docs[:3]])
                    if mode == "Smart Answer":
                        prompt = (
                            "You are an assistant. Using the context below, answer the question concisely and cite the context source names.\n\n"
                            f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"
                        )
                        with st.spinner("Asking Granite..."):
                            ans = call_granite(prompt, max_new_tokens=300)
                            st.success(ans)
                    elif mode == "ELI5":
                        prompt = f"Explain this to a beginner in simple terms: {q}\n\nContext: {context}\n\nAnswer simply:"
                        ans = call_granite(prompt, max_new_tokens=220, temperature=0.3)
                        st.info(ans)
                    else:
                        st.info("Search results shown above.")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
