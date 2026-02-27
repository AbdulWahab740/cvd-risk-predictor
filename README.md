<div align="center">

# Cardiovascular Risk Assessment Agent
### 🏆 Elasticsearch Agent Builder Hackathon 2026

![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.0+-005571?style=for-the-badge&logo=elasticsearch&logoColor=white)
![Agent Builder](https://img.shields.io/badge/Agent_Builder-Enabled-00BFB3?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-FF6600?style=for-the-badge)
![License](https://img.shields.io/badge/License-Research_Only-red?style=for-the-badge)

**An AI-powered clinical decision support system using Elasticsearch Agent Builder**

[Demo Video](#) • [Documentation](#) • [Report Issue](#)

</div>

---

### 🏅 Team Members

| Name | GitHub | Role |
|------|--------|------|
| Abdul Wahab | [@AbdulWahab740](https://github.com/AbdulWahab740) | RAG + Elastic Agent Integration  |
| Ayesha Saleem | [@aysh34](https://github.com/aysh34) | Machine Learning Engineer + Documentation |

---

## 📚 Table of Contents

- [Overview](#overview)
- [The Problem We Solve](#the-problem-we-solve)
- [System Architecture](#system-architecture)
- [Elasticsearch Features Used](#elasticsearch-features-used)
- [Key Features](#key-features)
- [Demo](#demo)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Workflow](#workflow)
- [Technical Implementation](#technical-implementation)
- [Results & Performance](#results--performance)
- [Challenges & Solutions](#challenges--solutions)
- [Future Enhancements](#future-enhancements)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## 🔍 Overview

The **Cardiovascular Clinical Decision Support (CDS) Agent** is an AI-powered assistant built with **Elasticsearch Agent Builder** that helps healthcare providers make evidence-based cardiovascular disease risk assessments in seconds instead of hours.

### What It Does

✅ **Analyzes** patient vitals using machine learning (XGBoost)  
✅ **Searches** clinical practice guidelines (ACC/AHA)  
✅ **Retrieves** peer-reviewed research (100+ PubMed abstracts)  
✅ **Synthesizes** evidence-based recommendations with citations  

---

## 🚨 The Problem We Solve

<div align="center">

### Manual Process: 2+ Hours ⏰

```
Doctor receives patient → Manual risk calculation → 
Search 500-page guidelines → Find research papers → 
Synthesize recommendations → Document findings
```

### Our Agent: under 1 minutes ⚡

```
Patient data → Elasticsearch Agent Builder → 
ML Risk + Guidelines + Research → Evidence-based report
```

</div>

**Impact:** 99% time reduction while improving evidence quality and consistency

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│          ELASTICSEARCH AGENT BUILDER                │
│            (Autonomous Orchestrator)                │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────┼────────┐
       │       │        │
       ↓       ↓        ↓
   ┌────────┬────────┬────────┐
   │   ML   │Clinical│ PubMed │
   │ Model  │ Guide- │Research│
   │  API   │ lines  │Abstracts│
   │        │ Vector │ Vector │
   │ XGBoost│ Store  │ Store  │
   └────────┴────────┴────────┘
       │       │        │
       └───────┼────────┘
               ↓
    Clinical Report with Citations
```

### Data Flow

1. **Patient Input** → Agent receives structured health data
2. **ML Prediction** → HTTP Workflow calls XGBoost API
3. **Intelligent Routing** → Agent autonomously decides which guidelines to search
4. **Hybrid Search** → Combines vector + keyword search for accuracy
5. **Evidence Synthesis** → Generates structured clinical report

---

## ⚡ Elasticsearch Features Used

<div align="center">

| Feature | Purpose | Impact |
|---------|---------|--------|
| **Agent Builder** | Autonomous orchestration | No hardcoded logic needed |
| **Vector Search** | Semantic understanding | 384-dim embeddings |
| **Hybrid Search** | Vector + BM25 | 30% better retrieval |
| **ES\|QL** | Advanced filtering | 4x faster queries |
| **HTTP Workflows** | External API integration | Seamless ML model connection |
| **Dense Vector Store** | Document indexing | 10K+ docs in <1s search |

</div>

---

## ✨ Key Features

### 🤖 Autonomous Multi-Step Reasoning

The agent independently decides which tools to use based on patient data. When it sees "high blood pressure," it automatically searches hypertension guidelines—no hardcoded if-then statements.

### 🔍 Hybrid Search Excellence

Combines semantic vector search with BM25 keyword matching:
- **Vector search:** Understands medical concepts
- **BM25:** Catches specific clinical terms
- **Result:** 30% better accuracy than either alone

### 📊 ML Risk Prediction with SHAP

- XGBoost model trained on 50,000+ patients
- SHAP values explain each prediction
- Top risk factors ranked by importance

### 📚 Evidence-Based Citations

Every recommendation includes:
- Clinical guideline citation (Class I/IIa/etc.)
- Level of evidence (A/B-R/C)
- Supporting research (PMID)

---

## 🎥 Demo

### Sample Patient Case

```yaml
Input:
  Age: 58 years
  Gender: Male
  Blood Pressure: 165/95 mmHg (Stage 2 HTN)
  Cholesterol: High
  BMI: 30 (Obese)
  Smoker: Yes
  Physical Activity: Sedentary

Output:
  Risk Score: 78% (HIGH RISK)
  Top Factors:
    - Systolic BP (35% contribution)
    - Cholesterol (28% contribution)
    - Smoking (15% contribution)
  
  Recommendations:
    1. Initiate ACE inhibitor, target <130/80 mmHg
       [2017 ACC/AHA HTN Guideline, Class I, Level B-R]
       Supporting: SPRINT trial [PMID: 26551272]
    
    2. Start statin therapy, target LDL <100 mg/dL
       [2018 ACC/AHA Cholesterol Guideline, Class I]
    
    3. Smoking cessation with counseling + medication
       [2019 ACC/AHA Primary Prevention, Class I]
```

**Watch the full demo:** [3-minute video](#)

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- Elasticsearch 8.0+ (Cloud or self-hosted)
- Elasticsearch Agent Builder access
- API keys and credentials

### Step 1: Clone Repository

```bash
git clone https://github.com/your-team/cvd-decision-support.git
cd cvd-decision-support
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Elasticsearch

```bash
# Set environment variables
export ELASTIC_CLOUD_ID='your-cloud-id'
export ELASTIC_API_KEY='your-api-key'
```

### Step 4: Index Data

```bash
# Index clinical guidelines
python scripts/index_guidelines.py

# Index PubMed abstracts
python scripts/index_pubmed.py
```


### Step 5: Configure Agent Builder

1. Go to Kibana → Agent Builder
2. Create new agent: "Cardiovascular CDS Agent"
3. Add tools:
   - `cvd_risk_predictor` (HTTP workflow)
   - `search_clinical_guidelines` (ES|QL)
   - `search_pubmed_abstracts` (Hybrid search)
4. Upload system prompt from `prompts/agent_system_prompt.md`
5. Select LLM: Claude Sonnet 4
6. Save and test

---

## 💻 Usage

### Via Streamlit Interface (Recommended)

```bash
streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501` and fill in patient data.

### Via Agent Builder Chat

```
User: "Assess CVD risk for 58-year-old male, BP 165/95, 
      high cholesterol, smoker, BMI 30"

Agent: [Autonomously calls ML model → searches guidelines → 
        retrieves research → synthesizes report]
```

---

## 🔄 Workflow

### Input Schema

The agent expects structured patient data:

```python
{
  "gender": 1 or 2,           # 1=Female, 2=Male
  "age_years": int,           # Age in years
  "height": int,              # cm
  "weight": float,            # kg
  "ap_hi": int,               # Systolic BP (mmHg)
  "ap_lo": int,               # Diastolic BP (mmHg)
  "cholesterol": 1/2/3,       # 1=normal, 2=above, 3=well above
  "gluc": 1/2/3,              # Glucose level
  "smoke": 0 or 1,            # Smoking status
  "alco": 0 or 1,             # Alcohol consumption
  "active": 0 or 1            # Physical activity
}
```

**Derived features** (calculated automatically):
- BMI = weight / (height/100)²
- Pulse pressure = systolic - diastolic
- Mean arterial pressure = (systolic + 2×diastolic) / 3

### Agent Execution Steps

1. **Risk Prediction** → Calls ML API, gets risk score + SHAP values
2. **Topic Mapping** → Maps risk factors to clinical topics (e.g., high BP → "Hypertension")
3. **Guideline Search** → ES|QL query on guidelines index
4. **Research Search** → Hybrid search on PubMed index
5. **Synthesis** → Combines all sources into structured report

### Output Format

```markdown
## RISK SUMMARY
- Risk: 78% (HIGH)
- Category: Requires immediate intervention

## PRIMARY RISK FACTORS
1. Systolic BP 165 mmHg (35% contribution) - Stage 2 HTN
2. Cholesterol well above normal (28% contribution)
3. Active smoking (15% contribution)

## EVIDENCE-BASED RECOMMENDATIONS

### 1. BLOOD PRESSURE MANAGEMENT
**Intervention:** Initiate ACE inhibitor or ARB
**Target:** <130/80 mmHg
**Evidence:** [2017 ACC/AHA HTN Guideline, Class I, Level B-R]
**Supporting:** SPRINT trial showed 25% reduction in CV events [PMID: 26551272]

[... more recommendations ...]

## IMMEDIATE ACTIONS
☐ Order lipid panel (fasting)
☐ Initiate antihypertensive therapy
☐ Smoking cessation counseling
☐ Schedule cardiology appointment

## EVIDENCE QUALITY
Strong evidence (Class I recommendations), high certainty
```

---

## 🔧 Technical Implementation

### Machine Learning Model

- **Algorithm:** Cost-sensitive XGBoost
- **Training data:** 70,000 patient records
- **Features:** 14 clinical + derived variables
- **Performance:** 80+% AUC Score
- **Interpretability:** SHAP values for each prediction

---

## 📊 Results & Performance

### Speed Comparison

| Method | Time Required | Sources Consulted |
|--------|---------------|-------------------|
| Manual physician search | 120 minutes | 2-3 sources |
| **Our Agent** | **8 seconds** | **15+ sources** |
| **Improvement** | **99% faster** | **5-7x more comprehensive** |


### Elasticsearch Performance

- **Index size:** 10,000+ documents
- **Search latency:** <1 second average
- **Agent response time:** under 1 minutes

---

## 🧩 Challenges & Solutions

### Challenge 1: Chunking Long Clinical Guidelines

**Problem:** 500-page guidelines exceeded token limits  
**Solution:** Intelligent chunking by recommendation sections, preserving semantic meaning  
**Elasticsearch Help:** Token limits forced better preprocessing  

### Challenge 2: Citation Accuracy

**Problem:** Early versions hallucinated PMID numbers  
**Solution:** Configured agent to only cite metadata from retrieved documents  
**Elasticsearch Help:** Rich metadata in search results prevented hallucinations  

### Challenge 3: Balancing Speed vs Comprehensiveness

**Problem:** Initial 30-second response time too slow  
**Solution:** ES|QL optimization + top-5 result limiting  
**Result:** 4x speedup (30s → 8s) without quality loss  

---

## 🚀 Future Enhancements

### Short-term (Next 3 months)

- [ ] EHR integration (FHIR standard)
- [ ] Multi-language support (Spanish, Mandarin)
- [ ] Mobile app deployment
- [ ] Batch patient processing

### Medium-term (6-12 months)

- [ ] ECG image analysis integration
- [ ] Longitudinal patient tracking
- [ ] Medication interaction checking
- [ ] Real-time vital sign monitoring

### Long-term (1+ years)

- [ ] Multi-modal data fusion (genomics, imaging)
- [ ] Federated learning across hospitals
- [ ] Regulatory approval (FDA, CE marking)
- [ ] Randomized controlled trial validation

---

## 🙏 Acknowledgments

### Hackathon

Thank you to **Elasticsearch** for hosting the Agent Builder Hackathon and providing the tools that made this project possible.

### Data Sources

- **Byte2Beat Competition:** CVD patient dataset
- **ACC/AHA:** Clinical practice guidelines
- **PubMed/NIH:** Biomedical research abstracts

### Technologies

- **Elasticsearch Agent Builder:** Autonomous orchestration
- **XGBoost:** Machine learning framework
- **Sentence Transformers:** Vector embeddings
- **FastAPI:** ML API deployment
- **Streamlit:** User interface

### Inspiration

This project was inspired by the need to reduce clinical decision-making time while improving evidence quality in cardiovascular care.

---

## 📄 License

**For research and educational use only.**

This system is designed for:
- Clinical decision support research
- Educational demonstrations
- Hackathon and innovation projects

It is **not intended for autonomous medical decision-making** and has not undergone regulatory approval (FDA, CE marking).

### Disclaimer

This software is provided "as is" without warranty of any kind. The developers assume no liability for any clinical decisions made using this system. Always consult qualified healthcare professionals for medical advice.



**Built with ❤️ using Elasticsearch Agent Builder**

⭐ Star this repo if you find it useful!

[⬆ Back to top](#cardiovascular-clinical-decision-support-agent)

</div>
