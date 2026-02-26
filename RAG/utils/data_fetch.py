from Bio import Entrez
import time
from xml.etree import ElementTree as ET # For XML parsing
import os
import re
import json

output_dir = "E:/Python/FastAPi/ClinicalDecisionSupportSystem/RAG/utils/data" 
Entrez.email = "abdulwahab41467@gmail.com" # Replace with your email

def search_pubmed_paginated(query, total_to_fetch=1000):
    """Search PubMed with pagination to fetch a specified number of PMIDs."""
    all_pmids = []
    retmax_per_call = 1000 # Max results per ESearch call
    retstart = 0

    print(f"Starting paginated search for up to {total_to_fetch} PMIDs for query: {query}")
    print(f"DEBUG: Actual query string being sent to Entrez:\n---\n{query}\n---") # <--- ADD THIS LINE FOR DEBUGGING
    
    while len(all_pmids) < total_to_fetch:
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query, # This is the 'query' variable we're inspecting
                retmax=str(retmax_per_call),
                retstart=str(retstart),
                retmode="xml",
                tool="MediCritiqueAI",
                email=Entrez.email
            )            
            record = Entrez.read(handle)
            handle.close()

            current_batch_ids = record["IdList"]
            if not current_batch_ids:
                print("No more PMIDs found in current batch. Ending search.")
                break

            all_pmids.extend(current_batch_ids)
            retstart += len(current_batch_ids)
            print(f"  Fetched {len(current_batch_ids)} PMIDs. Total fetched so far: {len(all_pmids)}. Next start index: {retstart}")
            time.sleep(0.5) # Crucial: respect rate limits

            if record.get("Count", "0") and int(record["Count"]) <= len(all_pmids):
                print(f"Reached total count ({record['Count']}) for the query or desired fetch limit.")
                break # All available PMIDs or target fetched

        except Exception as e:
            print(f"Error during PubMed paginated search at retstart {retstart}: {e}")
            break 

    return all_pmids[:total_to_fetch], record.get("Count", "0") 

def get_pmc_ids(pmids):
    """Fetch PMC IDs for a list of PMIDs using ELink in batches."""
    pmc_mapping = {}
    batch_size = 200 
    for i in range(0, len(pmids), batch_size):
        batch_pmids = pmids[i : i + batch_size]
        id_list_str = ",".join(batch_pmids)
        
        try:
            handle = Entrez.elink(
                dbfrom="pubmed",
                db="pmc",
                id=id_list_str,
                retmode="xml",
                tool="MediCritiqueAI",
                email=Entrez.email
            )
            records = Entrez.read(handle)
            handle.close()

            for record in records:
                pubmed_id = record['IdList'][0]
                if 'LinkSetDb' in record:
                    for link_set_db in record['LinkSetDb']:
                        if link_set_db['DbTo'] == 'pmc':
                            for link in link_set_db['Link']:
                                pmc_mapping[pubmed_id] = link['Id']
                                break
                time.sleep(0.1) 
        except Exception as e:
            print(f"Error in elink batch for PMIDs {batch_pmids[0]}-{batch_pmids[-1]}: {e}")
    return pmc_mapping

# Function to download PMC full text XML
def download_pmc_full_text_xml(pmcid):
    """Download full text XML for a given PMC ID."""
    if not pmcid:
        return None
    try:
        handle = Entrez.efetch(
            db="pmc",
            id=pmcid,
            retmode="xml",
            tool="MediCritiqueAI",
            email=Entrez.email
        )
        xml_content = handle.read().decode('utf-8')
        handle.close()
        return xml_content
    except Exception as e:
        print(f"Error downloading PMC XML for PMC{pmcid}: {e}")
        return None

def sanitize_filename(title):
    """core_topic = core_topics()
"""
    # Replace invalid characters with an underscore
    s = re.sub(r'[\\/:*?"<>|]', '_', title)
    s = s.strip()
    # Replace multiple spaces with a single underscore
    s = re.sub(r'\s+', '_', s)
    words = s.split('_')
    words = " ".join(words)
    return words

def core_topics():
    return {

    "Coronary_Artery_Disease": {
        "query": """
(
    ("Coronary Artery Disease"[MeSH] OR "Coronary Disease"[MeSH] OR "Myocardial Ischemia"[MeSH] OR
     "coronary artery disease"[tiab] OR "CAD"[tiab] OR "ischemic heart disease"[tiab] OR
     "coronary heart disease"[tiab] OR "CHD"[tiab] OR "angina pectoris"[tiab])
    AND ("randomized controlled trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR
         "clinical trial"[pt] OR "journal article"[pt])
    AND (
        treatment[tiab] OR therapy[tiab] OR intervention[tiab] OR management[tiab] OR
        outcome[tiab] OR prognosis[tiab] OR revascularization[tiab] OR stenting[tiab] OR
        "percutaneous coronary intervention"[tiab] OR "PCI"[tiab] OR "bypass surgery"[tiab] OR
        "CABG"[tiab] OR thrombolysis[tiab] OR antiplatelet[tiab] OR statin[tiab] OR
        "lipid lowering"[tiab] OR "cardiac rehabilitation"[tiab]
    )
    AND ("2010/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR "in vitro"[tiab] OR "cell culture"[tiab])
)
        """,
        "target_full_text_count": 2000,
        "aliases": ["CAD", "Ischemic Heart Disease", "Coronary Heart Disease", "CHD", "Angina",
                    "Atherosclerosis", "Coronary Stenosis", "Stable Angina", "Unstable Angina"]
    },

    "Heart_Failure": {
        "query": """
(
    ("Heart Failure"[MeSH] OR "Cardiac Failure"[MeSH] OR "Ventricular Dysfunction"[MeSH] OR
     "heart failure"[tiab] OR "HF"[tiab] OR "cardiac failure"[tiab] OR
     "congestive heart failure"[tiab] OR "CHF"[tiab] OR
     "HFrEF"[tiab] OR "HFpEF"[tiab] OR "HFmrEF"[tiab] OR
     "reduced ejection fraction"[tiab] OR "preserved ejection fraction"[tiab])
    AND ("randomized controlled trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR
         "clinical trial"[pt])
    AND (
        treatment[tiab] OR therapy[tiab] OR management[tiab] OR intervention[tiab] OR
        hospitalization[tiab] OR mortality[tiab] OR "ejection fraction"[tiab] OR
        "LVEF"[tiab] OR "cardiac output"[tiab] OR
        "ACE inhibitor"[tiab] OR "beta blocker"[tiab] OR "diuretic"[tiab] OR
        "sacubitril"[tiab] OR "valsartan"[tiab] OR "ARNI"[tiab] OR
        "SGLT2 inhibitor"[tiab] OR "dapagliflozin"[tiab] OR "empagliflozin"[tiab] OR
        "cardiac resynchronization"[tiab] OR "CRT"[tiab] OR "ICD"[tiab] OR
        "device therapy"[tiab] OR "cardiac transplant"[tiab] OR "LVAD"[tiab]
    )
    AND ("2010/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR "in vitro"[tiab])
)
        """,
        "target_full_text_count": 2000,
        "aliases": ["HF", "CHF", "Congestive Heart Failure", "HFrEF", "HFpEF",
                    "Systolic Heart Failure", "Diastolic Heart Failure", "Cardiac Failure",
                    "Left Ventricular Dysfunction"]
    },

    "Acute_Myocardial_Infarction": {
        "query": """
(
    ("Myocardial Infarction"[MeSH] OR "ST Elevation Myocardial Infarction"[MeSH] OR
     "Non-ST Elevated Myocardial Infarction"[MeSH] OR
     "myocardial infarction"[tiab] OR "heart attack"[tiab] OR
     "STEMI"[tiab] OR "NSTEMI"[tiab] OR "acute coronary syndrome"[tiab] OR "ACS"[tiab])
    AND ("randomized controlled trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR
         "clinical trial"[pt] OR "journal article"[pt])
    AND (
        reperfusion[tiab] OR thrombolysis[tiab] OR "primary PCI"[tiab] OR
        "percutaneous coronary intervention"[tiab] OR fibrinolysis[tiab] OR
        anticoagulation[tiab] OR antiplatelet[tiab] OR "dual antiplatelet"[tiab] OR
        "DAPT"[tiab] OR "door-to-balloon"[tiab] OR
        "cardiac biomarker"[tiab] OR troponin[tiab] OR "CK-MB"[tiab] OR
        mortality[tiab] OR outcome[tiab] OR complication[tiab] OR
        "cardiogenic shock"[tiab] OR "mechanical complication"[tiab]
    )
    AND ("2010/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR "in vitro"[tiab])
)
        """,
        "target_full_text_count": 2000,
        "aliases": ["MI", "Heart Attack", "STEMI", "NSTEMI", "ACS", "Acute Coronary Syndrome",
                    "Myocardial Infarction", "Cardiac Infarction"]
    },

    "Hypertension": {
        "query": """
(
    ("Hypertension"[MeSH] OR "Blood Pressure, High"[MeSH] OR "Essential Hypertension"[MeSH] OR
     "Resistant Hypertension"[MeSH] OR "Hypertensive Crisis"[MeSH] OR
     "hypertension"[tiab] OR "high blood pressure"[tiab] OR "arterial hypertension"[tiab] OR
     "resistant hypertension"[tiab] OR "hypertensive emergency"[tiab] OR
     "systolic blood pressure"[tiab] OR "diastolic blood pressure"[tiab])
    AND ("randomized controlled trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR
         "clinical trial"[pt] OR "guideline"[pt])
    AND (
        treatment[tiab] OR therapy[tiab] OR management[tiab] OR "blood pressure control"[tiab] OR
        antihypertensive[tiab] OR "ACE inhibitor"[tiab] OR "ARB"[tiab] OR
        "calcium channel blocker"[tiab] OR "CCB"[tiab] OR "beta blocker"[tiab] OR
        diuretic[tiab] OR "renin inhibitor"[tiab] OR "aldosterone antagonist"[tiab] OR
        "lifestyle modification"[tiab] OR "dietary intervention"[tiab] OR
        "DASH diet"[tiab] OR "sodium restriction"[tiab] OR
        "renal denervation"[tiab] OR "target blood pressure"[tiab] OR
        "cardiovascular risk"[tiab] OR "end organ damage"[tiab]
    )
    AND ("2010/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR "in vitro"[tiab])
)
        """,
        "target_full_text_count": 2000,
        "aliases": ["High Blood Pressure", "HTN", "Essential Hypertension", "Arterial Hypertension",
                    "Resistant Hypertension", "Primary Hypertension", "Secondary Hypertension"]
    },

    "Atrial_Fibrillation": {
        "query": """
(
    ("Atrial Fibrillation"[MeSH] OR "Atrial Flutter"[MeSH] OR
     "atrial fibrillation"[tiab] OR "AF"[tiab] OR "AFib"[tiab] OR
     "paroxysmal atrial fibrillation"[tiab] OR "persistent atrial fibrillation"[tiab] OR
     "permanent atrial fibrillation"[tiab] OR "atrial flutter"[tiab])
    AND ("randomized controlled trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR
         "clinical trial"[pt] OR "journal article"[pt])
    AND (
        treatment[tiab] OR management[tiab] OR "rhythm control"[tiab] OR
        "rate control"[tiab] OR cardioversion[tiab] OR ablation[tiab] OR
        "catheter ablation"[tiab] OR "pulmonary vein isolation"[tiab] OR "PVI"[tiab] OR
        anticoagulation[tiab] OR "stroke prevention"[tiab] OR "thromboembolism"[tiab] OR
        "warfarin"[tiab] OR "NOAC"[tiab] OR "DOAC"[tiab] OR
        "apixaban"[tiab] OR "rivaroxaban"[tiab] OR "dabigatran"[tiab] OR
        "antiarrhythmic"[tiab] OR "amiodarone"[tiab] OR "dronedarone"[tiab] OR
        "CHA2DS2-VASc"[tiab] OR "stroke risk"[tiab]
    )
    AND ("2010/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR "in vitro"[tiab])
)
        """,
        "target_full_text_count": 2000,
        "aliases": ["AF", "AFib", "Atrial Flutter", "Paroxysmal AF", "Persistent AF",
                    "Permanent AF", "Supraventricular Arrhythmia", "Irregular Heartbeat"]
    },

    "Cardiomyopathy": {
        "query": """
(
    ("Cardiomyopathies"[MeSH] OR "Cardiomyopathy, Dilated"[MeSH] OR
     "Cardiomyopathy, Hypertrophic"[MeSH] OR "Cardiomyopathy, Restrictive"[MeSH] OR
     "Arrhythmogenic Right Ventricular Dysplasia"[MeSH] OR
     "cardiomyopathy"[tiab] OR "dilated cardiomyopathy"[tiab] OR "DCM"[tiab] OR
     "hypertrophic cardiomyopathy"[tiab] OR "HCM"[tiab] OR
     "restrictive cardiomyopathy"[tiab] OR "ARVC"[tiab] OR
     "takotsubo"[tiab] OR "stress cardiomyopathy"[tiab] OR "peripartum cardiomyopathy"[tiab])
    AND ("randomized controlled trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR
         "clinical trial"[pt] OR "journal article"[pt] OR "review"[pt])
    AND (
        treatment[tiab] OR management[tiab] OR therapy[tiab] OR prognosis[tiab] OR
        outcome[tiab] OR "genetic testing"[tiab] OR "family screening"[tiab] OR
        "sudden cardiac death"[tiab] OR "SCD"[tiab] OR "ICD implantation"[tiab] OR
        "heart transplantation"[tiab] OR "LVAD"[tiab] OR
        "septal reduction"[tiab] OR "myomectomy"[tiab] OR "alcohol ablation"[tiab] OR
        mavacamten[tiab] OR "beta blocker"[tiab] OR "calcium channel blocker"[tiab]
    )
    AND ("2010/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR "in vitro"[tiab])
)
        """,
        "target_full_text_count": 1500,
        "aliases": ["DCM", "HCM", "Dilated Cardiomyopathy", "Hypertrophic Cardiomyopathy",
                    "Restrictive Cardiomyopathy", "ARVC", "Takotsubo", "Stress Cardiomyopathy"]
    },

    "Valvular_Heart_Disease": {
        "query": """
(
    ("Heart Valve Diseases"[MeSH] OR "Aortic Valve Stenosis"[MeSH] OR
     "Mitral Valve Insufficiency"[MeSH] OR "Aortic Valve Insufficiency"[MeSH] OR
     "Mitral Valve Stenosis"[MeSH] OR "Tricuspid Valve Insufficiency"[MeSH] OR
     "valvular heart disease"[tiab] OR "aortic stenosis"[tiab] OR "AS"[tiab] OR
     "aortic regurgitation"[tiab] OR "mitral regurgitation"[tiab] OR "MR"[tiab] OR
     "mitral stenosis"[tiab] OR "tricuspid regurgitation"[tiab] OR
     "heart valve"[tiab] OR "valve replacement"[tiab])
    AND ("randomized controlled trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR
         "clinical trial"[pt] OR "journal article"[pt])
    AND (
        treatment[tiab] OR intervention[tiab] OR repair[tiab] OR replacement[tiab] OR
        "surgical valve replacement"[tiab] OR "TAVR"[tiab] OR "TAVI"[tiab] OR
        "transcatheter aortic valve"[tiab] OR "MitraClip"[tiab] OR "TMVR"[tiab] OR
        "balloon valvuloplasty"[tiab] OR "valve repair"[tiab] OR
        outcome[tiab] OR mortality[tiab] OR "hemodynamic"[tiab] OR
        "valve gradient"[tiab] OR "valve area"[tiab] OR "regurgitation severity"[tiab]
    )
    AND ("2010/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR "in vitro"[tiab])
)
        """,
        "target_full_text_count": 1500,
        "aliases": ["Aortic Stenosis", "Mitral Regurgitation", "Aortic Regurgitation",
                    "Mitral Stenosis", "TAVR", "TAVI", "Valve Replacement", "Valvulopathy"]
    },

    "Cardiac_Arrhythmia": {
        "query": """
(
    ("Arrhythmias, Cardiac"[MeSH] OR "Tachycardia, Ventricular"[MeSH] OR
     "Ventricular Fibrillation"[MeSH] OR "Bradycardia"[MeSH] OR
     "Long QT Syndrome"[MeSH] OR "Brugada Syndrome"[MeSH] OR
     "cardiac arrhythmia"[tiab] OR "arrhythmia"[tiab] OR
     "ventricular tachycardia"[tiab] OR "VT"[tiab] OR
     "ventricular fibrillation"[tiab] OR "VF"[tiab] OR
     "sudden cardiac arrest"[tiab] OR "SCA"[tiab] OR
     "long QT syndrome"[tiab] OR "LQTS"[tiab] OR
     "Brugada syndrome"[tiab] OR "bradycardia"[tiab] OR "heart block"[tiab])
    AND ("randomized controlled trial"[pt] OR "meta-analysis"[pt] OR "systematic review"[pt] OR
         "clinical trial"[pt] OR "journal article"[pt])
    AND (
        treatment[tiab] OR management[tiab] OR ablation[tiab] OR
        "catheter ablation"[tiab] OR "ICD"[tiab] OR "defibrillator"[tiab] OR
        "cardiac pacing"[tiab] OR "pacemaker"[tiab] OR "CRT-D"[tiab] OR
        antiarrhythmic[tiab] OR "amiodarone"[tiab] OR "sotalol"[tiab] OR
        "flecainide"[tiab] OR "mexiletine"[tiab] OR "quinidine"[tiab] OR
        "sudden cardiac death prevention"[tiab] OR "risk stratification"[tiab] OR
        "electrophysiology study"[tiab] OR "EPS"[tiab]
    )
    AND ("2010/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (animal[tiab] OR mice[tiab] OR rats[tiab] OR "in vitro"[tiab])
)
        """,
        "target_full_text_count": 1500,
        "aliases": ["Arrhythmia", "Ventricular Tachycardia", "VT", "Ventricular Fibrillation",
                    "Sudden Cardiac Arrest", "Long QT", "Brugada", "Heart Block", "Bradycardia"]
    },

    "Cardiac_Diagnostics": {
        "query": """
(
    ("Echocardiography"[MeSH] OR "Electrocardiography"[MeSH] OR
     "Cardiac Catheterization"[MeSH] OR "Coronary Angiography"[MeSH] OR
     "Cardiac Imaging Techniques"[MeSH] OR "Exercise Test"[MeSH] OR
     "Biomarkers"[MeSH] OR "Natriuretic Peptides"[MeSH] OR
     "echocardiography"[tiab] OR "ECG"[tiab] OR "electrocardiogram"[tiab] OR
     "cardiac MRI"[tiab] OR "CMR"[tiab] OR "cardiac CT"[tiab] OR "CCTA"[tiab] OR
     "coronary angiography"[tiab] OR "cardiac catheterization"[tiab] OR
     "stress test"[tiab] OR "exercise ECG"[tiab] OR
     "troponin"[tiab] OR "BNP"[tiab] OR "NT-proBNP"[tiab] OR
     "cardiac biomarker"[tiab] OR "FFR"[tiab] OR "iFR"[tiab])
    AND ("journal article"[pt] OR "review"[pt] OR "clinical trial"[pt] OR
         "validation study"[pt] OR "evaluation study"[pt] OR
         "meta-analysis"[pt] OR "systematic review"[pt] OR "guideline"[pt])
    AND (
        diagnostic[tiab] OR "diagnostic accuracy"[tiab] OR sensitivity[tiab] OR
        specificity[tiab] OR "predictive value"[tiab] OR "AUC"[tiab] OR
        "ROC curve"[tiab] OR "risk stratification"[tiab] OR
        "imaging protocol"[tiab] OR "diagnostic yield"[tiab] OR
        "clinical utility"[tiab] OR "point of care"[tiab] OR
        "early detection"[tiab] OR screening[tiab]
    )
    AND ("2000/01/01"[pdat] : "2025/12/31"[pdat])
    NOT (therapy[tiab] OR "drug treatment"[tiab] OR "pharmacological intervention"[tiab])
)
        """,
        "target_full_text_count": 2000,
        "aliases": ["Echocardiography", "ECG", "Electrocardiogram", "Cardiac MRI", "CMR",
                    "Coronary Angiography", "Cardiac CT", "CCTA", "Troponin", "BNP", "NT-proBNP",
                    "FFR", "Stress Test", "Cardiac Catheterization"]
    },

   
}
# --- Main Execution ---

if __name__ == "__main__":
        
    core_topic= core_topics()
    for topic_name, topic_info in core_topic.items():
        print(f"\n--- Processing Core Topic: {topic_name} ---")
        topic_query = topic_info["query"]
        target_full_text_count = topic_info["target_full_text_count"]
        
        topic_output_dir = os.path.join(output_dir, topic_name)
        os.makedirs(topic_output_dir, exist_ok=True)
        print(f"Output directory for {topic_name}: {topic_output_dir}")

        # Fetch a larger pool of PMIDs, as only a subset will have PMC full text
        pmids_to_fetch_from_search = max(target_full_text_count * 5, 500) # Fetch at least 500, or 5x target
        all_pmids, total_search_count = search_pubmed_paginated(topic_query, total_to_fetch=pmids_to_fetch_from_search) 
        print(f"\nOverall, PubMed search found {total_search_count} articles for '{topic_name}'. Will process {len(all_pmids)} unique PMIDs.")

        # Step 2: Map PMIDs to PMCIDs
        print(f"\nMapping PMIDs to PMCIDs for '{topic_name}'...")
        pmid_to_pmcid = get_pmc_ids(all_pmids)
        pmids_with_pmc = [pmid for pmid, pmcid in pmid_to_pmcid.items() if pmcid]
        print(f"Found {len(pmids_with_pmc)} PMIDs with associated PMCIDs (potential full text) for '{topic_name}'.")

        # Step 3: Download and process full-text XML from PMC
        downloaded_count = 0
        processed_pmids_set = set() # Reset for each topic

        print(f"\nAttempting to download up to {target_full_text_count} full-text XML articles for '{topic_name}'...")
        for pmid in pmids_with_pmc:
            if downloaded_count >= target_full_text_count:
                print(f"Reached target of {target_full_text_count} full-text articles for '{topic_name}'.")
                break
            
            if pmid not in processed_pmids_set:
                pmcid = pmid_to_pmcid[pmid]
                # Attempt to get XML data first, to extract title
                xml_data = download_pmc_full_text_xml(pmcid)
                
                if xml_data:
                    try:
                        root = ET.fromstring(xml_data)
                        article_title_elem = root.find(".//article-meta/title-group/article-title")
                        article_title = article_title_elem.text if article_title_elem is not None else f"No_Title_PMCID_{pmcid}"
                        
                        # --- Extract Main Abstract Text ---
                        main_abstract_parts = []
                        for abstract_elem in root.findall(".//abstract"):
                            abstract_type = abstract_elem.get('abstract-type')
                            if abstract_type is None or abstract_type not in ['graphical', 'web', 'short-communication-abstract']:
                                for p_elem in abstract_elem.findall("p"):
                                    if p_elem.text:
                                        main_abstract_parts.append(p_elem.text.strip())
                        article_abstract = "\n".join(main_abstract_parts) if main_abstract_parts else ""
                        
                        # --- Extract Full Text (from body) ---
                        full_text_parts = []
                        body_elem = root.find(".//body")
                        if body_elem is not None:
                            for paragraph in body_elem.findall(".//p"):
                                if paragraph.text:
                                    full_text_parts.append(paragraph.text.strip())
                        
                        article_full_text = "\n".join(full_text_parts) if full_text_parts else ""

                        content_to_chunk = article_full_text
                        if not content_to_chunk:
                            content_to_chunk = article_abstract
                            print(f"  WARNING: No full text found for PMC{pmcid}. Using abstract as main content for {topic_name}.")
                        
                        if not content_to_chunk: # If still no content, skip or note as very sparse
                            print(f"  WARNING: No content (full text or abstract) found for PMC{pmcid}. Skipping for {topic_name}.")
                            continue # Skip to the next PMID
                            
                        sanitized_title = sanitize_filename(article_title)
                        xml_filename_base = f"{sanitized_title}_PMC{pmcid}"

                        
                        xml_filename = os.path.join(topic_output_dir, f"{xml_filename_base}.xml")
                        
                        # 1. Save the raw XML
                        with open(xml_filename, 'w', encoding='utf-8') as f:
                            f.write(xml_data)

                        try:
                            pub_date_elem = root.find(".//pub-date[@date-type='pub']")
                            year = pub_date_elem.find('year').text if pub_date_elem is not None and pub_date_elem.find('year') is not None else 'N/A'
                        except AttributeError:
                            year = 'N/A'

                        structured_data = {
                            "pmid": pmid,
                            "pmcid": pmcid,
                            "title": article_title,
                            "publication_year": year,
                            "abstract": article_abstract,
                            "full_text": article_full_text,
                            "content_for_embedding": content_to_chunk, # This will be chunked
                            "topic": topic_name # IMPORTANT: Tag the article with its topic
                        }
                        
                        jsonl_filename = os.path.join(topic_output_dir, f"{topic_name.lower()}_articles_data.jsonl")
                        with open(jsonl_filename, 'a', encoding='utf-8') as f:
                            json.dump(structured_data, f, ensure_ascii=False)
                            f.write('\n')
                        print(f"  SUCCESS: Processed PMC{pmcid} for {topic_name}. Saved data to JSONL.")

                        downloaded_count += 1
                        processed_pmids_set.add(pmid)
                        
                    except ET.ParseError as e:
                        print(f"  ERROR parsing XML for PMC{pmcid} for {topic_name}: {e}")
                    except Exception as e:
                        print(f"  An unexpected error occurred processing PMC{pmcid} for {topic_name}: {e}")
                else:
                    print(f"  Skipping PMID: {pmid} as XML data could not be downloaded for {topic_name}.")
                
                time.sleep(0.5)

        print(f"\nFinal count of full-text articles downloaded/processed for '{topic_name}': {downloaded_count}")

    print("\n--- All Core Topics Processed ---")