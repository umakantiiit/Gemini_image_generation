import streamlit as st
import json
import base64
import os
import tempfile
import google.generativeai as genai
import vertexai
from pathlib import Path
from datetime import datetime
import time


class HindiAudioAnalysisPipeline:
    """
    A complete pipeline to transcribe Hindi audio files and analyze the transcription.
    """
    
    def __init__(self, credentials_path: str, project_id: str = None, location: str = "us-central1"):
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        if project_id:
            self.project_id = project_id
        else:
            self.project_id = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))
        
        self.location = location
        vertexai.init(project=self.project_id, location=self.location)
        self.model = genai.GenerativeModel("gemini-2.5-pro")
        
        self.generation_config = genai.GenerationConfig(
            temperature=0.1
        )
        
        self.transcription_prompt = '''
This is a Hindi language conversation happens between a caller from the govt organisation and a tribal people . U need to pay close attention to the conversation and generate the transcript of it . Also make sure to do the speaker diarization. Donot pay much attention to the background noise and try not to include it in the transcript. Output it in the below mentioned json format .
If a speaker cannot be identified, label them as "Unknown."
Context about the caller and conversation - The conversation is all about the a speaker  asking  citizens about to whom they want to see as their next cheif minister and to whom they want to see as their next local leader. The speaker will give them multiple options of leader name as well as their party and the citizen have to choose between them. It is just for information purpose and not for actual casting of vote. The speaker may ask the vidhansabha constituency details and the mobile number and the religion of the citizen .The speaker may ask question about wheather they are satisfied by the work of the current leader or chief minister etc. and take opinion of the people.
OUTPUT THE EXACT NUMBER OF SPEAKER.DONOT BIASED BY THE CONTEXT. ANALYSE THE AUDIO PROPERLY .
Provide the output in the following JSON structure:
{
  "Call Details": {
    "Number of Speakers": "<total_number_of_speakers>",
    "Transcript": [
      {
        "Speaker": "<Speaker>",
        "Timestamp": {
          "Start": "<start_time>",
          "End": "<end_time>"
        },
        "Voice": "<extracted_text_from_audio>"
      },
      {
        "Speaker": "<Speaker>",
        "Timestamp": {
          "Start": "<start_time>",
          "End": "<end_time>"
        },
        "Voice": "<extracted_text_from_audio>"
      },
      ...
    ]
  }
}
'''
    self.analysis_prompt = '''
You are an expert analyst tasked with extracting specific information from Hindi language audio call transcripts. Your job is to carefully analyze the provided transcript and extract answers to predefined survey questions.

## CRITICAL INSTRUCTIONS:

1. **READ THE ENTIRE TRANSCRIPT CAREFULLY** before answering any question
2. **EXTRACT ANSWERS ONLY FROM THE TRANSCRIPT** - Do not make assumptions or infer answers
3. **MATCH ANSWERS TO PROVIDED OPTIONS** - You must select from the given options only
4. **LANGUAGE HANDLING**: The transcript is in Hindi. Understand the context and meaning in Hindi to match with the provided options
5. **HANDLE MISSING INFORMATION**: If an answer to a question is not found in the transcript, output "Not Available"
6. **BE PRECISE**: Match the respondent's answer to the closest provided option based on the meaning and context
7. **OUTPUT FORMAT**: Return responses in valid JSON format only

## SURVEY QUESTIONS WITH OPTIONS:

### Section 1 – Personal Questions

**Question 1:** क्या आपका वोट इसी विधानसभा में बना हुआ है ?
**Options:**
- हाँ
- नहीं

**Question 2:** क्या मैं आपकी उम्र जान सकता हूँ ?
**Options:**
- 18-30 वर्ष
- 31-45 वर्ष
- 46-60 वर्ष
- 60 वर्ष से अधिक
- बताना नहीं चाहते

**Question 3:** लिंग: (पूछना नहीं है देखकर भरें)
**Options:**
- पुरुष
- महिला

**Question 4:** आप क्या काम करते हैं ?
**Options:**
- कृषि
- मजदूरी/दैनिक वेतन
- सरकारी नौकरी
- निजी नौकरी
- स्वरोजगार/व्यवसाय
- बेरोजगार
- हाउसवाइफ / गृहिणी
- छात्र / पढाई
- अन्य (कृपया उल्लेख करें)

**Question 5:** आप कौन से धर्म से हैं?
**Options:**
- हिन्दू
- मुस्लिम
- सिख
- क्रिस्चियन
- जैन
- बुद्धिस्ट
- किसी धर्म को नहीं मानता
- बताना नहीं चाहते

**Question 6:** आपकी जाति क्या है ?
**Options:**
- (विधानसभा के टॉप 12 जातियों की लिस्ट के अनुसार)

---

### Section 2 – Political Questions

**Question 7:** आपके एमएलए/ विधायक का क्या नाम है?
**Options:**
- MLA का नाम सही बताया
- MLA का नाम नहीं पता
- MLA का नाम गलत बताया

**Question 8:** क्या आप अपने विधायक के काम से खुश हैं ?
**Options:**
- हाँ
- नहीं
- कुछ कह नहीं सकते

**Question 9:** अगर आपका विधायक फिर से चुनाव लड़ता है तो क्या आप उन्हें वोट देंगे ?
**Options:**
- हाँ
- नहीं
- कुछ कह नहीं सकते

**Question 9.1:** अगर नहीं तो, आप अगले विधायक के रूप में किसे देखना चाहते हैं ?
**Options:**
- (संभावित उम्मीदवारों की सुची के अनुसार)

**Question 10:** आप आने वाले विधानसभा चुनाव में किस पार्टी को वोट देंगे?
**Options:**
- JDU | NDA
- महागठबंधन | MGB
- जन सुराज पार्टी
- अन्य
- बीएसपि | BSP
- NOTA
- कह नहीं सकते

**Question 11:** 2020 के पिछले विधानसभा चुनाव में आपने किस पार्टी को वोट दिया था?
**Options:**
- JDU | NDA
- महागठबंधन | MGB
- लोक जनशक्ति पार्टी (लोजपा)
- अन्य
- बीएसपि | BSP
- NOTA
- कह नहीं सकते / याद नहीं
- वोट नहीं दिया

**Question 12:** क्या आप बिहार सरकार द्वारा 2020 – 25 में किए गए काम से शंतुष्ट हैं ?
**Options:**
- बिलकुल शंतुष्ट हैं
- बिलकुल शंतुष्ट नहीं
- कुछ हद तक शंतुष्ट हैं
- कुछ कह नहीं सकते

**Question 13:** आप अगले मुख्यमंत्री के रूप में किसे देखना चाहते हैं ?
**Options:**
- नितीश कुमार
- तेजश्वी यादव
- सम्राट चौधरी
- चिराग पासवान
- प्रशांत किशोर
- अन्य

---

### Section 3 – Observational Questions

**Question 14:** आप गाँव में रहते हैं या शहर में?
**Options:**
- गाँव
- शहर

**Question 15:** ब्लाक का नाम (आप जहाँ है उस ब्लाक का नाम चुनें)
**Note:** Extract the exact block name mentioned in the transcript

**Question 16:** गाँव / पंचायत का नाम (आप जिस गाँव/ पंचायत में है उस पंचायत का नाम चुनें)
**Note:** Extract the exact village/panchayat name mentioned in the transcript

**Question 17:** सर क्या मैं आपका मोबाइल नंबर जान सकता हूँ ?
**Note:** Extract the mobile number if provided in the transcript

---

## ANALYSIS GUIDELINES:

1. **For Questions 1-14**: Select EXACTLY ONE option from the provided list that best matches the respondent's answer
2. **For Question 15-17**: Extract the exact information (block name, village/panchayat name, mobile number) as mentioned in the transcript
3. **Contextual Understanding**: Understand variations in Hindi responses. For example:
   - "हाँ", "जी हाँ", "बिल्कुल", "ठीक है" all mean "Yes"
   - "नहीं", "जी नहीं", "बिल्कुल नहीं" all mean "No"
   - Age ranges: Listen for specific numbers and categorize appropriately
4. **Partial Answers**: If the respondent gives an incomplete or unclear answer, choose "कुछ कह नहीं सकते" or equivalent option if available
5. **Question 9.1**: Only answer this if the respondent answered "नहीं" to Question 9

## OUTPUT FORMAT:

Return your analysis in the following JSON structure:

{
  "section_1": {
    "question_1": "selected option or Not Available",
    "question_2": "selected option or Not Available",
    "question_3": "selected option or Not Available",
    "question_4": "selected option or Not Available",
    "question_5": "selected option or Not Available",
    "question_6": "answer or Not Available"
  },
  "section_2": {
    "question_7": "selected option or Not Available",
    "question_8": "selected option or Not Available",
    "question_9": "selected option or Not Available",
    "question_9_1": "answer or Not Available",
    "question_10": "selected option or Not Available",
    "question_11": "selected option or Not Available",
    "question_12": "selected option or Not Available",
    "question_13": "selected option or Not Available"
  },
  "section_3": {
    "question_14": "selected option or Not Available",
    "question_15": "block name or Not Available",
    "question_16": "village/panchayat name or Not Available",
    "question_17": "mobile number or Not Available"
  }
}

## IMPORTANT REMINDERS:

- **DO NOT TRANSLATE** the options - keep them in Hindi as provided
- **DO NOT INVENT** answers - only extract from the transcript. U can do a bit assumption by taking word from the transcript but DONOT GUESS COMPLETELY.
- **ENSURE VALID JSON** - proper formatting with quotes and commas
- **ONE ANSWER PER QUESTION** - select only one option from the provided list
- **EXACT OPTION MATCH** - use the exact text of the option as provided above

Now, please analyze the given Hindi transcript and provide the extracted information in the specified JSON format:
'''
    
    def _load_audio_to_base64(self, file_path: str) -> str:
        with open(file_path, "rb") as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode("utf-8")
        return audio_data
    
    def _load_json_to_base64(self, file_path: str) -> str:
        with open(file_path, "rb") as json_file:
            json_data = base64.standard_b64encode(json_file.read()).decode("utf-8")
        return json_data
    
    def _clean_json_output(self, content: str) -> str:
        lines = content.splitlines(keepends=True)
        if len(lines) >= 2:
            lines = lines[1:-1]
        return ''.join(lines)
    
    def transcribe_audio(self, audio_file_path: str, audio_mime_type: str = "audio/m4a") -> str:
        audio_base64 = self._load_audio_to_base64(audio_file_path)
        
        contents_transcription = [
            {'mime_type': audio_mime_type, 'data': audio_base64},
            self.transcription_prompt
        ]
        
        transcription_response = self.model.generate_content(
            contents_transcription,
            generation_config=self.generation_config
        )
        
        return self._clean_json_output(transcription_response.text)
    
    def analyze_transcript(self, transcript_path: str) -> str:
        transcript_base64 = self._load_json_to_base64(transcript_path)
        
        contents_analysis = [
            {'mime_type': 'text/plain', 'data': transcript_base64},
            self.analysis_prompt
        ]
        
        analysis_response = self.model.generate_content(
            contents_analysis,
            generation_config=self.generation_config
        )
        
        return self._clean_json_output(analysis_response.text)


# Streamlit Page Configuration
st.set_page_config(
    page_title="Convolyser-Indic Demo",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(120deg, #2d3748 0%, #1a202c 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
    }
    
    .company-name {
        font-size: 2.5rem;
        font-weight: 800;
        color: #fbbf24;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
    }
    
    .product-name {
        font-size: 1.5rem;
        font-weight: 600;
        color: #60a5fa;
        margin-bottom: 1rem;
    }
    
    .welcome-text {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin: 2rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Card styling */
    .info-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        margin: 1.5rem 0;
    }
    
    /* Success message */
    .success-box {
        background: linear-gradient(120deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    /* JSON display */
    .json-container {
        background: #1e293b;
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        max-height: 500px;
        overflow-y: auto;
        margin: 1rem 0;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(120deg, #2d3748 0%, #1a202c 100%);
        color: #fbbf24;
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(120deg, #8b5cf6 0%, #6d28d9 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
    }
    
    /* File uploader styling */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Progress animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .processing {
        animation: pulse 2s ease-in-out infinite;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="main-header">
        <div class="company-name">🚀 Ybrantworks</div>
        <div class="product-name">📊 Convolyser-Indic</div>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="welcome-text">
        🎙️ Welcome to the Demo of Convolyser-Indic 🎙️
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'transcription_done' not in st.session_state:
    st.session_state.transcription_done = False
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'transcript_json' not in st.session_state:
    st.session_state.transcript_json = None
if 'analysis_json' not in st.session_state:
    st.session_state.analysis_json = None
if 'transcript_path' not in st.session_state:
    st.session_state.transcript_path = None

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Vertex AI Credentials")
    credentials_file = st.file_uploader(
        "Upload Vertex AI JSON",
        type=['json'],
        help="Upload your Google Cloud service account JSON file",
        key="credentials"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.markdown("### 🎵 Audio File")
    audio_file = st.file_uploader(
        "Please Upload Audio File",
        type=['mp3', 'm4a', 'wav', 'ogg'],
        help="Upload your Hindi audio file for transcription",
        key="audio"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Show file upload status
if credentials_file and audio_file:
    st.markdown("""
        <div class="success-box">
            ✅ Files uploaded successfully! Ready to process.
        </div>
    """, unsafe_allow_html=True)

# Transcription Section
st.markdown("---")
st.markdown('<div class="info-card">', unsafe_allow_html=True)
st.markdown("### 📝 Step 1: Transcription")

if st.button("🎯 View Transcript", disabled=not (credentials_file and audio_file), use_container_width=True):
    with st.spinner("🔄 Processing audio file... This may take a few moments..."):
        try:
            # Save uploaded files temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_cred:
                tmp_cred.write(credentials_file.getvalue())
                credentials_path = tmp_cred.name
            
            # Determine audio file extension and mime type
            audio_extension = Path(audio_file.name).suffix.lower()
            mime_type_map = {
                '.mp3': 'audio/mpeg',
                '.m4a': 'audio/m4a',
                '.wav': 'audio/wav',
                '.ogg': 'audio/ogg'
            }
            audio_mime_type = mime_type_map.get(audio_extension, 'audio/m4a')
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=audio_extension) as tmp_audio:
                tmp_audio.write(audio_file.getvalue())
                audio_path = tmp_audio.name
            
            # Initialize pipeline
            pipeline = HindiAudioAnalysisPipeline(
                credentials_path=credentials_path,
                project_id=None  # Will use from credentials
            )
            
            # Transcribe
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            transcript_text = pipeline.transcribe_audio(audio_path, audio_mime_type)
            
            # Save transcript
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_transcript:
                tmp_transcript.write(transcript_text)
                st.session_state.transcript_path = tmp_transcript.name
            
            st.session_state.transcript_json = transcript_text
            st.session_state.transcription_done = True
            
            st.success("✅ Transcription completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during transcription: {str(e)}")

# Display transcript
if st.session_state.transcription_done and st.session_state.transcript_json:
    st.markdown("#### 📄 Transcription Result")
    
    # Format JSON for display
    try:
        json_obj = json.loads(st.session_state.transcript_json)
        formatted_json = json.dumps(json_obj, indent=2, ensure_ascii=False)
    except:
        formatted_json = st.session_state.transcript_json
    
    st.code(formatted_json, language='json')
    
    # Download button for transcript
    st.download_button(
        label="⬇️ Download Transcript JSON",
        data=st.session_state.transcript_json,
        file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Analysis Section
st.markdown("---")
st.markdown('<div class="info-card">', unsafe_allow_html=True)
st.markdown("### 🔍 Step 2: Analysis")

if st.button("📊 View Analysis of Audio", disabled=not st.session_state.transcription_done, use_container_width=True):
    with st.spinner("🧠 Analyzing transcript... Please wait..."):
        try:
            # Save uploaded credentials again
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_cred:
                tmp_cred.write(credentials_file.getvalue())
                credentials_path = tmp_cred.name
            
            # Initialize pipeline
            pipeline = HindiAudioAnalysisPipeline(
                credentials_path=credentials_path,
                project_id=None
            )
            
            # Analyze
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            analysis_text = pipeline.analyze_transcript(st.session_state.transcript_path)
            
            st.session_state.analysis_json = analysis_text
            st.session_state.analysis_done = True
            
            st.success("✅ Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")

# Display analysis
if st.session_state.analysis_done and st.session_state.analysis_json:
    st.markdown("#### 📊 Analysis Result")
    
    # Format JSON for display
    try:
        json_obj = json.loads(st.session_state.analysis_json)
        formatted_json = json.dumps(json_obj, indent=2, ensure_ascii=False)
    except:
        formatted_json = st.session_state.analysis_json
    
    st.code(formatted_json, language='json')
    
    # Download button for analysis
    st.download_button(
        label="⬇️ Download Analysis JSON",
        data=st.session_state.analysis_json,
        file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        🙏 Thank You For Using Convolyser-Indic 🙏
    </div>
""", unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.info("""
    **Convolyser-Indic** is an advanced audio analysis tool that:
    
    ✨ Transcribes Hindi audio files
    
    📊 Analyzes conversation content
    
    🎯 Extracts key insights
    
    💼 Built by Ybrantworks
    """)
    
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload your Vertex AI credentials (JSON)
    2. Upload your Hindi audio file
    3. Click 'View Transcript' to transcribe
    4. Click 'View Analysis' to analyze
    5. Download results as JSON files
    """)
    
    st.markdown("### 🎨 Features")
    st.success("✅ Real-time processing")
    st.success("✅ Beautiful JSON display")
    st.success("✅ Easy downloads")
    st.success("✅ Professional UI")
