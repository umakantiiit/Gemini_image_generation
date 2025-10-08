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
        
        self.transcription_prompt = """
        Please transcribe this Hindi audio file accurately. 
        Provide the transcription in JSON format with proper structure.
        Include timestamps if possible and maintain speaker identification.
        """
        
        self.analysis_prompt = """
        Analyze the provided Hindi conversation transcript and extract the following information in JSON format:
        
        1. Call Summary: Brief overview of the conversation
        2. Key Topics: Main topics discussed
        3. Sentiment: Overall sentiment of the conversation
        4. Action Items: Any tasks or follow-ups mentioned
        5. Important Dates/Times: Any dates or times mentioned
        6. Customer Issues: Problems or concerns raised
        7. Resolution Status: Whether issues were resolved
        
        Provide the analysis in structured JSON format.
        """
    
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
