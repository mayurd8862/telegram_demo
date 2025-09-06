import streamlit as st
from src.speech_to_text import transcribe_audio
from src.text_to_speech import text_to_speech
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq

from streamlit_chat import message
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import tempfile
import asyncio
import io

# Load environment variables
load_dotenv()

# Initialize Groq
llm = ChatGroq(model_name="Llama3-8b-8192")
client = Groq()

# Header
st.title("🎙️ Voice to voice Chatbot")

@st.cache_resource
def load_and_process_data(files):
    """Load and split documents into a FAISS vector DB."""
    all_splits = []
    for file in files:
        # Write uploaded file to temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.getbuffer())   # ✅ safe binary write
            file_path = temp_file.name

        # Loader (PDF or Markdown)
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredMarkdownLoader(file_path)

        data = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(data)
        all_splits.extend(splits)
    
    # Embeddings + VectorDB
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(all_splits, embedding)
    return vectordb

async def response_generator(vectordb, query):
    """Generate answer from retriever + LLM."""
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum. Keep the answer as concise as possible. 
    {context} 
    Question: {question} 
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"], 
        template=template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=vectordb.as_retriever(), 
        return_source_documents=True, 
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    result = await asyncio.to_thread(qa_chain, {"query": query})
    return result["result"]


with st.sidebar:
    files = st.file_uploader("Upload PDF/Markdown File(s)", type=["pdf","md"], accept_multiple_files=False)
    submit_pdf = st.checkbox('Submit and chat (PDF/MD)')

st.subheader('', divider="rainbow")

if files and submit_pdf:
    vectordb = load_and_process_data(files)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "audio" in message:
                st.audio(io.BytesIO(message["audio"]), format="audio/mp3")  # ✅ wrap raw bytes

    # Audio input
    audio_value = st.audio_input("Record a voice message")

    if audio_value:
        with st.spinner("Processing voice message..."):
            # Save user audio to file
            webm_file_path = "temp_audio.mp3"
            with open(webm_file_path, "wb") as f:
                f.write(audio_value.read())
            
            # Transcribe audio
            transcript = transcribe_audio(client, webm_file_path)
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(transcript)
            st.session_state.messages.append({"role": "user", "content": transcript})
            
            # Generate assistant response
            with st.spinner("Thinking..."):
                output = asyncio.run(response_generator(vectordb, transcript))
                
            # Display assistant response + TTS
            with st.chat_message("assistant"):
                st.markdown(output)
                audio_bytes = text_to_speech(output)  # ✅ raw bytes
                st.audio(io.BytesIO(audio_bytes), format="audio/mp3", autoplay=True)
            
            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": output,
                "audio": audio_bytes
            })








# import streamlit as st
# from src.speech_to_text import transcribe_audio
# from dotenv import load_dotenv
# from groq import Groq
# from langchain_groq import ChatGroq

# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
# from langchain_community.vectorstores import FAISS
# from langchain.prompts import PromptTemplate
# from langchain.chains import RetrievalQA
# import tempfile
# import asyncio
# from gtts import gTTS
# import io

# # Load environment variables
# load_dotenv()

# # Initialize Groq
# llm = ChatGroq(model_name="Llama3-8b-8192")
# client = Groq()

# # Header
# st.title("🎙️ Voice to Voice Chatbot")

# @st.cache_resource
# def load_and_process_data(files):
#     """Load and split documents into a FAISS vector DB."""
#     all_splits = []
#     for file in files:
#         # Write uploaded file to temp path
#         with tempfile.NamedTemporaryFile(delete=False, suffix=file.name.split(".")[-1]) as temp_file:
#             temp_file.write(file.getbuffer())
#             file_path = temp_file.name

#         # Loader
#         if file.name.endswith(".pdf"):
#             loader = PyPDFLoader(file_path)
#         else:
#             loader = UnstructuredMarkdownLoader(file_path)

#         data = loader.load()
        
#         # Split into chunks
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=100
#         )
#         splits = text_splitter.split_documents(data)
#         all_splits.extend(splits)
    
#     # Embeddings + VectorDB
#     embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#     vectordb = FAISS.from_documents(all_splits, embedding)
#     return vectordb

# def response_generator(vectordb, query):
#     """Generate answer from retriever + LLM."""
#     template = """Use the following pieces of context to answer the question at the end. 
#     If you don't know the answer, just say you don't know. 
#     Use three sentences maximum. Keep the answer concise.
#     {context} 
#     Question: {question} 
#     Helpful Answer:"""

#     QA_CHAIN_PROMPT = PromptTemplate(
#         input_variables=["context", "question"], 
#         template=template
#     )
#     qa_chain = RetrievalQA.from_chain_type(
#         llm, 
#         retriever=vectordb.as_retriever(), 
#         return_source_documents=True, 
#         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
#     )
    
#     result = qa_chain.invoke({"query": query})
#     return result["result"]

# def text_to_speech_file_gtts(text: str):
#     """Convert text to speech using gTTS and return raw audio bytes."""
#     tts = gTTS(text=text, lang="en")
#     audio_buffer = io.BytesIO()
#     tts.write_to_fp(audio_buffer)
#     audio_buffer.seek(0)
#     return audio_buffer.read()

# # Sidebar uploader
# with st.sidebar:
#     files = st.file_uploader("Upload PDF/Markdown File(s)", type=["pdf", "md"], accept_multiple_files=True)
#     submit_pdf = st.button("Submit and Chat (PDF/MD)")

# if files and submit_pdf:
#     vectordb = load_and_process_data(files)
    
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#             if message["role"] == "assistant" and "audio" in message:
#                 st.audio(io.BytesIO(message["audio"]), format="audio/mp3")

#     # Audio input
#     audio_value = st.audio_input("🎤 Record a voice message")

#     if audio_value:
#         with st.spinner("Processing voice message..."):
#             # Save audio input
#             webm_file_path = "temp_audio.webm"
#             with open(webm_file_path, "wb") as f:
#                 f.write(audio_value.read())
            
#             # Transcribe audio
#             transcript = transcribe_audio(client, webm_file_path)
            
#             # Display user message
#             with st.chat_message("user"):
#                 st.markdown(transcript)
#             st.session_state.messages.append({"role": "user", "content": transcript})
            
#             # Generate assistant response
#             with st.spinner("Thinking..."):
#                 output = response_generator(vectordb, transcript)
                
#             # Display assistant response + TTS
#             with st.chat_message("assistant"):
#                 st.markdown(output)
#                 audio_bytes = text_to_speech_file_gtts(output)
#                 st.audio(io.BytesIO(audio_bytes), format="audio/mp3")
            
#             # Save to chat history
#             st.session_state.messages.append({
#                 "role": "assistant",
#                 "content": output,
#                 "audio": audio_bytes
#             })
