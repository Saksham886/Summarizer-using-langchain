# ==== Imports ====
import os
import sys
import asyncio
import datetime
import subprocess
from dotenv import load_dotenv
import streamlit as st
from fpdf import FPDF

# Langchain Imports
from langchain_community.document_loaders import (
    YoutubeLoader, PlaywrightURLLoader, PyPDFLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint
import pdfplumber
from langchain.schema import Document
# ==== Event Loop Fix for Windows ====
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# ==== Load Environment Variables ====
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# ==== Optional Playwright Install ====
if not os.path.exists("/home/appuser/.cache/ms-playwright"):
    try:
        subprocess.run(["playwright", "install"], check=True)
        print("✅ Playwright installed at runtime")
    except Exception as e:
        print(f"⚠️ Failed to install Playwright: {e}")

# ==== PDF Generator ====
def create_pdf(summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in summary_text.split('\n'):
        pdf.multi_cell(0, 10, line)
    return pdf.output(dest='S').encode('latin1')
def load_pdf_from_memory(file):
    """Reads PDF file from memory and returns LangChain Document list."""
    docs = []
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        docs.append(Document(page_content=text))
    return docs

# ==== Prompt Templates ====
basic_prompt_template = PromptTemplate(
    input_variables=["text","ln"],
    template="You are a helpful assistant. Provide a short 300-word summary of the text:\n\n{text}in {ln} language"
)

map_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
Please summarize the below speech:
Speech:`{text}`
Summary:
"""
)

combine_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
Provide the final summary of the entire speech with these important points.
Add a Motivation Title, start with an introduction, and present the summary as numbered points in {ln} language.
Speech:{text}

"""
)


# ==== Streamlit UI ====
st.title("📚 Langchain Summarizer")

with st.sidebar:
    st.header("🔧 Configuration")
    summarization_type = st.selectbox('What do you want to summarize?', ('URL', 'PDF'))
    model_choice = st.selectbox("Choose Model", ("Groq", "HuggingFace"))

# ==== Select LLM ====
def get_llm(model_choice):
    if model_choice == "Groq":
        return ChatGroq(model="gemma2-9b-it", api_key=GROQ_API_KEY)
    elif model_choice == "HuggingFace":
        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            temperature=1,
            token=HF_TOKEN
        )
    else:
        st.error("❌ Invalid model choice")
        return None

llm = get_llm(model_choice)


# ==== URL Summarizer ====
if summarization_type == 'URL':
    st.subheader("Summarize YouTube Video or Website")
    url = st.text_input("Enter your URL")
    ln=st.text_input("Enter the language")
    if st.button("Summarize"):
        with st.spinner("⏳ Summarizing..."):
            if not url:
                st.error("Please enter a valid URL.")
            else:
                try:
                    if 'youtube.com' in url or 'youtu.be' in url:
                        loader = YoutubeLoader.from_youtube_url(url)
                    else:
                        loader = PlaywrightURLLoader(
                            urls=[url], remove_selectors=["header", "footer"]
                        )
                    documents = loader.load()

                    # Summarization Chain
                    chain = load_summarize_chain(
                        llm=llm,
                        chain_type="stuff",
                        prompt=basic_prompt_template
                    )
                    response = chain.invoke({
                        "input_documents": documents,
                        "text": documents[0].page_content,
                        "ln": ln
                    })


                    # Show result
                    st.markdown("### 📄 Summary")
                    st.success(response["output_text"])
                    # PDF Download
                    pdf_data = create_pdf(response["output_text"])
                    filename_pdf = f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button(
                        label="📥 Download Summary as PDF",
                        data=pdf_data,
                        file_name=filename_pdf,
                        mime="application/pdf"
                    )

                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

# ==== PDF Summarizer ====
# ==== PDF Summarizer ====
elif summarization_type == 'PDF':
    st.subheader("Summarize PDF Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    ln = st.text_input("Enter the language")

    if st.button("Summarize"):
        if not uploaded_file:
            st.error("Please upload a PDF")
        elif not ln:
            st.error("Please write language")
        else:
            with st.spinner("⏳ Processing PDF..."):
                try:
                    docs = load_pdf_from_memory(uploaded_file)

                    final_documents = RecursiveCharacterTextSplitter(
                        chunk_size=2000,
                        chunk_overlap=100
                    ).split_documents(docs)

                    # Map-Reduce Summarization
                    summary_chain = load_summarize_chain(
                        llm=llm,
                        chain_type="map_reduce",
                        map_prompt=map_prompt_template,
                        combine_prompt=combine_prompt_template
                    )

                    response = summary_chain.invoke({
                        "input_documents": final_documents,
                        "text": final_documents[0].page_content,
                        "ln": ln
                    })

                    # Show result
                    st.markdown("### 📄 Summary")
                    st.success(response["output_text"])

                    # PDF Download
                    pdf_data = create_pdf(response["output_text"])
                    filename_pdf = f"summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button(
                        label="📥 Download Summary as PDF",
                        data=pdf_data,
                        file_name=filename_pdf,
                        mime="application/pdf"
                    )

                except Exception as e:
                    st.error(f"⚠️ Error: {e}")


