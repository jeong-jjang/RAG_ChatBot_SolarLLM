"""
아직 수정 중
"""
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import time
import base64
import uuid
import tempfile
from PyPDF2 import PdfReader, PdfWriter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain

# 세션 상태 초기화 및 PDF 파일 업로드
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None

def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="600" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

with st.sidebar:
    st.header(f"Upload your documents!")

    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            file_key = f"{session_id}-{uploaded_file.name}"

            # PDF 파일 프리뷰 표시
            display_pdf(uploaded_file)

            # 페이지 범위 선택
            st.subheader("Select pages to use:")
            start_page = st.number_input("Start page", min_value=1, max_value=PdfReader(uploaded_file).numPages, value=1)
            end_page = st.number_input("End page", min_value=start_page, max_value=PdfReader(uploaded_file).numPages, value=PdfReader(uploaded_file).numPages)

            # 임시 디렉토리에 파일 저장
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # PDF 일부분만 추출하여 새로운 PDF 파일 생성
                output_pdf_path = os.path.join(temp_dir, "selected_pages.pdf")
                reader = PdfReader(file_path)
                writer = PdfWriter()

                for page_num in range(start_page - 1, end_page):
                    writer.add_page(reader.pages[page_num])

                with open(output_pdf_path, "wb") as output_pdf_file:
                    writer.write(output_pdf_file)

                # 문서 로드 및 임베딩 생성
                loader = PyPDFLoader(output_pdf_path)
                pages = loader.load_and_split()
                vectorstore = Chroma.from_documents(pages, UpstageEmbeddings(model="solar-embedding-1-large"))
                retriever = vectorstore.as_retriever(k=2)

                chat = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"))

                contextualize_q_system_prompt = """Contextualization prompt..."""
                contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )

                history_aware_retriever = create_history_aware_retriever(
                    chat, retriever, contextualize_q_prompt
                )

                qa_system_prompt = """QA prompt..."""
                qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", qa_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )

                question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                st.success("Ready to Chat!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

st.title("Solar LLM Chatbot")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

MAX_MESSAGES_BEFORE_DELETION = 4

if prompt := st.chat_input("Ask a question!"):
    if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
        del st.session_state.messages[0]
        del st.session_state.messages[0]

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        result = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.messages})

        with st.expander("Evidence context"):
            st.write(result["context"])

        for chunk in result["answer"].split(" "):
            full_response += chunk + " "
            time.sleep(0.2)
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
