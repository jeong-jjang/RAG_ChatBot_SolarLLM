import os
from dotenv import load_dotenv
import uuid
import tempfile
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

# 환경 변수 로드
load_dotenv()

# 세션 ID 생성
session_id = uuid.uuid4()

# PDF 파일 처리 및 인덱싱 함수
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    vectorstore = Chroma.from_documents(pages, UpstageEmbeddings(model="solar-embedding-1-large"))
    return vectorstore

# RAG 체인 생성 함수
def create_rag_chain(vectorstore, api_key):
    retriever = vectorstore.as_retriever(k=2)
    from langchain_upstage import ChatUpstage
    chat = ChatUpstage(upstage_api_key=api_key)

    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    contextualize_q_system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. 
    이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 
    질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(chat, retriever, contextualize_q_prompt)

    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    qa_system_prompt = """질문-답변 업무를 돕는 보조원입니다. 
    질문에 답하기 위해 검색된 내용을 사용하세요. 
    답을 모르면 모른다고 말하세요. 
    답변은 세 문장 이내로 간결하게 유지하세요.

    ## 답변 예시
    📍답변 내용: 
    📍증거: 

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

# PDF 파일 처리 예제
uploaded_file_path = "path/to/your/pdf/file.pdf"

# 임시 파일 디렉토리에서 파일 처리
with tempfile.TemporaryDirectory() as temp_dir:
    file_path = os.path.join(temp_dir, os.path.basename(uploaded_file_path))
    with open(file_path, "wb") as f:
        f.write(open(uploaded_file_path, "rb").read())

    # PDF 파일 처리 및 인덱싱
    vectorstore = process_pdf(file_path)

# RAG 체인 생성
rag_chain = create_rag_chain(vectorstore, os.getenv("UPSTAGE_API_KEY"))

# 질문에 대한 답변 생성 예제
question = "What is the history of AI?"
chat_history = []

result = rag_chain.invoke({"input": question, "chat_history": chat_history})
answer = result["answer"]
context = result["context"]

print("Answer:", answer)
print("Context:", context)
출처: https://faiiry9.tistory.com/150 [데싸 되기:티스토리]
