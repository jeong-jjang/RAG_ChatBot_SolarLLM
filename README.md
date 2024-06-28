# RAG_ChatBot_SolarLLM
**SolarLLM 및 Solar Embedding 모델을 이용한 RAG ChatBot service 개발**

## RAG(Retrieval-Augmented Generation)
사용자가 쿼리를 보냈을 때, 질문에 대한 답을 지식 db에서 찾아서 LLM 모델을 사용하여 생성해내는 것

### Preprocessing
1. Indexing
- Load: 문맥 정보 및 데이터 업로딩
- Split: 로딩된 파일 쪼개서 검색/모델 인풋에 넣기 좋게 만듦
- Store: split한 데이터 DB에 임베딩하 업로드
2. Retrieval and Generation
- Retrieve: 사용자의 검색 쿼리에 따라 DB 검색
- Generate: LLM이 찾은 답변을 다시 문장으로 생성
