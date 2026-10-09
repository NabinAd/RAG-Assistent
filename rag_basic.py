from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser 

#Dokument laden
loader = PyPDFDirectoryLoader ("docs/")
docs = loader.load()
print(f"Es wurden {len(docs)} Dokumente geladen.")

# Spilitten
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 200 )
chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model='nomic-embed-text')
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory = './chroma_db')

#retriever + LLM 
retriever = vectorstore.as_retriever(search_kwargs={'k':4})
LLM = ChatOllama(model='llama3.2' , temperature=0.3)

#Promt 
prompt =  ChatPromptTemplate.from_template('''Du bist ein hilfreicher Assistent für Unternehmensfragen.
Antworte nur mit gegebenen kontexten. Wenn du es nicht weißt, sag "Ich weiß es nicht".
Kontext: {context}
Frage: {question}
Antwort:
                                            ''')

chain = ({'context': retriever, 'question': RunnablePassthrough()}  | prompt | LLM | StrOutputParser() )

print(chain.invoke('was ist machine learning und wie hilft es Autonom fahren'))