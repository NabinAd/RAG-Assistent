import streamlit as st 
from langchain_community.document_loaders import PyPDFDirectoryLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_ollama import OllamaEmbeddings, ChatOllama 
from langchain_chroma import Chroma 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.documents import Document  # Wichtig für den Upload-Konverter
from pypdf import PdfReader                     # Für das direkte Auslesen im Speicher
import os
import shutil

# ====================== Konfiguration ====================== 
st.set_page_config(page_title="Unternehmens-KI-Assistent", page_icon="🤖", layout="wide") 

st.title("🤖 Unternehmens-KI-Assistent") 
st.caption("RAG-basiertes Dokumenten-Chat für dein Praktikumsportfolio") 

# Sidebar 
with st.sidebar: 
    st.header("Einstellungen") 
    model_name = 'llama3.2' 
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3) 
    k_results = st.slider("Anzahl Dokumente (k)", 3, 8, 4) 
    st.divider() 
    if st.button("Vector Store zurücksetzen"): 
        # 1. Verbindung zu Chroma schließen / Referenz aufheben
        if "vectorstore" in globals():
            global vectorstore
            vectorstore = None
        
        # 2. Ordner löschen
        if os.path.exists("./chroma_db"): 
            try:
                shutil.rmtree("./chroma_db") 
                st.toast("Wissensbasis erfolgreich gelöscht!", icon="🗑️")
            except PermissionError:
                st.error("Datei ist noch blockiert. Bitte starte die App neu oder schließe geöffnete Prozesse.")
        
        # 3. Streamlit Resource-Cache leeren & App neu starten
        st.cache_resource.clear()
        st.rerun()

# ====================== Embeddings & LLM ====================== 
@st.cache_resource 
def get_embeddings(): 
    return OllamaEmbeddings(model="nomic-embed-text") 

@st.cache_resource 
def get_llm(): 
    return ChatOllama(model=model_name, temperature=temperature) 

embeddings = get_embeddings() 
llm = get_llm() 

# ====================== Vector Store ====================== 
def get_vectorstore(): 
    if os.path.exists("./chroma_db") and len(os.listdir("./chroma_db")) > 0: 
        return Chroma(persist_directory="./chroma_db", embedding_function=embeddings) 
    return None 

# Initialer Abruf des Vector Stores
vectorstore = get_vectorstore() 

# ====================== Dokumente laden ====================== 
def load_documents(): 
    if os.path.exists("docs") and len(os.listdir("docs")) > 0: 
        loader = PyPDFDirectoryLoader("docs/") 
        return loader.load() 
    return [] 

# ====================== RAG Chain ====================== 
def create_rag_chain(vs, k=4): 
    if not vs: 
        return None 
    retriever = vs.as_retriever(search_kwargs={"k": k}) 
    
    prompt = ChatPromptTemplate.from_template(""" 
    Du bist ein professioneller und hilfreicher Assistent für Wirtschaftsinformatik-Themen. 
    Antworte ausschließlich auf Basis der folgenden Kontext-Dokumente. 
    Wenn du die Antwort nicht aus dem Kontext ableiten kannst, sage ehrlich: 
    "Dazu habe ich keine Informationen in den Dokumenten." 
    
    Kontext: {context} 
    Frage: {question} 
    Antwort: 
    """) 
    
    chain = ( 
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser() 
    ) 
    return chain 

# ====================== Haupt-Interface ====================== 
tab1, tab2 = st.tabs(["💬 Chat", "📤 Dokumente hochladen"]) 

with tab2: 
    st.subheader("Neue Dokumente hochladen") 
    uploaded_files = st.file_uploader("PDFs hochladen", type="pdf", accept_multiple_files=True) 
    
    if uploaded_files: 
        if st.button("Dokumente verarbeiten und in Wissensbasis laden"): 
            with st.spinner("Dokumente werden verarbeitet..."): 
                docs = [] 
                
                # PDFs direkt aus dem Streamlit-Speicher parsen
                for uploaded_file in uploaded_files: 
                    pdf_reader = PdfReader(uploaded_file)
                    text = ""
                    for page in pdf_reader.pages:
                        extracted_text = page.extract_text()
                        if extracted_text:
                            text += extracted_text + "\n"
                    
                    # In LangChain-Dokumenten-Struktur umwandeln
                    if text.strip():
                        docs.append(Document(page_content=text, metadata={"source": uploaded_file.name}))
                
                if docs:
                    # Splitten 
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) 
                    chunks = splitter.split_documents(docs) 
                    
                    # Vector Store erstellen/ergänzen 
                    if vectorstore is None: 
                        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db") 
                    else: 
                        vectorstore.add_documents(chunks) 
                    
                    st.success(f"{len(chunks)} Chunks erfolgreich hinzugefügt!") 
                    st.rerun()  # App neu laden, damit tab1 sofort Zugriff auf den neuen Vectorstore hat
                else:
                    st.error("Es konnte kein Text aus den PDFs extrahiert werden.")

with tab1: 
    # Initialisiere Chat-History 
    if "messages" not in st.session_state: 
        st.session_state.messages = [] 

    # Zeige Chat-Verlauf 
    for message in st.session_state.messages: 
        with st.chat_message(message["role"]): 
            st.markdown(message["content"]) 

    # User Input 
    if prompt_input := st.chat_input("Stelle eine Frage zu deinen Dokumenten..."): 
        st.session_state.messages.append({"role": "user", "content": prompt_input}) 
        with st.chat_message("user"): 
            st.markdown(prompt_input) 
        
        with st.chat_message("assistant"): 
            with st.spinner("Denke nach..."): 
                chain = create_rag_chain(vectorstore, k=k_results) 
                
                if vectorstore is None or chain is None: 
                    response = "Noch keine Dokumente in der Datenbank. Lade zuerst PDFs hoch!" 
                    st.markdown(response) 
                else: 
                    response = chain.invoke(prompt_input) 
                    st.markdown(response) 
                    
                    # Quellen (optional) 
                    docs = vectorstore.similarity_search(prompt_input, k=3) 
                    with st.expander("Quellen anzeigen"): 
                        for i, doc in enumerate(docs): 
                            st.write(f"**Dokument {i+1}:** {doc.metadata.get('source', 'Unbekannt')}") 
            
            st.session_state.messages.append({"role": "assistant", "content": response}) 

    # Hinweis, falls die DB leer ist
    if vectorstore is None: 
        st.info("👉 Lade zuerst PDFs im Tab 'Dokumente hochladen' hoch, damit der Assistent antworten kann!")