import os
import bs4
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
os.environ["GROQ_API_KEY"] = "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"



# Initialize the LLM
def initialize_llm():
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable not set")

    return init_chat_model("llama3-8b-8192", model_provider="groq")


# Initialize embeddings
def initialize_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# Initialize vector store
def initialize_vector_store(embeddings):
    return InMemoryVectorStore(embeddings)


# Load documents from URL
def load_from_url(url):
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    return loader.load()


# Load documents from PDF
def load_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    return loader.load()


# Split documents
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return text_splitter.split_documents(docs)


# Define state for RAG application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Retrieve step
def retrieve(state: State, vector_store):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


# Generate step
def generate(state: State, llm, prompt):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Create and compile the graph
def create_rag_pipeline(vector_store, llm):
    prompt = hub.pull("rlm/rag-prompt")

    # Create wrapper functions that include the dependencies
    def retrieve_with_deps(state: State):
        return retrieve(state, vector_store)

    def generate_with_deps(state: State):
        return generate(state, llm, prompt)

    # Build the graph
    graph_builder = StateGraph(State).add_sequence([retrieve_with_deps, generate_with_deps])
    graph_builder.add_edge(START, "retrieve_with_deps")
    return graph_builder.compile()
