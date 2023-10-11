import os
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI


# Loading pdf, docx, and txt files 
def load_documents(file):
    name, ext = os.path.splitext(file)

    if ext == ".pdf":
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif ext == ".docx":
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif ext == ".txt":
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        st.write(f'Document format is not supported!')
        return None

    data = loader.load()
    return data

# Chunk data
def chunk_data(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    return chunks

# Create embeddings and vector store
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                        model_kwargs={'device': 'cpu'})
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# Create the conversation chain
def create_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain

# Create qna function
def conversational_chat(query, chain):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def main():
    #load_dotenv()
    st.set_page_config(page_title="LLM Question-Answering Application", page_icon=":books:")
    st.subheader("LLM Question-Answering Application 🤖")

    with st.expander("About the App"):
        st.markdown(
            """
            This is a Generative AI powered Question and Answering app that responds to questions about your uploaded file.
            """
        )

    api_key = os.getenv('OPENAI_API_KEY')

    # Ask the user to enter the API key if not found
    if not api_key:
        api_key = st.sidebar.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        else:
            st.sidebar.warning("Please enter your OpenAI API Key.")
            return

    uploaded_file = st.sidebar.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
    if uploaded_file:

        with st.spinner('Processing'):
            bytes_data = uploaded_file.read()
            file_name = os.path.join("./", uploaded_file.name)
            with open(file_name, 'wb') as f:
                f.write(bytes_data)
                data = load_documents(file_name)
            if data:
                chunks = chunk_data(data)
                vector_store = create_vector_store(chunks)
                st.success('File uploaded, chunked and embedded successfully')
                #st.session_state.conversation = create_conversation_chain(vector_store)
                chain = create_conversation_chain(vector_store)

                # def conversational_chat(query):
                #     result = chain({"question": query, "chat_history": st.session_state['history']})
                #     st.session_state['history'].append((query, result["answer"]))
                #     return result["answer"]
                
                if 'history' not in st.session_state:
                    st.session_state['history'] = []

                if 'generated' not in st.session_state:
                    st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

                if 'past' not in st.session_state:
                    st.session_state['past'] = ["Hey ! 👋"]
                    
                # Container for the chat history
                response_container = st.container()
                # cCntainer for the user's text input
                container = st.container()

                with container:
                    with st.form(key='my_form', clear_on_submit=True):
                        
                        user_input = st.text_input("Query:", placeholder="Ask about me about your document (:", key='input')
                        submit_button = st.form_submit_button(label='Send')
                        
                    if submit_button and user_input:
                        output = conversational_chat(user_input, chain)
                        
                        st.session_state['past'].append(user_input)
                        st.session_state['generated'].append(output)

                if st.session_state['generated']:
                    with response_container:
                        for i in range(len(st.session_state['generated'])):
                            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                            message(st.session_state["generated"][i], key=str(i))



        


if __name__ == "__main__":
    main()



