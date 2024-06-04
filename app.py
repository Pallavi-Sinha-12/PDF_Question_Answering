import streamlit as st
from ctransformers import AutoModelForCausalLM
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Load the model and set up necessary configurations
def hugging_face_model_load():
    config = {'max_new_tokens': 1000, 'context_length': 6000, 'repetition_penalty': 1.1,
              'temperature': 0.1, 'stream': True}

    llm = AutoModelForCausalLM.from_pretrained(
        "TheBloke/orca_mini_3B-GGML", model_file='orca-mini-3b.ggmlv3.q4_0.bin',
        model_type="llama",
        gpu_layers=86,  # 110 for 7b, 130 for 13b
        **config)

    return llm

# Function to process the query and generate the response
def process_query(llm, query, data, chat_history):
    embeddings = HuggingFaceEmbeddings()
    db = Chroma.from_documents(data, embeddings)

    docs = db.similarity_search(query)

    prompt = f"""You are a friendly chatbot assistant that responds in a conversational
    manner to user's questions. Consider the information available in the provided context as well as the chat history to answer the question.
    Review the chat history for relevant details and incorporate them into your response. 
    If the answer is not present within this context or the chat history, kindly state that you don't have the required information.
    Chat History: {chat_history}
    Context: {docs[0]}
    user_query: {query}
    bot_response : """

    tokens = llm.tokenize(prompt)
    result = ""
    for token in llm.generate(tokens):
        result += llm.detokenize(token)

    return result

def save_uploaded_file(uploaded_file):
    os.makedirs('uploads', exist_ok=True)
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

def main():
    st.title("📚 PDF-based Question Answering Assistant 🤖")
    st.markdown("---")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []


    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Select a PDF file", type="pdf")

    reset_button = st.button("Reset Session")

    if reset_button:
        uploaded_file = None
        st.session_state["chat_history"] = []
        st.success("Session has been reset. Upload a PDF file.")


    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        st.success("📤 **PDF Uploaded Successfully!**")

        # Process the uploaded PDF file
        loader = PyPDFLoader(file_path=file_path)
        data = loader.load()

        # Load the model
        llm = hugging_face_model_load()

        st.header("Ask a Question")
        query = st.text_input("🤔 Enter your question:")
        generate_button = st.button("🔍 Generate Answer")

        if generate_button:
            if query:
                with st.spinner('🔄 Processing...'):
                    result = process_query(llm, query, data, st.session_state["chat_history"])
                    st.session_state['chat_history'].append({"user_query" : query, "bot_response" : result})
                    st.success(f"📝 **Answer:** {result}")
                print(st.session_state)

        history_button = st.button("🕰️ Show Previous Queries and Answers")
        if history_button and st.session_state['chat_history']:
            st.header("Previous Queries and Answers")

            for idx, chat in enumerate(st.session_state["chat_history"]):
                st.write(f"**Query {idx + 1} :** {chat['user_query']}")
                st.write(f"**Answer :** {chat['bot_response']}")
                st.markdown("---")

if __name__ == "__main__":
    main()

