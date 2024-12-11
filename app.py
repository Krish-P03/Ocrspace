import gradio as gr
import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["GROQ_API_KEY"] = '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'
# config = {'max_new_tokens': 512, 'context_length': 8000}
llm = ChatGroq(
    model='llama3-70b-8192',
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


# Define OCR functions for image and PDF files
def ocr_image(image_path, language='eng+guj'):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=language)
    return text

def ocr_pdf(pdf_path, language='eng+guj'):
    images = convert_from_path(pdf_path)
    all_text = ""
    for img in images:
        text = pytesseract.image_to_string(img, lang=language)
        all_text += text + "\n"
    return all_text

def ocr_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".pdf":
        text_re = ocr_pdf(file_path, language='guj+eng')
    elif file_extension in [".jpg", ".jpeg", ".png", ".bmp"]:
        text_re = ocr_image(file_path, language='guj+eng')
    else:
        raise ValueError("Unsupported file format. Supported formats are PDF, JPG, JPEG, PNG, BMP.")
    return text_re

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    os.makedirs("faiss_index", exist_ok=True)
    vector_store.save_local("faiss_index")
    return vector_store

def process_ocr_and_pdf_files(file_paths):
    raw_text = ""
    for file_path in file_paths:
        raw_text += ocr_file(file_path) + "\n"
    text_chunks = get_text_chunks(raw_text)
    return get_vector_store(text_chunks)

def get_conversational_chain():
    template = """You are an intelligent educational assistant specialized in handling queries about documents. You have been provided with OCR-processed text from the uploaded files that contains important educational information.

Core Responsibilities:
1. Language Processing:
   - Identify the language of the user's query (English or Gujarati)
   - Respond in the same language as the query
   - If the query is in Gujarati, ensure the response maintains proper Gujarati grammar and terminology
   - For technical terms, provide both English and Gujarati versions when relevant

2. Document Understanding:
   - Analyze the OCR-processed text from the uploaded files
   - Account for potential OCR errors or misinterpretations
   - Focus on extracting accurate information despite possible OCR imperfections

3. Response Guidelines:
   - Provide direct, clear answers based solely on the document content
   - If information is unclear due to OCR quality, mention this limitation
   - For numerical data (dates, percentages, marks), double-check accuracy before responding
   - If information is not found in the documents, clearly state: \"This information is not present in the uploaded documents\"

4. Educational Context:
   - Maintain focus on educational queries related to the document content
   - For admission-related queries, emphasize important deadlines and requirements
   - For scholarship information, highlight eligibility criteria and application processes
   - For course-related queries, provide detailed, accurate information from the documents

5. Response Format:
   - Structure responses clearly with relevant subpoints when necessary
   - For complex information, break down the answer into digestible parts
   - Include relevant reference points from the documents when applicable
   - Format numerical data and dates clearly

6. Quality Control:
   - Verify that responses align with the document content
   - Don't make assumptions beyond the provided information
   - If multiple interpretations are possible due to OCR quality, mention all possibilities
   - Maintain consistency in terminology throughout the conversation

Important Rules:
- Never make up information not present in the documents
- Don't combine information from previous conversations or external knowledge
- Always indicate if certain parts of the documents are unclear due to OCR quality
- Maintain professional tone while being accessible to students and parents
- If the query is out of scope of the uploaded documents, politely redirect to relevant official sources

Context from uploaded documents:
{context}

Chat History:
{history}

Current Question: {question}
Assistant: Let me provide a clear and accurate response based on the uploaded documents..."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    new_vector_store = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=new_vector_store.as_retriever(), chain_type='stuff', verbose=True, chain_type_kwargs={"verbose": True,"prompt": QA_CHAIN_PROMPT,"memory": ConversationBufferMemory(memory_key="history",input_key="question"),})
    return qa_chain

def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "query": user_question}, return_only_outputs=True)
    return response.get("result", "No result found")

def gradio_interface():

    def process_text_files(files):
        file_contents = []
        temp_dir = "temp_text_files"
        os.makedirs(temp_dir, exist_ok=True)

        for file in files:
            try:
                # Save the file locally for processing
                file_path = os.path.join(temp_dir, file.orig_name)
            
                # Handle different file attributes
                if hasattr(file, "name"):  # File has a name attribute, read from the path
                    with open(file.name, "r", encoding="utf-8") as f:
                        content = f.read()
                elif hasattr(file, "value"):  # File content is directly in value
                    content = file.value.decode("utf-8")
                else:
                    raise TypeError("Unsupported file object type.")
            
                # Save the content to a local file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            
                file_contents.append((file.orig_name, content))  # Store filename and content for further processing
            except Exception as e:
                print(f"Error processing file {file.orig_name}: {e}")

            # Process the collected text data
        process_text_data(file_contents)
        return "Files processed successfully!"
    
    def process_text_data(file_contents):
        """
        Example function to process text data. You can replace this with your logic.
        """
        for filename, content in file_contents:
            print(f"Processing file: {filename}")
            print("Content:")
            print(content)

    def ask_question(user_question):
        return user_input(user_question)

    # Separate interfaces for file upload and question-answering
    file_upload_interface = gr.Interface(
        fn=process_uploaded_files,
        inputs=gr.Files(label="Upload Files", file_types=[".pdf", ".jpg", ".jpeg", ".png", ".bmp"]),
        outputs=gr.Textbox(label="File Processing Status")
    )

    question_answering_interface = gr.Interface(
        fn=ask_question,
        inputs=gr.Textbox(label="Ask a question related to the uploaded documents:"),
        outputs=gr.Textbox(label="Answer")
    )

    # Combine interfaces into a tabbed layout
    interface = gr.TabbedInterface(
        interface_list=[file_upload_interface, question_answering_interface],
        tab_names=["Upload Files", "Ask Questions"]
    )

    interface.launch()


if __name__ == "__main__":
    gradio_interface()
