import os
tesseract_path = "/usr/bin/tesseract"  # Replace with the actual path to tesseract
os.environ["PATH"] += os.pathsep + tesseract_path
from uuid import uuid4
import openai
import streamlit as st
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tempfile import NamedTemporaryFile
import tempfile
import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#from langchain_google_vertexai import ChatVertexAI
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
import openai
from unstructured.partition.pdf import partition_pdf
import uuid
import base64
from langchain.schema.messages import HumanMessage, SystemMessage
import base64
from langchain_core.messages import HumanMessage
import uuid
from langchain.embeddings import VertexAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import io
import re
from IPython.display import HTML, display
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from PIL import Image
from langchain.chat_models import ChatOpenAI
from chromadb.config import Settings
import chromadb



os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "multi_model_rag_mvr"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
api_key = st.secrets["GOOGLE_API_KEY"]
openai.api_key = st.secrets["OPENAI_API_KEY"]



st.set_page_config(layout='wide', initial_sidebar_state='expanded')

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.sidebar.header('Multi-Modal RAG App`PDF`')

st.sidebar.subheader('Text Summarization Model')
time_hist_color = st.sidebar.selectbox('Summarize by', ('gpt-4-turbo', 'gemini-1.5-pro-latest', 'gpt-4o','llama3'))

st.sidebar.subheader('Image Summarization Model')
immage_sum_model = st.sidebar.selectbox('Summarize by', ('gpt-4-vision-preview', 'gemini-1.5-pro-latest','gpt-4o'))

#st.sidebar.subheader('Embedding Model')
#embedding_model = st.sidebar.selectbox('Select data', ('OpenAIEmbeddings', 'GoogleGenerativeAIEmbeddings'))

st.sidebar.subheader('Response Generation Model')
generation_model = st.sidebar.selectbox('Select data', ('gpt-4-vision-preview', 'gemini-1.5-pro-latest','gpt-4o'))


max_concurrecy = st.sidebar.slider('Maximum Concurrency', 3, 4, 7)


st.sidebar.subheader('Upload your file')
uploaded_file = st.sidebar.file_uploader(label = "Upload your file",type="pdf")

st.sidebar.markdown('''
---
Multi-Modal RAG App with Multi Vector Retriever
''')

#st.write(tables)


bullet_point = "◇"
question = st.text_input('Enter a question')
pr = st.button("Generate")

if uploaded_file is not None:
    
    if "pdf_elements" not in st.session_state:
        st.title("Extraction process:-")
        st.write(f"{bullet_point} Extraction process started")
        temp_file="./temp.pdf"
        with open(temp_file,"wb") as file:
            file.write(uploaded_file.getvalue())
       
        image_path = "./"

        #@st.cache_data(show_spinner=False)
        def pdf_ele(image_path,ele_path):
            pdf_elements = partition_pdf(
                ele_path,
                chunking_strategy="by_title",
                #chunking_strategy="basic",
                extract_images_in_pdf=True,
                infer_table_structure=True,
                strategy='hi_res',
                max_characters=3200,
                new_after_n_chars=3000,
                combine_text_under_n_chars=2200,
                image_output_dir_path=image_path
            )
            return pdf_elements
        
        pdf_elements = pdf_ele(image_path,temp_file)
        st.session_state["pdf_elements"] = pdf_elements
        st.write(f"{bullet_point} Extraction process completed")
    else:
        # st.write(f"{bullet_point} Extraction already done") 
        pdf_elements = st.session_state["pdf_elements"]        

    # Categorize elements by type
    #@st.cache_data(show_spinner=False)
    def categorize_elements(_raw_pdf_elements):
      """
      Categorize extracted elements from a PDF into tables and texts.
      raw_pdf_elements: List of unstructured.documents.elements
      """
      tables = []
      texts = []
      for element in _raw_pdf_elements:
          if "unstructured.documents.elements.Table" in str(type(element)):
              tables.append(str(element))
          elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
              texts.append(str(element))
      return texts, tables

    texts, tables = categorize_elements(pdf_elements)

    if "texts" not in st.session_state or "tables" not in st.session_state:
        # Create session state variables
        with st.spinner("Categorizing Text & Table elements....."):
            texts, tables = categorize_elements(pdf_elements)
        st.session_state["texts"] = texts
        st.session_state["tables"] = tables
        st.write(f"{bullet_point} \t\tCategorize elements completed") 
    else:
        # Use already populated session state variables
        texts = st.session_state["texts"]
        tables = st.session_state["tables"]
    
    

    # Generate summaries of text elements
    #@st.cache_data(show_spinner=False)
    def generate_text_summaries(texts, tables, summarize_texts=False):
      """
      Summarize text elements
      texts: List of str
      tables: List of str
      summarize_texts: Bool to summarize texts
      """

      # Prompt
      prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
      These summaries will be embedded and used to retrieve the raw text or table elements. \
      Give a concise summary of the table or text that is well-optimized for retrieval. Table \
      or text: {element} """

      prompt = PromptTemplate.from_template(prompt_text)
      empty_response = RunnableLambda(
          lambda x: AIMessage(content="Error processing document")
      )
      # Text summary chain

      if time_hist_color == 'gpt-4-turbo':
        model = ChatOpenAI(
          temperature=0, model= "gpt-4-turbo", openai_api_key = openai.api_key, max_tokens=1024)

      elif time_hist_color == 'gemini-1.5-pro-latest':
        model = ChatGoogleGenerativeAI(
            #temperature=0, model="gemini-pro", max_output_tokens=1024
          temperature=0, model="gemini-1.5-pro-latest", max_output_tokens=1024
        )
      else:
        model = ChatOpenAI(model="gpt-4o", openai_api_key = openai.api_key, max_tokens=1024)

      summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

      # Initialize empty summaries
      text_summaries = []
      table_summaries = []

      # Apply to text if texts are provided and summarization is requested
      if texts and summarize_texts:
          text_summaries = summarize_chain.batch(texts, {"max_concurrency": 10})
      elif texts:
          text_summaries = texts

      # Apply to tables if tables are provided
      if tables:
          table_summaries = summarize_chain.batch(tables, {"max_concurrency":10})

      return text_summaries, table_summaries
        

    
    if "text_summaries" not in st.session_state or "table_summaries" not in st.session_state:  
        st.title("Summary generation process:-")
        st.write(f"{bullet_point} Summary generation process started")   
        # Create session state variables
        with st.spinner("Generating Text & Table summaries....."):    
            text_summaries, table_summaries = generate_text_summaries(texts, tables, summarize_texts=True)
        st.session_state["text_summaries"] = text_summaries
        st.session_state["table_summaries"] = table_summaries
        st.write(f"{bullet_point} \t\tText & Table summaries generation completed")     
    else:
        # Use already populated session state variables
        text_summaries = st.session_state["text_summaries"]
        table_summaries = st.session_state["table_summaries"]
     
                                                                                            
    
    

    #@st.cache_data(show_spinner=False)
    def encode_image(image_path):
      """Getting the base64 string"""
      with open(image_path, "rb") as image_file:
          return base64.b64encode(image_file.read()).decode("utf-8")
   # @st.cache_data(show_spinner=False)
    def image_summarize(img_base64, prompt):
      """Make image summary"""
      if immage_sum_model == 'gpt-4-vision-preview':
        model = ChatOpenAI(
          temperature=0, model=immage_sum_model, openai_api_key = openai.api_key, max_tokens=1024)
      elif immage_sum_model == 'gemini-1.5-pro-latest':
        #model = ChatGoogleGenerativeAI(model="gemini-pro-vision", max_output_tokens=1024)
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", max_output_tokens=1024)
      else:
        model = ChatOpenAI(model="gpt-4o", openai_api_key = openai.api_key, max_tokens=1024)
    
      msg = model(
          [
              HumanMessage(
                  content=[
                      {"type": "text", "text": prompt},
                      {
                          "type": "image_url",
                          "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                      },
                  ]
              )
          ]
      )
      return msg.content
    
    #@st.cache_data(show_spinner=False)
    def generate_img_summaries(path):
        """
        Generate summaries and base64 encoded strings for images
        path: Path to list of .jpg files extracted by Unstructured
        """
        
        # Store base64 encoded images
        img_base64_list = []
        
        # Store image summaries
        image_summaries = []
        
        # Prompt
        prompt = """You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval."""
        
        # Apply to images
        for img_file in os.listdir(path):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(path, img_file)
                base64_image = encode_image(img_path)
                img_base64_list.append(base64_image)
                image_summaries.append(image_summarize(base64_image, prompt))
        
        return img_base64_list, image_summaries

    fpath= './figures'
    if 'image_summaries' not in st.session_state:
        with st.spinner("Generating Images summaries......"):
            img_base64_list, image_summaries = generate_img_summaries(fpath)

        st.session_state["img_base64_list"] = img_base64_list
        st.session_state["image_summaries"] = image_summaries
        st.write(f"{bullet_point} \t\tImage summaries generation completed") 
    else:
        img_base64_list = st.session_state["img_base64_list"]  
        image_summaries = st.session_state["image_summaries"]  
    
 
    def create_multi_vector_retriever(
      vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
    ):
      """
      Create retriever that indexes summaries, but returns raw images or texts
      """
    
      # Initialize the storage layer
      store = InMemoryStore()
      id_key = "doc_id"
    
      # Create the multi-vector retriever
      retriever = MultiVectorRetriever(
          vectorstore=vectorstore,
          docstore=store,
          id_key=id_key,
      )
      # Helper function to add documents to the vectorstore and docstore
      def add_documents(retriever, doc_summaries, doc_contents):
          doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
          summary_docs = [
              Document(page_content=s, metadata={id_key: doc_ids[i]})
              for i, s in enumerate(doc_summaries)
          ]
          retriever.vectorstore.add_documents(summary_docs)
          retriever.docstore.mset(list(zip(doc_ids, doc_contents)))
    
      # Add texts, tables, and images
      # Check that text_summaries is not empty before adding
      if text_summaries:
          add_documents(retriever, text_summaries, texts)
      # Check that table_summaries is not empty before adding
      if table_summaries:
          add_documents(retriever, table_summaries, tables)
      # Check that image_summaries is not empty before adding
      if image_summaries:
          add_documents(retriever, image_summaries, images)
      return retriever
    
    
    client = chromadb.Client()

    
    def looks_like_base64(sb):
      """Check if the string looks like base64"""
      return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None
    
    
    def is_image_data(b64data):
      """
      Check if the base64 data is an image by looking at the start of the data
      """
      image_signatures = {
          b"\xFF\xD8\xFF": "jpg",
          b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
          b"\x47\x49\x46\x38": "gif",
          b"\x52\x49\x46\x46": "webp",
      }
      try:
          header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
          for sig, format in image_signatures.items():
              if header.startswith(sig):
                  return True
          return False
      except Exception:
          return False
    
    def resize_base64_image(base64_string, size=(128, 128)):
      """
      Resize an image encoded as a Base64 string
      """
      # Decode the Base64 string
      img_data = base64.b64decode(base64_string)
      img = Image.open(io.BytesIO(img_data))
    
      # Resize the image
      resized_img = img.resize(size, Image.LANCZOS)
    
      # Save the resized image to a bytes buffer
      buffered = io.BytesIO()
      resized_img.save(buffered, format=img.format)
      # Encode the resized image to Base64
      return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    #context creation
    def split_image_text_types(docs):
      """
      Split base64-encoded images and texts
      """
      b64_images = []
      texts = []
      for doc in docs:
          # Check if the document is of type Document and extract page_content if so
          if isinstance(doc, Document):
              doc = doc.page_content
          if looks_like_base64(doc) and is_image_data(doc):
              doc = resize_base64_image(doc, size=(1300, 600))
              b64_images.append(doc)
          else:
              texts.append(doc)
      if len(b64_images) > 0:
          return {"images": b64_images[:1], "texts": []}
      return {"images": b64_images, "texts": texts}
    
    
    #response generation
    def img_prompt_func(data_dict):
      """
      Join the context into a single string
      """
      formatted_texts = "\n".join(data_dict["context"]["texts"])
      messages = []
    
      # Adding the text for analysis
      text_message = {
          "type": "text",
          "text": (
              "You are an AI scientist tasking with providing factual answers.\n"
              "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
              "Use this information to provide answers related to the user question. \n"
              "Final answer should be easily readable and structured. \n"
              f"User-provided question: {data_dict['question']}\n\n"
              "Text and / or tables:\n"
              f"{formatted_texts}"
          ),
      }
      messages.append(text_message)
      # Adding image(s) to the messages if present
      if data_dict["context"]["images"]:
          for image in data_dict["context"]["images"]:
              image_message = {
                  "type": "image_url",
                  "image_url": {"url": f"data:image/jpeg;base64,{image}"},
              }
              messages.append(image_message)
      return [HumanMessage(content=messages)]

    def multi_modal_rag_chain(retriever):
        """
        Multi-modal RAG chain
        """

        if generation_model == 'gemini-1.5-pro-latest':
            model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",max_output_tokens=1024)
        elif generation_model == 'gpt-4-vision-preview':
            try:
              model = ChatOpenAI(model="gpt-4-vision-preview", openai_api_key = openai.api_key, max_tokens=1024)
            except Exception as e:
              model = ChatOpenAI(model="gpt-4-turbo", openai_api_key = openai.api_key, max_tokens=1024)
        else:
            model = ChatOpenAI(model="gpt-4o", openai_api_key = openai.api_key, max_tokens=1024)


        # RAG pipeline
        chain = (
            {
                "context": retriever | RunnableLambda(split_image_text_types),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(img_prompt_func)
            | model
            | StrOutputParser()
        )

        return chain
    

    
    
     
    
    
    if pr==True:
        vectorstore = Chroma(collection_name="mm_rag_mistral04",embedding_function=OpenAIEmbeddings(openai_api_key = openai.api_key))
        retriever_multi_vector_img=create_multi_vector_retriever(vectorstore,text_summaries,texts,table_summaries,tables,image_summaries,img_base64_list)
        chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)
        docs = retriever_multi_vector_img.get_relevant_documents(question, limit=1)
        st.write(docs)
    
        response= chain_multimodal_rag.invoke(question)
        st.write(response)
    
    
        
        found_image = False  # Flag variable to track if an image has been found
    
        for i in range(len(docs)):
          if docs[i].startswith('/9j') and not found_image:
              #display.display(HTML(f'<img src="data:image/jpeg;base64,{docs[i]}">'))
    
              base64_image = docs[i]
              image_data = base64.b64decode(base64_image)
    
              # Display the image
              #img = Image.open(BytesIO(image_data))
              #img.show()
              #img = load_image(image_data)
              st.image(image_data)
              found_image = True  # Set the flag to True to indicate that an image has been found
        client.delete_collection("mm_rag_mistral04")
          
    
    
