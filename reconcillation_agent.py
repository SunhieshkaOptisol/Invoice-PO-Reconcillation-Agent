import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from elsai_core.model import AzureOpenAIConnector
from elsai_core.extractors.azure_document_intelligence import AzureDocumentIntelligence
from elsai_core.config.loggerConfig import setup_logger
from elsai_core.prompts import PezzoPromptRenderer
import pandas as pd
 
load_dotenv()
logger = setup_logger()
 
st.set_page_config(page_title="Invoice and PO Comparison", page_icon="📄", layout="wide")
 
# Initialize session state variables if they don't exist
if 'invoice_content' not in st.session_state:
    st.session_state.invoice_content = None
if 'po_content' not in st.session_state:
    st.session_state.po_content = None
if 'invoice_path' not in st.session_state:
    st.session_state.invoice_path = None
if 'po_path' not in st.session_state:
    st.session_state.po_path = None

def extract_content_from_pdf(pdf_path):
    logger.info(f"Extracting from PDF: {os.path.basename(pdf_path)}")
    try:
        doc_processor = AzureDocumentIntelligence(pdf_path)
        extracted_text = doc_processor.extract_text()
        extracted_tables = doc_processor.extract_tables()
        return extracted_text, extracted_tables
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}", exc_info=True)
        raise

def extract_content_from_csv(csv_path):
    logger.info(f"Extracting from CSV: {os.path.basename(csv_path)}")
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        # Convert dataframe to text for processing
        text_content = df.to_string(index=False)
        # Tables are the actual dataframe representation
        tables = [df]
        return text_content, tables
    except Exception as e:
        logger.error(f"CSV extraction error: {str(e)}", exc_info=True)
        raise

def process_file(file_path, document_type):
    try:
        # Determine file type based on extension
        if file_path.lower().endswith('.pdf'):
            content = extract_content_from_pdf(file_path)
        elif file_path.lower().endswith('.csv'):
            content = extract_content_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {os.path.splitext(file_path)[1]}")
        
        # Store in session state
        if document_type == "invoice":
            st.session_state.invoice_content = content
        else:
            st.session_state.po_content = content
            
        return content
    except Exception as e:
        logger.error(f"Processing error: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"
    
def generate_summary(invoice_path, po_path):
    # Check if content exists in session state
    if st.session_state.invoice_content is None or st.session_state.po_content is None:
        # Try to extract content if paths are available but content is not
        if st.session_state.invoice_path and st.session_state.invoice_content is None:
            process_file(st.session_state.invoice_path, "invoice")
        if st.session_state.po_path and st.session_state.po_content is None:
            process_file(st.session_state.po_path, "purchase_order")
            
        # Check again after trying to extract
        if st.session_state.invoice_content is None or st.session_state.po_content is None:
            return "Error: Invoice or PO content not available. Please extract both documents first."

    connector = AzureOpenAIConnector()
    llm = connector.connect_azure_open_ai(deploymentname="gpt-4o-mini")
    renderer = PezzoPromptRenderer(
        api_key=st.secrets["PEZZO_API_KEY"],
        project_id=st.secrets["PEZZO_PROJECT_ID"],
        environment=st.secrets["PEZZO_ENVIRONMENT"],
        server_url=st.secrets["PEZZO_SERVER_URL"]
    )
 
    prompt_name = "PurchaseOrder"
    prompt = renderer.get_prompt(prompt_name)
 
    prompt_txt = f"{prompt}\n\nText: {st.session_state.invoice_content},{st.session_state.po_content}"
    response = llm.invoke(prompt_txt)
    return response.content
 
def process_uploaded_file(uploaded_file, document_type):
    st.write(f"Processing {document_type}: {uploaded_file.name}")
    progress = st.progress(0)
    status = st.empty()
    status.text("Saving file...")
    
    # Get file extension
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    # Create temp file with appropriate extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name
 
    try:
        progress.progress(20)
        status.text("Extracting data...")
        result = process_file(temp_path, document_type)
   
        progress.progress(100)
        status.text("Done ✅")
 
        with st.expander(f"Results: {uploaded_file.name}", expanded=True):
            st.markdown("Extracted Successfully")
        
        # Store the path in session state
        if document_type == "invoice":
            st.session_state.invoice_path = temp_path
        else:
            st.session_state.po_path = temp_path
            
        return result, temp_path
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, temp_path
 
def main():
    st.title("📄 Document Parser")
    st.markdown("Upload Invoices or Purchase Orders (PDF or CSV) and get structured results.")
 
    vision_endpoint = st.secrets.get("VISION_ENDPOINT")
    vision_key = st.secrets.get("VISION_KEY")
    if not vision_endpoint or not vision_key:
        st.error("Azure Vision credentials missing.")
        return
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.subheader("Invoice")
        invoice_file = st.file_uploader("Upload Invoice (PDF or CSV)", type=["pdf", "csv"], key="invoice")
        if invoice_file and st.button("Extract Invoice"):
            _, temp_path = process_uploaded_file(invoice_file, "invoice")
 
    with col2:
        st.subheader("Purchase Order")
        po_file = st.file_uploader("Upload PO (PDF or CSV)", type=["pdf", "csv"], key="po")
        if po_file and st.button("Extract Purchase Order"):
            _, temp_path = process_uploaded_file(po_file, "purchase_order")
 
    st.markdown("---")
    st.subheader("📊 Compare Invoice and Purchase Order")
 
    # Debug information
    if st.checkbox("Show debug info"):
        st.write("Invoice path:", st.session_state.invoice_path)
        st.write("PO path:", st.session_state.po_path)
        st.write("Invoice content exists:", st.session_state.invoice_content is not None)
        st.write("PO content exists:", st.session_state.po_content is not None)
 
    if st.session_state.invoice_path and st.session_state.po_path:
        if st.button("Compare Documents"):
            with st.spinner("Generating comparison summary..."):
                comparison = generate_summary(st.session_state.invoice_path, st.session_state.po_path)
                st.markdown(comparison)
                st.download_button(
                    "Download Comparison",
                    data=comparison,
                    file_name="invoice_po_comparison.md",
                    mime="text/markdown"
                )
    else:
        st.info("Upload both Invoice and Purchase Order to enable comparison.")
 
if __name__ == "__main__":
    main()