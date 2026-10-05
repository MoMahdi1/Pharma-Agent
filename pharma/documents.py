"""Simple LangChain document loading and splitting. No OCR."""
import hashlib
import json
import re
import uuid
import shutil
from pathlib import Path
from .retrieval import add_documents
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TEXT = 200_000
MAX_CHUNKS = 200


def load_document(file_path):
    ext = Path(file_path).suffix.lower()
    if ext == '.pdf':
        loader = PyPDFLoader(str(file_path))
    elif ext == '.txt':
        loader = TextLoader(str(file_path), encoding='utf-8-sig')
    elif ext == '.docx':
        loader = Docx2txtLoader(str(file_path))
    else:
        raise ValueError(f'Unsupported file type: {ext}')
    try:
        docs = loader.load()
    except Exception:
        raise ValueError('Could not read this file. Use a valid, unlocked PDF, DOCX or UTF-8 TXT.') from None
    if not any(doc.page_content.strip() for doc in docs):
        raise ValueError('No extractable text found. Image-only PDFs are not supported; no OCR is performed.')
    if any('\x00' in doc.page_content for doc in docs):
        raise ValueError('The document contains binary text data.')
    if sum(len(doc.page_content) for doc in docs) > MAX_TEXT:
        raise ValueError('Extracted text exceeds 200,000 characters. Upload a smaller document.')
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150, length_function=len)
    return splitter.split_documents(docs)


def clean_filename(name):
    name = name.replace('\\', '/').split('/')[-1]
    name = re.sub(r'[\x00-\x1f]', '', name).strip()[:160]
    if Path(name).suffix.lower() not in {'.pdf','.docx','.txt'}:
        raise ValueError('Supported files: PDF, DOCX and TXT. Legacy DOC is not supported.')
    return name


def make_records(filename, content):
    """Adapt the browser's bytes to load_document, then retain source metadata."""
    filename = clean_filename(filename)
    if not content or len(content) > MAX_FILE_BYTES:
        raise ValueError('Select a nonempty file of at most 10 MB.')
    staging = Path(__file__).resolve().parents[1] / 'uploads' / '.staging'
    staging.mkdir(parents=True,exist_ok=True)
    file_path = staging / (uuid.uuid4().hex + Path(filename).suffix.lower())
    try:
        file_path.write_bytes(content)
        docs = load_document(file_path)
        chunks = split_documents(docs)
    finally:
        file_path.unlink(missing_ok=True)
    if len(chunks) > MAX_CHUNKS:
        raise ValueError('Document exceeds 200 chunks. Split it into smaller documents.')
    document_id = hashlib.sha256(content).hexdigest()
    records = []
    for i,chunk in enumerate(chunks,1):
        page = chunk.metadata.get('page')
        location = f'Page {page+1}' if isinstance(page,int) else 'Document text'
        records.append({'source_id':f'upload-{document_id[:16]}-{i}', 'title':filename, 'section':f'{location}; chunk {i}', 'url':f'/api/documents/{document_id}', 'text':chunk.page_content, 'kind':'uploaded-document', 'document_id':document_id, 'filename':filename})
    warnings = ['Pages without extractable text were skipped; no OCR is performed.'] if any(not d.page_content.strip() for d in docs) else []
    return document_id,records,warnings


def upload_document(root, filename, content, provider, store):
    document_id, records, warnings = make_records(filename, content)
    filename = records[0]['filename']
    folder = Path(root) / document_id
    folder.mkdir(parents=True,exist_ok=True)
    original = folder / ('original' + Path(filename).suffix.lower())
    original.write_bytes(content)
    # Existing Gemini embeddings and Chroma persistence stay unchanged.
    add_documents(store, records, provider)
    info = {'id':document_id,'filename':filename,'chunks':len(records),'warnings':warnings,'embedding_model':provider['embedding_model'],'original':original.name}
    pending=folder/'document.pending.json'
    pending.write_text(json.dumps(info),encoding='utf-8')
    pending.replace(folder/'document.json')
    return info


def list_uploads(root):
    return [json.loads(file.read_text(encoding='utf-8')) for file in sorted(Path(root).glob('*/document.json'))]


def delete_document(root, document_id, store):
    if not re.fullmatch('[a-f0-9]{64}', document_id):
        raise ValueError('Invalid document selection.')
    root = Path(root).resolve()
    folder = root / document_id
    if folder.is_symlink() or folder.resolve().parent != root:
        raise ValueError('Invalid document location.')
    if not (folder / 'document.json').is_file():
        raise FileNotFoundError('Document not found.')
    # Delete from every upload embedding model, preserving the sample corpus.
    for collection in store['client'].list_collections():
        if collection.name.startswith('uploads-'):
            collection.delete(where={'document_id': document_id})
    shutil.rmtree(folder)
