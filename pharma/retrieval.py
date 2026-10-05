"""Persistent ChromaDB cosine retrieval with Gemini-supplied embeddings."""
import hashlib
import json
import math
import os
from pathlib import Path
from .provider import embed_text


def normalize(vector):
    if not vector or not all(math.isfinite(x) for x in vector):
        raise ValueError('Invalid embedding.')
    norm = math.sqrt(sum(x*x for x in vector))
    if not norm:
        raise ValueError('Zero embedding.')
    return [x/norm for x in vector]


def load_documents(path):
    path = Path(path)
    files = sorted(path.glob('*.json')) if path.is_dir() else [path]
    docs = [d for file in files for d in json.loads(file.read_text(encoding='utf-8'))]
    ids = [d['source_id'] for d in docs]
    if not docs or len(ids) != len(set(ids)):
        raise ValueError('Knowledge base must have unique source IDs.')
    for doc in docs:
        for key in ('source_id','title','text','url','section'):
            if not isinstance(doc.get(key), str) or not doc[key].strip():
                raise ValueError(f'Missing document field: {key}')
        if len(doc['text']) > 4000:
            raise ValueError('Split long documents into section-level chunks first.')
    return docs


def open_store(path):
    store = {}
    import chromadb
    from chromadb.config import Settings
    store['path'] = Path(path)
    store['path'].mkdir(parents=True, exist_ok=True)
    store['client'] = chromadb.PersistentClient(path=str(store['path']), settings=Settings(anonymized_telemetry=False))
    store['collection'] = None
    store['uploads'] = None
    store['manifest'] = store['path'] / 'index.json'
    return store

def fingerprint(docs, model):
    return hashlib.sha256((model + json.dumps(docs, sort_keys=True)).encode()).hexdigest()

def index_documents(store, docs, provider):
    vectors = [normalize(embed_text(provider, d['title']+'\n'+d['text'], 'RETRIEVAL_DOCUMENT')) for d in docs]
    if len({len(v) for v in vectors}) != 1:
        raise ValueError('Inconsistent embedding dimensions.')
    index_id = fingerprint(docs, provider['embedding_model'])
    name = 'pharma-' + index_id[:24]
    collection = store['client'].get_or_create_collection(name=name, embedding_function=None, configuration={'hnsw':{'space':'cosine'}})
    collection.upsert(ids=[d['source_id'] for d in docs], documents=[d['text'] for d in docs], embeddings=vectors, metadatas=[{k:v for k,v in d.items() if k != 'text'} for d in docs])
    temporary = store['path'] / 'index.pending.json'
    temporary.write_text(json.dumps({'fingerprint':index_id,'collection':name,'dimensions':len(vectors[0]),'count':len(docs)}), encoding='utf-8')
    os.replace(temporary, store['manifest'])
    store['collection'] = collection

def check_index(store, docs, provider):
    store['uploads'] = upload_collection(store, provider)
    if not store['manifest'].exists():
        if store['uploads'] is not None and store['uploads'].count():
            return
        raise ValueError('Missing vector index. Run python -m pharma index.')
    info = json.loads(store['manifest'].read_text(encoding='utf-8'))
    if info['fingerprint'] != fingerprint(docs, provider['embedding_model']):
        raise ValueError('Stale vector index. Run python -m pharma index.')
    store['collection'] = store['client'].get_collection(info['collection'], embedding_function=None)
    if store['collection'].count() != info['count']:
        raise ValueError('Incomplete vector index. Reindex the knowledge base.')

def upload_collection(store, provider):
    from chromadb.errors import NotFoundError
    name = 'uploads-' + hashlib.sha256(provider['embedding_model'].encode()).hexdigest()[:24]
    try:
        return store['client'].get_collection(name, embedding_function=None)
    except NotFoundError:
        return None

def add_documents(store, records, provider):
    vectors = [normalize(embed_text(provider, d['title']+'\n'+d['text'], 'RETRIEVAL_DOCUMENT')) for d in records]
    if len({len(v) for v in vectors}) != 1:
        raise ValueError('Inconsistent document embedding dimensions.')
    name = 'uploads-' + hashlib.sha256(provider['embedding_model'].encode()).hexdigest()[:24]
    collection = store['client'].get_or_create_collection(name=name, embedding_function=None, configuration={'hnsw':{'space':'cosine'}}, metadata={'embedding_model':provider['embedding_model'],'dimensions':len(vectors[0])})
    if collection.metadata['dimensions'] != len(vectors[0]):
        raise ValueError('Upload embedding dimensions changed; choose a different embedding model.')
    collection.upsert(ids=[d['source_id'] for d in records], documents=[d['text'] for d in records], embeddings=vectors, metadatas=[{k:v for k,v in d.items() if k != 'text'} for d in records])
    store['uploads'] = collection

def search(store, query, provider, top_k=4, min_score=0.35, document_id=None):
    if store['collection'] is None and store['uploads'] is None:
        raise ValueError('Check or build the vector index first.')
    vector = normalize(embed_text(provider, query, 'RETRIEVAL_QUERY'))
    hits = []
    for collection in [store['collection'],store['uploads']]:
        if document_id and collection is store['collection']:
            continue
        if collection is None or not collection.count():
            continue
        dimensions = json.loads(store['manifest'].read_text(encoding='utf-8'))['dimensions'] if collection is store['collection'] else collection.metadata['dimensions']
        if len(vector) != dimensions:
            raise ValueError('Embedding dimensions changed; reindex.')
        options = {'where':{'document_id':document_id}} if document_id else {}
        result = collection.query(query_embeddings=[vector], n_results=min(top_k,collection.count()), include=['documents','metadatas','distances'], **options)
        for text, metadata, distance in zip(result['documents'][0], result['metadatas'][0], result['distances'][0]):
            score = 1.0-distance
            if score >= min_score:
                hits.append({**metadata,'text':text,'score':score})
    return sorted(hits,key=lambda h:h['score'],reverse=True)[:top_k]

def close_store(store):
    # Persistence is managed by Chroma; PersistentClient has no public close.
    store['collection'] = None
    store['uploads'] = None
