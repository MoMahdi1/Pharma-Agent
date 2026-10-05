"""A simple Flask server. Run with python -m pharma.web."""
import json
import os
import re
from pathlib import Path
from urllib.parse import unquote
from flask import Flask, request, jsonify, send_file
from .agent import ask_question
from .errors import error_message
from .provider import connect_gemini
from .retrieval import open_store, check_index, close_store, load_documents
from .documents import MAX_FILE_BYTES, clean_filename, make_records, upload_document, list_uploads
from .documents import delete_document

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_ROOT = ROOT / 'uploads'
PORT = 8000
app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_BYTES


def reply(status, data):
    return jsonify(data), status


@app.before_request
def check_origin():
    origin = request.headers.get('Origin')
    if request.method != 'POST' or origin is None:
        return None
    normalized_origin = origin.rstrip('/')
    # Local port forwarding can expose Flask on a different browser port.
    raw_host = request.host
    host = (raw_host[1:raw_host.index(']')] if raw_host.startswith('[') and ']' in raw_host
            else raw_host.split(':', 1)[0]).lower()
    local_same_origin = (
        host in {'127.0.0.1', 'localhost', '::1'}
        and normalized_origin == request.host_url.rstrip('/')
    )
    allowed_origins = set()
    configured_origin = os.getenv('APP_ORIGIN')
    if configured_origin:
        allowed_origins.add(configured_origin.rstrip('/'))
    railway_domain = os.getenv('RAILWAY_PUBLIC_DOMAIN')
    if railway_domain:
        allowed_origins.add(('https://' + railway_domain).rstrip('/'))
    if not local_same_origin and normalized_origin not in allowed_origins:
        return reply(403, {'error': 'Origin not allowed'})


@app.after_request
def add_headers(response):
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
    return response


@app.errorhandler(413)
def too_large(error):
    return reply(413, {'error': 'File limit: 10 MB. Select a nonempty PDF, DOCX or TXT.'})


@app.route('/')
def home():
    return send_file(ROOT / 'frontend' / 'index.html')


@app.route('/style.css')
def styles():
    return send_file(ROOT / 'frontend' / 'style.css')


@app.route('/app.js')
def javascript():
    return send_file(ROOT / 'frontend' / 'app.js')


@app.route('/api/status')
def status():
    docs = load_documents(ROOT / 'knowledge')
    uploaded = list_uploads(UPLOAD_ROOT)
    model = os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')
    current = [d for d in uploaded if d['embedding_model'] == model]
    return jsonify({
        'key_configured': bool(os.getenv('GEMINI_API_KEY')),
        'index_present': (ROOT / 'chroma_db/index.json').exists() or bool(current),
        'document_count': len(docs) + sum(d['chunks'] for d in current),
        'drug_count': len({d['drug'] for d in docs}),
        'source_count': len({d['url'] for d in docs}) + len(current),
    })


@app.route('/api/documents')
def documents():
    return jsonify({'documents': list_uploads(UPLOAD_ROOT)})


@app.route('/api/documents/<document_id>')
def download_document(document_id):
    if not re.fullmatch('[a-f0-9]{64}', document_id):
        return reply(404, {'error': 'Document not found'})
    folder = UPLOAD_ROOT / document_id
    info_path = folder / 'document.json'
    if not info_path.exists():
        return reply(404, {'error': 'Document not found'})
    info = json.loads(info_path.read_text(encoding='utf-8'))
    return send_file(folder / info['original'], as_attachment=True,
                     download_name='document' + Path(info['original']).suffix)


@app.route('/api/documents/<document_id>/delete', methods=['POST'])
def remove_document(document_id):
    if not re.fullmatch('[a-f0-9]{64}', document_id):
        return reply(400, {'error': 'Invalid document selection.'})
    store = None
    try:
        store = open_store(ROOT / 'chroma_db')
        delete_document(UPLOAD_ROOT, document_id, store)
        return jsonify({'message': 'Document deleted from files and the search index.'})
    except FileNotFoundError:
        return reply(404, {'error': 'Document not found.'})
    except ValueError:
        return reply(400, {'error': 'Invalid document location.'})
    except Exception as exc:
        return reply(503, {'error': error_message(exc)})
    finally:
        if store:
            close_store(store)


@app.route('/api/ask', methods=['POST'])
def ask():
    if request.mimetype != 'application/json':
        return reply(415, {'error': 'JSON required'})
    if not request.content_length or request.content_length > 16000:
        return reply(400, {'error': 'Invalid request size'})
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return reply(400, {'error': 'Invalid JSON request'})
    query = data.get('query')
    document_id = data.get('document_id')
    if document_id is not None and (not isinstance(document_id, str) or not re.fullmatch('[a-f0-9]{64}', document_id)):
        return reply(400, {'error': 'Invalid document selection.'})
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 3000:
        return reply(400, {'error': 'Enter a question of 1–3000 characters.'})
    if not os.getenv('GEMINI_API_KEY'):
        return reply(503, {'error': 'Set GEMINI_API_KEY in the server environment, then restart the server.'})
    store = None
    try:
        provider = connect_gemini()
        store = open_store(ROOT / 'chroma_db')
        check_index(store, load_documents(ROOT / 'knowledge'), provider)
        k = int(os.getenv('RETRIEVAL_TOP_K', '4'))
        threshold = float(os.getenv('RETRIEVAL_MIN_SCORE', '0.35'))
        if k < 1 or not -1 <= threshold <= 1:
            return reply(503, {'error': 'Invalid server retrieval settings.'})
        result = ask_question(query, provider, store, k, threshold, document_id=document_id)
        return jsonify(result)
    except Exception as exc:
        return reply(503, {'error': error_message(exc)})
    finally:
        if store:
            close_store(store)


@app.route('/api/upload', methods=['POST'])
def upload():
    if request.mimetype != 'application/octet-stream':
        return reply(415, {'error': 'Upload a file using the document form.'})
    if not request.content_length or request.content_length > MAX_FILE_BYTES:
        return too_large(None)
    try:
        filename = clean_filename(unquote(request.headers.get('X-Filename', '')))
        content = request.get_data()
        make_records(filename, content)
    except ValueError as exc:
        return reply(400, {'error': str(exc)})
    if not os.getenv('GEMINI_API_KEY'):
        return reply(503, {'error': 'Set GEMINI_API_KEY in the server environment before indexing documents.'})
    store = None
    try:
        provider = connect_gemini()
        store = open_store(ROOT / 'chroma_db')
        info = upload_document(UPLOAD_ROOT, filename, content, provider, store)
        return jsonify({'document': info, 'message': 'Document indexed. You can now ask research questions about it.'})
    except Exception as exc:
        return reply(503, {'error': error_message(exc)})
    finally:
        if store:
            close_store(store)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=PORT, debug=False)
