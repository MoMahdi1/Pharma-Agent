"""Useful error messages without exposing API keys or request contents."""


def error_message(exc):
    local_errors = (
        'Missing vector index. Run python -m pharma index.',
        'Stale vector index. Run python -m pharma index.',
        'Incomplete vector index. Reindex the knowledge base.',
        'Set GEMINI_API_KEY privately in your environment.',
        'Response does not match the required JSON format.',
        'Gemini returned no embedding.',
        'Gemini returned no structured answer.',
    )
    if isinstance(exc, ValueError) and str(exc) in local_errors:
        return str(exc)
    code = getattr(exc, 'code', None)
    if code in (401, 403):
        return 'Gemini rejected API access. Check your API key and model permissions.'
    if code == 429:
        return 'Gemini quota or rate limit reached. Check your quota and retry later.'
    if code == 404:
        return 'Gemini model not found. Check GEMINI_MODEL and GEMINI_EMBEDDING_MODEL.'
    if code == 400:
        return 'Gemini rejected the request. Check API key validity, model configuration and input format.'
    if isinstance(code, int) and code >= 500:
        return 'Gemini is temporarily unavailable. Retry later.'
    if isinstance(exc, ImportError):
        return 'Missing dependency. Run python -m pip install -r requirements.txt.'
    return f'Service failed ({type(exc).__name__}). Check connectivity and the database; retry indexing if needed.'
