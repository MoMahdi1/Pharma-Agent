"""Connect to Gemini and call it through ordinary functions."""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from .schema import validate_data

load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)


def connect_gemini():
    key = os.getenv('GEMINI_API_KEY')
    if not key or key == 'replace_with_your_private_key':
        raise ValueError('Set GEMINI_API_KEY privately in your environment.')
    return {
        'client': genai.Client(api_key=key, http_options=types.HttpOptions(timeout=60000)),
        'model': os.getenv('GEMINI_MODEL', 'gemini-3.1-flash-lite'),
        'embedding_model': os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001'),
    }


def embed_text(gemini, text, task):
    result = gemini['client'].models.embed_content(
        model=gemini['embedding_model'], contents=text,
        config=types.EmbedContentConfig(task_type=task, output_dimensionality=768),
    )
    if not result.embeddings or not result.embeddings[0].values:
        raise ValueError('Gemini returned no embedding.')
    return result.embeddings[0].values


def generate_answer(gemini, system, prompt, schema):
    result = gemini['client'].models.generate_content(
        model=gemini['model'], contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system, temperature=0,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            response_mime_type='application/json', response_json_schema=schema,
        ),
    )
    if not result.text:
        raise ValueError('Gemini returned no structured answer.')
    data = json.loads(result.text)
    return validate_data(data, schema)
