"""JSON response formats: plain dictionaries, no model classes."""
from jsonschema import validate, ValidationError


def object_schema(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def array_schema(items, minimum=0):
    return {'type': 'array', 'items': items, 'minItems': minimum}


TEXT = {'type': 'string'}
NONEMPTY_TEXT = {'type': 'string', 'minLength': 1}
BOOLEAN = {'type': 'boolean'}
EVIDENCE = object_schema({'source_id': TEXT, 'quote': NONEMPTY_TEXT})
CLAIM = object_schema({'text': NONEMPTY_TEXT, 'evidence': array_schema(EVIDENCE, 1)})
ANSWER = object_schema({
    'topic': TEXT,
    'status': {'type': 'string', 'enum': ['answered', 'insufficient_evidence', 'refused']},
    'summary': array_schema(CLAIM), 'risks': array_schema(CLAIM),
    'interactions': array_schema(CLAIM), 'explanation': TEXT,
})
PLAN = object_schema({'safe': BOOLEAN, 'in_domain': BOOLEAN, 'search_query': TEXT})
REVIEW = object_schema({'supported': BOOLEAN, 'safe': BOOLEAN, 'sufficient': BOOLEAN})
SOURCE = object_schema({'source_id': TEXT, 'title': TEXT, 'url': TEXT, 'section': TEXT})
RESPONSE = object_schema({
    **ANSWER['properties'], 'sources': array_schema(SOURCE),
    'confidence': {'type': 'string', 'enum': ['limited_evidence', 'none']},
    'disclaimer': TEXT,
})


def validate_data(data, schema):
    try:
        validate(instance=data, schema=schema)
    except ValidationError:
        raise ValueError('Response does not match the required JSON format.') from None
    return data
