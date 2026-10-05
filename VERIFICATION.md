# Observed verification

Verified on October 5, 2026 with the actual project's Python 3.14.8 environment.

- **39 offline tests passed** with `python -m pytest tests -q -p no:cacheprovider`.
- All application modules parse successfully; there are no custom class definitions in application code or tests.
- Real persistent Chroma collections were used with controlled embeddings, covering retrieval, reopening, all five medicines, stale-index detection and uploaded-document filtering.
- PDF page extraction, DOCX text/tables, UTF-8 TXT, chunking (700/150), no-text rejection without OCR, duplicate upsert and failed indexing handling passed.
- Flask tests checked frontend routes, status without secrets, invalid input/origin, missing-key errors, question routing, document selection, file upload and original download.
- Google's SDK accepted plain `response_json_schema` dictionaries; mocked generation verified local schema validation. No live request was made.
- Evaluation and CLI were checked after conversion to functions. Test fixtures use plain functions and dictionaries.
- Runtime dependencies, Flask and test dependencies were installed in the actual project's existing `.venv`. `pip check` found no broken requirements.

## Verification limits

After fixing automatic project `.env` loading, actual Gemini document embedding calls succeeded and 15 corpus chunks were indexed in the user's local ChromaDB. The model's medical accuracy and the full 16-case live evaluation remain unverified. Offline success is not clinical validation. The ZIP excludes API keys, uploaded files, virtual environments and generated databases.

Error-reporting update: 10 relevant tests passed, including quota/access error redaction, specific missing-index messages, question/upload routes, CLI and generation validation. Two new tests bring the available suite to 41 tests; the full expanded suite was not rerun for this small update.

The source corpus contains 15 records for five medicines from six sources. The real NEJM paper is represented by curated abstract-based records, not a redistributed full-paper PDF. Uploaded documents remain unverified evidence.

Live smoke test: Gemini 2.5 Flash returned a model-not-found error, and the flash-latest alias returned a service error. Gemini 3.1 Flash Lite completed a real research query with status answered, cited sources and exact corpus quotations. The local project .env model setting and default were updated accordingly. This one successful live query does not validate medical accuracy or the full evaluation.

Document deletion verification: 10 relevant offline tests passed, including removal across upload embedding models, preservation of other files and sample corpus, invalid-path rejection, same-origin deletion, missing-document responses and unavailable downloads after deletion. JavaScript syntax check passed. The expanded suite now contains 44 tests; the full suite was not rerun for this focused change.
