'use strict';
const form = document.querySelector('#research-form');
const question = document.querySelector('#question');
const submit = document.querySelector('#submit');
const feedback = document.querySelector('#feedback');
const results = document.querySelector('#results');
let lastAnswer = null;
let busy = false;
let deleting = false;
const deleteButton = document.querySelector('#delete-document');
const scopeSelect = document.querySelector('#document-scope');
function updateDeleteButton() {
  deleteButton.disabled = !scopeSelect.value || busy || deleting;
}
scopeSelect.addEventListener('change', updateDeleteButton);
deleteButton.addEventListener('click', async () => {
  const id = scopeSelect.value;
  if (!id || busy || deleting) return;
  const name = scopeSelect.selectedOptions[0].textContent;
  if (!window.confirm(`Delete ${name} and its search index?`)) return;
  deleting = true;
  updateDeleteButton();
  submit.disabled = true;
  document.querySelector('#upload-button').disabled = true;
  scopeSelect.disabled = true;
  message('Deleting document…');
  try {
    const response = await fetch(`/api/documents/${id}/delete`, {method: 'POST'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not delete document.');
    scopeSelect.value = '';
    await loadDocuments();
    await serviceStatus();
    lastAnswer = null;
    results.hidden = true;
    message('Document deleted. Search scope is now All indexed evidence.');
  } catch (error) { message(error.message, true); }
  finally {
    deleting = false;
    submit.disabled = false;
    document.querySelector('#upload-button').disabled = false;
    scopeSelect.disabled = false;
    updateDeleteButton();
  }
});
const statuses = {answered: 'Evidence found', insufficient_evidence: 'Insufficient evidence', refused: 'Outside research scope'};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
function count() { document.querySelector('#char-count').textContent = `${question.value.length} / 3000`; }
question.addEventListener('input', count);
document.querySelectorAll('[data-query]').forEach(button => button.addEventListener('click', () => {
  question.value = button.dataset.query;
  count();
  question.focus();
}));
function message(text, error = false) {
  feedback.hidden = false;
  feedback.className = error ? 'error' : '';
  feedback.textContent = text;
}
function render(answer) {
  lastAnswer = answer;
  document.querySelector('#empty-state').hidden = true;
  document.querySelector('#result-topic').textContent = answer.topic;
  document.querySelector('#result-status').textContent = statuses[answer.status] || answer.status;
  document.querySelector('#confidence').textContent = answer.confidence === 'none' ? 'No evidence-based answer' : 'Limited corpus evidence · not clinical certainty';
  document.querySelector('#explanation').textContent = answer.explanation;
  document.querySelector('#disclaimer').textContent = answer.disclaimer;
  const claims = document.querySelector('#claims');
  claims.replaceChildren();
  if (answer.status === 'answered') {
    for (const [field, title] of [['summary','Research summary'],['risks','Reported risks & contraindications'],['interactions','Reported interactions']]) {
      const section = element('section', undefined, 'card claim-section');
      section.append(element('h3', title));
      const entries = answer[field];
      if (!entries.length) section.append(element('p', 'No findings included for this section in the current response.', 'no-claims'));
      for (const claim of entries) {
        const card = element('div', undefined, 'claim');
        card.append(element('p', claim.text));
        for (const evidence of claim.evidence) {
          card.append(element('blockquote', evidence.quote), element('small', `Evidence: ${evidence.source_id}`));
        }
        section.append(card);
      }
      claims.append(section);
    }
  }
  const sources = document.querySelector('#sources');
  sources.replaceChildren();
  document.querySelector('#sources-section').hidden = !answer.sources.length;
  for (const source of answer.sources) {
    const link = element('a', `${source.title} ↗`, 'source');
    try {
      const url = new URL(source.url, window.location.origin);
      if (url.protocol === 'https:' || /^\/api\/documents\/[a-f0-9]{64}$/.test(source.url)) { link.href = url.href; link.target = '_blank'; link.rel = 'noopener noreferrer'; }
    } catch { /* Source metadata remains visible without an unsafe link. */ }
    link.append(element('small', `${source.source_id} · Section ${source.section}`));
    sources.append(link);
  }
  results.hidden = false;
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (busy || deleting) return;
  const query = question.value.trim();
  if (busy) return;
  if (!query) { message('Enter a research question first.', true); question.focus(); return; }
  busy = true;
  updateDeleteButton();
  scopeSelect.disabled = true;
  document.querySelector('#upload-button').disabled = true;
  submit.disabled = true;
  question.disabled = true;
  document.querySelectorAll('[data-query]').forEach(button => button.disabled = true);
  results.hidden = true;
  lastAnswer = null;
  message('Retrieving evidence and checking the answer. This may take a moment…');
  try {
    const selected = document.querySelector('#document-scope').value;
    const response = await fetch('/api/ask', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query, ...(selected ? {document_id:selected} : {})})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The research request failed.');
    render(data);
    feedback.hidden = true;
    results.scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) {
    message(error.message === 'Failed to fetch' ? 'Cannot reach the local server. Start it with python -m pharma.web.' : error.message, true);
  } finally {
    busy = false;
    scopeSelect.disabled = false;
    document.querySelector('#upload-button').disabled = false;
    updateDeleteButton();
    submit.disabled = false;
    question.disabled = false;
    document.querySelectorAll('[data-query]').forEach(button => button.disabled = false);
  }
});
document.querySelector('#download').addEventListener('click', () => {
  if (!lastAnswer) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastAnswer, null, 2)], {type: 'application/json'}));
  const link = element('a'); link.href = url; link.download = 'pharma-research.json';
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
async function serviceStatus() {
  const label = document.querySelector('#service-text');
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('status');
    const data = await response.json();
    document.querySelector('#document-count').textContent = String(data.document_count);
    document.querySelector('#source-count').textContent = String(data.source_count);
    label.textContent = !data.key_configured ? 'Setup needed: Gemini API key' : !data.index_present ? 'Setup needed: index knowledge base' : 'Server configured · live access unverified';
    document.querySelector('#service-dot').classList.toggle('warning', !data.key_configured || !data.index_present);
  } catch { label.textContent = 'Local server unavailable'; document.querySelector('#service-dot').classList.add('warning'); }
}
serviceStatus();

async function loadDocuments(selected) {
  const response = await fetch('/api/documents');
  if (!response.ok) throw new Error('Could not load uploaded documents.');
  const data = await response.json();
  const scope = document.querySelector('#document-scope');
  const previous = selected || scope.value;
  scope.replaceChildren(element('option','All indexed evidence'));
  scope.firstChild.value = '';
  const list = document.querySelector('#document-list');
  list.replaceChildren();
  for (const doc of data.documents) {
    const option = element('option',doc.filename); option.value = doc.id; scope.append(option);
    const item = element('li');
    item.append(element('strong',doc.filename), element('small',`${doc.chunks} chunks · indexed with ${doc.embedding_model}`));
    list.append(item);
  }
  if ([...scope.options].some(option => option.value === previous)) scope.value = previous;
  updateDeleteButton();
}
document.querySelector('#upload-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (busy || deleting) return;
  const input = document.querySelector('#document-file');
  const file = input.files[0];
  const button = document.querySelector('#upload-button');
  const status = document.querySelector('#upload-feedback');
  status.hidden = false; status.className = '';
  if (!file || !/\.(pdf|docx|txt)$/i.test(file.name) || file.size === 0 || file.size > 10*1024*1024) {
    status.className = 'error'; status.textContent = 'Choose a nonempty PDF, DOCX or TXT of at most 10 MB.'; return;
  }
  button.disabled = true; input.disabled = true;
  busy = true;
  submit.disabled = true;
  scopeSelect.disabled = true;
  updateDeleteButton();
  status.textContent = 'Extracting text and indexing with Gemini… Keep this page open.';
  try {
    const response = await fetch('/api/upload', {method:'POST', headers:{'Content-Type':'application/octet-stream','X-Filename':encodeURIComponent(file.name)}, body:file});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Document upload failed.');
    status.textContent = `${data.document.filename}: ${data.document.chunks} chunks indexed. ${(data.document.warnings || []).join(' ')}`;
    await loadDocuments(data.document.id);
    await serviceStatus();
    question.value = `Summarize the pharmaceutical research in ${data.document.filename}, including findings and limitations.`;
    count();
  } catch (error) { status.className = 'error'; status.textContent = error.message; }
  finally {
    busy = false;
    button.disabled = false; input.disabled = false;
    submit.disabled = false; scopeSelect.disabled = false;
    updateDeleteButton();
  }
});
  loadDocuments().catch(() => { /* Status area already indicates connection availability. */ });
