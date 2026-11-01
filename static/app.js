const $ = (selector) => document.querySelector(selector);
const state = { busy: false };

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || '请求失败');
  return data;
}
function toast(message) {
  $('#toast').textContent = message;
  $('#toast').hidden = false;
  setTimeout(() => { $('#toast').hidden = true; }, 2800);
}
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
}
async function loadDocuments() {
  const docs = await api('/api/documents');
  $('#doc-count').textContent = docs.length;
  $('#documents').innerHTML = docs.length ? docs.map(doc => `<div class="doc-item"><span class="doc-icon">▤</span><span class="doc-name" title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</span><button class="delete-doc" data-id="${doc.id}" aria-label="删除 ${escapeHtml(doc.name)}">×</button></div>`).join('') : '<p class="muted">还没有文档。<br>上传后即可开始检索。</p>';
  document.querySelectorAll('.delete-doc').forEach(button => button.addEventListener('click', async () => {
    if (!confirm('从知识库删除这份文档？')) return;
    await api(`/api/documents/${button.dataset.id}`, { method: 'DELETE' });
    loadDocuments();
  }));
}
async function upload(file) {
  if (!file || state.busy) return;
  const body = new FormData();
  body.append('file', file);
  $('#upload-progress').hidden = false;
  $('#upload-progress').textContent = `正在解析并建立索引：${file.name}`;
  try {
    await api('/api/documents', { method: 'POST', body });
    await loadDocuments();
    toast('文档已加入知识库');
  } catch (error) { toast(error.message); }
  finally { $('#upload-progress').hidden = true; $('#file-input').value = ''; }
}
$('#dropzone').addEventListener('click', () => $('#file-input').click());
$('#upload-top').addEventListener('click', () => $('#file-input').click());
$('#file-input').addEventListener('change', event => upload(event.target.files[0]));
['dragenter', 'dragover'].forEach(type => $('#dropzone').addEventListener(type, event => { event.preventDefault(); $('#dropzone').classList.add('dragging'); }));
['dragleave', 'drop'].forEach(type => $('#dropzone').addEventListener(type, event => { event.preventDefault(); $('#dropzone').classList.remove('dragging'); }));
$('#dropzone').addEventListener('drop', event => upload(event.dataTransfer.files[0]));

function addMessage(role, content, mode = '') {
  const node = document.createElement('div');
  node.className = `message ${role}`;
  node.innerHTML = `<div class="message-label">${role === 'user' ? '你' : 'LATTICE'}</div><div class="bubble">${escapeHtml(content)}</div>${mode ? `<div class="answer-mode">${mode === 'ollama' ? 'Ollama 本地生成' : '本地检索 · 抽取式回答'}</div>` : ''}`;
  $('#messages').appendChild(node);
  $('.conversation').scrollTop = $('.conversation').scrollHeight;
  return node;
}
function inspect(result) {
  $('#inspector').hidden = false;
  $('#source-count').textContent = result.sources.length;
  $('#sources-panel').innerHTML = result.sources.length ? result.sources.map((source, index) => `<article class="source-card"><div class="source-top"><span>来源 [${index + 1}]</span><span>${Math.round(source.score * 100)}%</span></div><div class="source-name">${escapeHtml(source.documentName)} · 片段 ${source.index + 1}</div><div class="source-text">${escapeHtml(source.text)}</div></article>`).join('') : '<p class="muted">暂无可引用来源。</p>';
  $('#trace-panel').innerHTML = result.trace.map(item => `<div class="trace-item"><span class="trace-step">${item.step}</span><strong>${escapeHtml(item.title)}</strong><p>${escapeHtml(item.detail)}</p></div>`).join('');
}
$('#query-form').addEventListener('submit', async event => {
  event.preventDefault();
  const question = $('#question').value.trim();
  if (!question || state.busy) return;
  state.busy = true;
  $('#welcome').hidden = true;
  $('#conversation').hidden = false;
  addMessage('user', question);
  $('#question').value = '';
  $('#question').style.height = 'auto';
  $('#send').disabled = true;
  const thinking = document.createElement('div');
  thinking.className = 'thinking';
  thinking.innerHTML = '<i></i><i></i><i></i>';
  $('#messages').appendChild(thinking);
  try {
    const result = await api('/api/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question, topK: 5, useModel: true }) });
    thinking.remove();
    addMessage('assistant', result.answer, result.mode);
    inspect(result);
  } catch (error) { thinking.remove(); addMessage('assistant', `发生错误：${error.message}`); }
  finally { state.busy = false; $('#send').disabled = false; $('#question').focus(); }
});
$('#question').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('#query-form').requestSubmit(); }
});
$('#question').addEventListener('input', event => { event.target.style.height = 'auto'; event.target.style.height = `${Math.min(event.target.scrollHeight, 130)}px`; });
$('#new-chat').addEventListener('click', () => { $('#messages').innerHTML = ''; $('#conversation').hidden = true; $('#welcome').hidden = false; $('#inspector').hidden = true; });
document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(item => item.classList.toggle('active', item === tab));
  $('#sources-panel').hidden = tab.dataset.tab !== 'sources';
  $('#trace-panel').hidden = tab.dataset.tab !== 'trace';
}));

async function openSettings() {
  const settings = await api('/api/settings');
  $('#ollama-url').value = settings.ollamaUrl;
  $('#model-name').value = settings.model;
  $('#settings-dialog').showModal();
}
$('#settings-open').addEventListener('click', openSettings);
['#settings-close', '#settings-cancel'].forEach(id => $(id).addEventListener('click', () => $('#settings-dialog').close()));
$('#settings-form').addEventListener('submit', async event => {
  event.preventDefault();
  await api('/api/settings', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ollamaUrl: $('#ollama-url').value, model: $('#model-name').value }) });
  $('#settings-dialog').close();
  toast('模型设置已保存');
});
loadDocuments().catch(error => toast(error.message));
