"""Comprehensive regression tests for frontend audit fixes (FE-01 through FE-17).

Executes real JavaScript from app/static/*.js and inline <script> blocks in
app/static/*.html inside Node.js vm sandboxes with a lightweight DOM/fetch harness,
verifying API contracts, XSS escaping, SSE streaming, async race conditions,
job pollers, and UI state transitions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import textwrap

from bs4 import BeautifulSoup
import pytest


ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "app" / "static"
NODE_BIN = shutil.which("node") or r"D:\Anaconda3\tools\nodejs\node.exe"


def _require_node() -> str:
    if not NODE_BIN or not Path(NODE_BIN).exists():
        pytest.skip("Node.js runtime not available")
    return NODE_BIN


def _extract_inline_scripts(html_filename: str) -> str:
    html = (STATIC_DIR / html_filename).read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    scripts = [s.get_text() for s in soup.find_all("script") if not s.get("src")]
    return "\n;\n".join(scripts)


# Lightweight DOM + browser + timer-tracking harness executed inside Node.js.
NODE_DOM_HARNESS = r"""
const fs = require('fs');
const path = require('path');
const vm = require('vm');

class MockClassList {
  constructor(el) { this.el = el; }
  _list() { return (this.el.className || '').split(/\s+/).filter(Boolean); }
  _save(arr) { this.el.className = Array.from(new Set(arr)).join(' '); }
  add(...cls) { this._save([...this._list(), ...cls.filter(Boolean)]); }
  remove(...cls) { this._save(this._list().filter(c => !cls.includes(c))); }
  contains(c) { return this._list().includes(c); }
  toggle(c) {
    if (this.contains(c)) { this.remove(c); return false; }
    this.add(c); return true;
  }
}

function parseAttributes(attrStr) {
  const attrs = {};
  const re = /([a-zA-Z0-9_:-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;
  let m;
  while ((m = re.exec(attrStr || '')) !== null) {
    const k = m[1].toLowerCase();
    const v = m[2] !== undefined ? m[2] : (m[3] !== undefined ? m[3] : (m[4] !== undefined ? m[4] : ''));
    attrs[k] = v;
  }
  return attrs;
}

function decodeBasicEntities(str) {
  return String(str)
    .replace(/&quot;/g, '"')
    .replace(/&#039;|&#39;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&');
}

class MockElement {
  constructor(tagName, ownerDoc) {
    this.tagName = String(tagName || 'div').toUpperCase();
    this.ownerDocument = ownerDoc;
    this.attributes = {};
    this.dataset = {};
    this.style = {};
    this.children = [];
    this.parentNode = null;
    this._innerHTML = '';
    this._text = '';
    this.value = '';
    this.checked = false;
    this.disabled = false;
    this.readOnly = false;
    this.scrollTop = 0;
    this.scrollHeight = 100;
    this.listeners = {};
    this.onclick = null;
    this.classList = new MockClassList(this);
  }
  get id() { return this.attributes.id || ''; }
  set id(v) {
    const val = String(v || '');
    this.attributes.id = val;
    if (val && this.ownerDocument) this.ownerDocument._byId.set(val, this);
  }
  get className() { return this.attributes.class || ''; }
  set className(v) { this.attributes.class = String(v || ''); }
  get innerText() { return this._text; }
  set innerText(v) {
    this._text = String(v ?? '');
    this._innerHTML = this._text;
    this.children = [];
  }
  get textContent() { return this._text; }
  set textContent(v) {
    this._text = String(v ?? '');
    this._innerHTML = this._text;
    this.children = [];
  }
  get innerHTML() {
    if (this.children.length > 0 && this._dirtyChildren) {
      return this.children.map(c => c._serializeOuter()).join('');
    }
    return this._innerHTML;
  }
  _serializeOuter() {
    const tag = this.tagName.toLowerCase();
    const escAttr = (v) => String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    const attrs = Object.entries(this.attributes).map(([k, v]) => ` ${k}="${escAttr(v)}"`).join('');
    return `<${tag}${attrs}>${this.innerHTML}</${tag}>`;
  }
  set innerHTML(html) {
    this._innerHTML = String(html ?? '');
    this._text = this._innerHTML.replace(/<[^>]+>/g, '');
    this._dirtyChildren = false;
    this.children = [];
    this._parseChildrenFromHtml(this._innerHTML);
    this._markAncestorsDirty();
  }
  _markAncestorsDirty() {
    let p = this.parentNode;
    while (p) {
      p._dirtyChildren = true;
      p = p.parentNode;
    }
  }
  _parseChildrenFromHtml(html) {
    const voidTags = new Set(['input', 'br', 'hr', 'img', 'meta', 'link', 'line', 'polygon', 'path', 'rect']);
    const stack = [this];
    const tokenRe = /<(\/?)([a-zA-Z][a-zA-Z0-9-]*)([^>]*?)(\/?)>/g;
    let m;
    let lastIndex = 0;
    while ((m = tokenRe.exec(html)) !== null) {
      if (m.index > lastIndex) {
        const chunk = html.slice(lastIndex, m.index);
        const cur = stack[stack.length - 1];
        if (cur !== this) {
          cur._innerHTML += chunk;
          cur._text += decodeBasicEntities(chunk);
        }
      }
      lastIndex = tokenRe.lastIndex;
      const isClose = m[1] === '/';
      const tag = m[2].toLowerCase();
      const attrStr = m[3] || '';
      const selfClose = m[4] === '/' || voidTags.has(tag);
      if (isClose) {
        for (let i = stack.length - 1; i >= 1; i--) {
          if (stack[i].tagName.toLowerCase() === tag) {
            const closed = stack[i];
            stack.length = i;
            const parent = stack[stack.length - 1];
            if (parent !== this) {
              parent._innerHTML += `<${tag}>${closed._innerHTML}</${tag}>`;
              parent._text += closed._text;
            }
            break;
          }
        }
        continue;
      }
      const el = new MockElement(tag, this.ownerDocument);
      const attrs = parseAttributes(attrStr);
      for (const [k, v] of Object.entries(attrs)) {
        el.setAttribute(k, decodeBasicEntities(v));
      }
      stack[stack.length - 1].appendChild(el, true);
      if (!selfClose) stack.push(el);
    }
  }
  setAttribute(k, v) {
    const key = String(k).toLowerCase();
    const val = String(v ?? '');
    this.attributes[key] = val;
    if (key === 'id' && val && this.ownerDocument) {
      this.ownerDocument._byId.set(val, this);
    }
    if (key === 'disabled') this.disabled = true;
    if (key === 'readonly') this.readOnly = true;
    if (key === 'checked') this.checked = true;
    if (key === 'value') this.value = val;
    if (key.startsWith('data-')) {
      const camel = key.slice(5).replace(/-([a-z])/g, (_, ch) => ch.toUpperCase());
      this.dataset[camel] = val;
    }
  }
  getAttribute(k) {
    const key = String(k).toLowerCase();
    return Object.prototype.hasOwnProperty.call(this.attributes, key) ? this.attributes[key] : null;
  }
  appendChild(child, _fromParse = false) {
    if (child.parentNode) {
      const idx = child.parentNode.children.indexOf(child);
      if (idx >= 0) child.parentNode.children.splice(idx, 1);
    }
    child.parentNode = this;
    this.children.push(child);
    if (!_fromParse) {
      this._dirtyChildren = true;
      this._markAncestorsDirty();
    }
    if (child.id && this.ownerDocument) this.ownerDocument._byId.set(child.id, child);
    return child;
  }
  insertBefore(newNode, refNode) {
    newNode.parentNode = this;
    const idx = refNode ? this.children.indexOf(refNode) : -1;
    if (idx >= 0) this.children.splice(idx, 0, newNode);
    else this.children.push(newNode);
    this._dirtyChildren = true;
    this._markAncestorsDirty();
    if (newNode.id && this.ownerDocument) this.ownerDocument._byId.set(newNode.id, newNode);
    return newNode;
  }
  get firstChild() { return this.children[0] || null; }
  get nextSibling() {
    if (!this.parentNode) return null;
    const i = this.parentNode.children.indexOf(this);
    return i >= 0 ? (this.parentNode.children[i + 1] || null) : null;
  }
  get nextElementSibling() { return this.nextSibling; }
  addEventListener(evt, fn) {
    if (!this.listeners[evt]) this.listeners[evt] = [];
    this.listeners[evt].push(fn);
  }
  click() {
    if (this.disabled) return;
    if (typeof this.onclick === 'function') this.onclick({ target: this, stopPropagation() {}, preventDefault() {} });
    for (const fn of (this.listeners.click || [])) {
      fn({ target: this, stopPropagation() {}, preventDefault() {} });
    }
  }
  focus() {}
  reset() {
    for (const el of this.querySelectorAll('input,textarea')) {
      el.value = '';
    }
  }
  _matches(sel) {
    sel = sel.trim();
    if (!sel) return false;
    if (sel.includes(',')) return sel.split(',').some(s => this._matches(s));
    if (sel.includes(' ')) {
      const parts = sel.split(/\s+/);
      const last = parts[parts.length - 1];
      if (!this._matches(last)) return false;
      let cur = this.parentNode;
      let pIdx = parts.length - 2;
      while (cur && pIdx >= 0) {
        if (cur._matches(parts[pIdx])) pIdx--;
        cur = cur.parentNode;
      }
      return pIdx < 0;
    }
    const attrMatch = sel.match(/^([a-zA-Z0-9-]*)?\[([a-zA-Z0-9_-]+)(?:=["']?([^"'\]]+)["']?)?\]$/);
    if (attrMatch) {
      if (attrMatch[1] && this.tagName.toLowerCase() !== attrMatch[1].toLowerCase()) return false;
      const val = this.getAttribute(attrMatch[2]);
      if (val === null) return false;
      return attrMatch[3] === undefined ? true : val === attrMatch[3];
    }
    if (sel.startsWith('#')) return this.id === sel.slice(1);
    if (sel.startsWith('.')) {
      const classes = sel.slice(1).split('.');
      return classes.every(c => this.classList.contains(c));
    }
    const tagAndClass = sel.match(/^([a-zA-Z0-9-]+)\.(.+)$/);
    if (tagAndClass) {
      return this.tagName.toLowerCase() === tagAndClass[1].toLowerCase()
        && tagAndClass[2].split('.').every(c => this.classList.contains(c));
    }
    return this.tagName.toLowerCase() === sel.toLowerCase();
  }
  _walk(out) {
    for (const c of this.children) {
      out.push(c);
      c._walk(out);
    }
  }
  querySelectorAll(sel) {
    const all = [];
    this._walk(all);
    return all.filter(el => el._matches(sel));
  }
  querySelector(sel) {
    return this.querySelectorAll(sel)[0] || null;
  }
  closest(sel) {
    let cur = this;
    while (cur) {
      if (cur._matches(sel)) return cur;
      cur = cur.parentNode;
    }
    return null;
  }
}

class MockDocument {
  constructor(html) {
    this._byId = new Map();
    this.listeners = {};
    this.readyState = 'complete';
    this.body = new MockElement('body', this);
    if (html) {
      const bodyMatch = html.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
      const bodyHtml = bodyMatch ? bodyMatch[1] : html;
      // Strip script tags before parsing DOM
      const cleanHtml = bodyHtml.replace(/<script[\s\S]*?<\/script>/gi, '');
      this.body._parseChildrenFromHtml(cleanHtml);
    }
  }
  createElement(tag) { return new MockElement(tag, this); }
  getElementById(id) {
    if (this._byId.has(id)) return this._byId.get(id);
    return this.body.querySelector('#' + id);
  }
  querySelector(sel) { return this.body.querySelector(sel); }
  querySelectorAll(sel) { return this.body.querySelectorAll(sel); }
  addEventListener(evt, fn) {
    if (!this.listeners[evt]) this.listeners[evt] = [];
    this.listeners[evt].push(fn);
  }
}

function createSandbox({ htmlPath, jsPaths = [], pathname = '/', fetchImpl, confirmImpl, promptImpl }) {
  const html = htmlPath ? fs.readFileSync(htmlPath, 'utf8') : '';
  const document = new MockDocument(html);
  const alerts = [];
  const intervals = new Map();
  let nextIntervalId = 1;
  const timeouts = new Map();
  let nextTimeoutId = 1;

  const logs = [];
  const sandbox = {
    console: {
      log: (...a) => logs.push(['log', ...a]),
      warn: (...a) => logs.push(['warn', ...a]),
      error: (...a) => logs.push(['error', ...a]),
    },
    __logs: logs,
    __eval: (expr) => vm.runInContext(expr, sandbox),
    document,
    URLSearchParams,
    TextDecoder,
    TextEncoder,
    AbortController,
    encodeURIComponent,
    decodeURIComponent,
    Date,
    Math,
    Number,
    String,
    Boolean,
    Array,
    Object,
    Set,
    Map,
    RegExp,
    JSON,
    Error,
    TypeError,
    Promise,
    parseInt,
    parseFloat,
    isNaN,
    fetch: fetchImpl || (async () => ({ ok: true, status: 200, json: async () => ({}) })),
    alert: (m) => alerts.push(String(m)),
    confirm: confirmImpl || (() => true),
    prompt: promptImpl || (() => ''),
    setInterval: (fn, ms) => {
      const id = nextIntervalId++;
      intervals.set(id, { fn, ms });
      return id;
    },
    clearInterval: (id) => { intervals.delete(id); },
    setTimeout: (fn, ms) => {
      const id = nextTimeoutId++;
      const t = setTimeout(() => { timeouts.delete(id); fn(); }, Math.min(ms || 0, 5));
      timeouts.set(id, t);
      return id;
    },
    clearTimeout: (id) => {
      const t = timeouts.get(id);
      if (t) clearTimeout(t);
      timeouts.delete(id);
    },
    __alerts: alerts,
    __intervals: intervals,
    __tickIntervals: async () => {
      const entries = Array.from(intervals.entries());
      for (const [id, entry] of entries) {
        if (intervals.has(id)) {
          await entry.fn();
        }
      }
    },
  };
  sandbox.window = sandbox;
  sandbox.window.location = { pathname, reload() { sandbox.__reloaded = true; } };
  sandbox.window.addEventListener = (evt, fn) => {
    if (!sandbox.__winListeners) sandbox.__winListeners = {};
    if (!sandbox.__winListeners[evt]) sandbox.__winListeners[evt] = [];
    sandbox.__winListeners[evt].push(fn);
  };

  vm.createContext(sandbox);

  for (const jp of jsPaths) {
    const code = fs.readFileSync(jp, 'utf8');
    vm.runInContext(code, sandbox, { filename: jp });
  }
  if (html) {
    const scriptRe = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;
    let m;
    while ((m = scriptRe.exec(html)) !== null) {
      vm.runInContext(m[1], sandbox, { filename: htmlPath });
    }
  }
  return sandbox;
}

function makeSseResponse(chunks) {
  const enc = new TextEncoder();
  const encoded = chunks.map(c => (c instanceof Uint8Array ? c : enc.encode(c)));
  let idx = 0;
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    body: {
      getReader() {
        return {
          async read() {
            if (idx < encoded.length) {
              return { done: false, value: encoded[idx++] };
            }
            return { done: true, value: undefined };
          }
        };
      }
    }
  };
}
"""


def _run_node(test_js: str) -> dict:
    node_bin = _require_node()
    script = (
        NODE_DOM_HARNESS
        + "\nconst STATIC_DIR = "
        + json.dumps(str(STATIC_DIR))
        + ";\n"
        + "(async () => {\n"
        + test_js
        + "\n})().then((res) => {\n"
        + "  process.stdout.write(JSON.stringify({ ok: true, result: res ?? null }));\n"
        + "  process.exit(0);\n"
        + "}).catch((err) => {\n"
        + "  process.stderr.write(String(err && err.stack ? err.stack : err));\n"
        + "  process.exit(1);\n"
        + "});\n"
    )
    proc = subprocess.run(
        [node_bin, "-e", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )
    if proc.returncode != 0:
        raise AssertionError(f"Node test script failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}")
    return json.loads(proc.stdout)["result"]


# ==============================================================================
# Syntax & HTML Validity Across All Static Files
# ==============================================================================

def test_all_static_js_and_inline_scripts_parse_cleanly():
    """Every .js file and every inline <script> in app/static/*.html must be syntactically valid JS."""
    html_files = sorted(STATIC_DIR.glob("*.html"))
    js_files = sorted(STATIC_DIR.glob("*.js"))
    assert len(html_files) >= 11
    assert len(js_files) >= 2

    for hf in html_files:
        soup = BeautifulSoup(hf.read_text(encoding="utf-8"), "html.parser")
        assert soup.find("body") is not None, f"{hf.name} missing <body>"

    res = _run_node(r"""
      const checked = [];
      for (const f of fs.readdirSync(STATIC_DIR).sort()) {
        const full = path.join(STATIC_DIR, f);
        const content = fs.readFileSync(full, 'utf8');
        if (f.endsWith('.js')) {
          new vm.Script(content, { filename: f });
          checked.push(f);
        } else if (f.endsWith('.html')) {
          const re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;
          let m, idx = 0;
          while ((m = re.exec(content)) !== null) {
            new vm.Script(m[1], { filename: `${f}#script${++idx}` });
            checked.push(`${f}#script${idx}`);
          }
        }
      }
      return { count: checked.length, checked };
    """)
    assert res["count"] >= 13


# ==============================================================================
# FE-01: review_queue.html full API contract against review_queue_routes.py
# ==============================================================================

def test_fe01_review_queue_api_contract_and_detail_lazy_load():
    html = (STATIC_DIR / "review_queue.html").read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    tab_statuses = [el.get("data-status") for el in soup.select("#status-tabs .tab")]
    assert "待审" in tab_statuses, f"Filter tabs must send backend ReviewStatus enum '待审', got {tab_statuses}"
    assert "通过" in tab_statuses, f"Filter tabs must send backend ReviewStatus enum '通过', got {tab_statuses}"
    assert "驳回" in tab_statuses, f"Filter tabs must send backend ReviewStatus enum '驳回', got {tab_statuses}"

    res = _run_node(r"""
      const calls = [];
      const sb = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'review_queue.html'),
        pathname: '/review-queue',
        fetchImpl: async (url, opts) => {
          calls.push({ url, opts });
          if (url.startsWith('/api/review-queue/101/reject')) {
            return { ok: true, status: 200, json: async () => ({ status: 'rejected', id: 101 }) };
          }
          if (url === '/api/review-queue/101') {
            return {
              ok: true, status: 200,
              json: async () => ({
                id: 101,
                normalized_question: '如何开发票',
                ai_suggested_answer: '订单完成后可申请电子发票',
                occurrence_count: 5,
                review_status: '待审',
                raw_queries: [
                  {
                    id: 1,
                    raw_question: '发票咋开啊',
                    created_at: '2026-11-01T10:00:00',
                    retrieved_chunks: [
                      { text: '电子发票规则', score: 0.42 },
                      { chunk: '旧字段兜底片段', score: null }
                    ]
                  }
                ]
              })
            };
          }
          if (url.startsWith('/api/review-queue')) {
            return {
              ok: true, status: 200,
              json: async () => ({
                total: 2,
                items: [
                  {
                    id: 101,
                    normalized_question: '如何开发票',
                    ai_suggested_answer: '订单完成后可申请电子发票',
                    occurrence_count: 5,
                    review_status: '待审',
                    approved_answer: null,
                    created_at: '2026-11-01T10:00:00',
                    raw_questions_count: 1
                  },
                  {
                    id: 102,
                    normalized_question: '运费怎么算',
                    ai_suggested_answer: '满99包邮',
                    occurrence_count: 2,
                    review_status: '通过',
                    approved_answer: '满99包邮',
                    created_at: '2026-11-01T11:00:00',
                    raw_questions_count: 1
                  }
                ]
              })
            };
          }
          return { ok: false, status: 404, json: async () => ({ detail: 'not found' }) };
        },
        promptImpl: () => '重复问题',
      });

      await sb.loadQueue();
      const tbodyHtml = sb.document.getElementById('queue-tbody').innerHTML;

      // Open approve modal with numeric ID
      sb.openApproveModal(101);
      const modalOpen = sb.document.getElementById('approve-modal').classList.contains('active');
      const modalQ = sb.document.getElementById('approve-question').value;

      // Lazy-load detail row
      await sb.toggleDetail(101);
      const detailHtml = sb.document.getElementById('detail-101').innerHTML;

      // Reject sends { reason }
      await sb.rejectItem(101);
      const rejectCall = calls.find(c => c.url === '/api/review-queue/101/reject');

      return {
        tbodyHtml,
        modalOpen,
        modalQ,
        detailHtml,
        fetchedDetail: calls.some(c => c.url === '/api/review-queue/101'),
        rejectBody: rejectCall ? JSON.parse(rejectCall.opts.body) : null,
      };
    """)

    assert "#101" in res["tbodyHtml"]
    assert "待审核" in res["tbodyHtml"]
    assert "已核准" in res["tbodyHtml"]
    assert "openApproveModal(101)" in res["tbodyHtml"]
    assert res["modalOpen"] is True
    assert res["modalQ"] == "如何开发票"
    assert res["fetchedDetail"] is True
    assert "发票咋开啊" in res["detailHtml"]
    assert "电子发票规则" in res["detailHtml"]
    assert "0.4200" in res["detailHtml"]
    assert "旧字段兜底片段" in res["detailHtml"]
    assert "Invalid Date" not in res["detailHtml"]
    assert res["rejectBody"] == {"reason": "重复问题"}


# ==============================================================================
# FE-02: index.html handleFeedback ReferenceError & res.ok guard
# ==============================================================================

def test_fe02_index_feedback_no_reference_error_and_checks_res_ok():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "lastUserMessageText" not in html, "Undeclared global lastUserMessageText must be removed"

    res = _run_node(r"""
      const feedbackCalls = [];
      let feedbackOk = false;
      const sb = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'index.html'),
        fetchImpl: async (url, opts) => {
          if (url === '/api/conversations') {
            return { ok: true, status: 200, json: async () => [] };
          }
          if (url === '/api/chat/stream') {
            return makeSseResponse([
              'data: {"event_type":"text","conversation_id":77,"content":"七天无理由退货"}\n\n',
              'data: [DONE]\n\n',
            ]);
          }
          if (url === '/api/chat/feedback') {
            feedbackCalls.push(JSON.parse(opts.body));
            return { ok: feedbackOk, status: feedbackOk ? 200 : 500, json: async () => ({ status: 'success' }) };
          }
          return { ok: true, status: 200, json: async () => ({}) };
        }
      });

      sb.document.getElementById('user-input').value = '退货政策是什么';
      await sb.sendMessage();
      // Wait for token streamer to finish and attach .feedback-bar
      await new Promise(r => setTimeout(r, 120));

      const downBtn = sb.document.querySelector('.btn-fb-thumb.down');
      if (!downBtn) throw new Error('feedback down button was not rendered');

      // 1. Failing feedback response should NOT permanently lock buttons
      feedbackOk = false;
      await sb.handleFeedback(downBtn, 'down');
      const disabledAfterFail = downBtn.disabled;

      // 2. Successful feedback response sends query and locks buttons
      feedbackOk = true;
      await sb.handleFeedback(downBtn, 'down');
      const disabledAfterOk = downBtn.disabled;

      return {
        disabledAfterFail,
        disabledAfterOk,
        feedbackCalls,
      };
    """)

    assert res["disabledAfterFail"] is False, "Buttons must not lock when /api/chat/feedback fails"
    assert res["disabledAfterOk"] is True
    assert len(res["feedbackCalls"]) == 2
    assert res["feedbackCalls"][1]["conversation_id"] == 77
    assert res["feedbackCalls"][1]["feedback_type"] == "down"
    assert res["feedbackCalls"][1]["query"] == "退货政策是什么"


# ==============================================================================
# FE-03: Stored/DOM XSS Escaping Across All 7 Frontend Files
# ==============================================================================

def test_fe03_xss_escaping_across_all_seven_files():
    res = _run_node(r"""
      const payload = '<img src=x onerror=alert(1)>';
      const quotePayload = "1001');alert(1);//";
      const results = {};

      // 1. rag_eval.html
      {
        const sb = createSandbox({ htmlPath: path.join(STATIC_DIR, 'rag_eval.html') });
        sb.renderCasesList([{
          id: 1,
          eval_id: payload,
          bucket: payload,
          strategy: payload,
          judge_model: payload,
          query: payload,
          answer: payload,
          reason: payload,
          resolution: payload,
          resolved_at: payload,
          status: '未解决',
          seen_count: 1,
          cited_indices: [1],
          citations: [{ n: 1, section_path: payload, question: payload, answer: payload }],
        }]);
        const casesHtml = sb.document.getElementById('cases-container').innerHTML;
        sb.renderCasesList([], payload);
        const emptyHtml = sb.document.getElementById('cases-container').innerHTML;
        results.rag_eval = {
          hasRawImg: casesHtml.includes('<img') || emptyHtml.includes('<img'),
          hasEscaped: casesHtml.includes('&lt;img') && emptyHtml.includes('&lt;img'),
        };
      }

      // 2. review_queue.html
      {
        const sb = createSandbox({
          htmlPath: path.join(STATIC_DIR, 'review_queue.html'),
          fetchImpl: async (url) => {
            if (url === '/api/review-queue/5') {
              return {
                ok: true, status: 200,
                json: async () => ({
                  id: 5,
                  normalized_question: payload,
                  ai_suggested_answer: payload,
                  occurrence_count: 1,
                  review_status: '待审',
                  raw_queries: [{
                    id: 9,
                    raw_question: payload,
                    created_at: '2026-11-01T10:00:00',
                    retrieved_chunks: [{ text: payload, score: 0.2 }]
                  }]
                })
              };
            }
            return {
              ok: true, status: 200,
              json: async () => ({
                total: 1,
                items: [{
                  id: 5,
                  normalized_question: payload,
                  ai_suggested_answer: payload,
                  occurrence_count: 1,
                  review_status: '待审',
                  created_at: '2026-11-01T10:00:00',
                }]
              })
            };
          }
        });
        await sb.loadQueue();
        await sb.toggleDetail(5);
        const html = sb.document.getElementById('queue-tbody').innerHTML;
        results.review_queue = {
          hasRawImg: html.includes('<img'),
          hasEscaped: html.includes('&lt;img'),
        };
      }

      // 3. acceptance_errors.html
      {
        const sb = createSandbox({
          htmlPath: path.join(STATIC_DIR, 'acceptance_errors.html'),
          jsPaths: [path.join(STATIC_DIR, 'admin.js'), path.join(STATIC_DIR, 'acceptance.js')],
          fetchImpl: async () => ({
            ok: true, status: 200,
            json: async () => ({
              present: true,
              error_accounting: { under_tag_count: 1, over_tag_count: 0, displaced_count: 0, total_penalty: 1, confusion_matrix_sum: 1, balanced: true },
              total_error_samples: 1,
              error_samples: [{
                text: payload,
                true_labels: [payload],
                pred_labels: [payload],
                under_tag_count: 1,
                over_tag_count: 0,
                displaced_count: 0,
                boundary_note: payload,
              }]
            })
          })
        });
        await sb.loadErrors();
        const html = sb.document.getElementById('error-samples-tbody').innerHTML;
        results.acceptance_errors = {
          hasRawImg: html.includes('<img'),
          hasEscaped: html.includes('&lt;img'),
        };
      }

      // 4. kb.html
      {
        const sb = createSandbox({
          htmlPath: path.join(STATIC_DIR, 'kb.html'),
          fetchImpl: async (url) => {
            if (url === '/api/kb/materials') {
              return { ok: true, status: 200, json: async () => [{ filename: payload, file_path: payload, file_size_bytes: 100, modified_time: '2026', estimated_chunks: 1 }] };
            }
            if (url.startsWith('/api/kb/chunks')) {
              return { ok: true, status: 200, json: async () => ({ total: 1, page: 1, page_size: 10, items: [{ id: 1, category: payload, questions: payload, answer: payload, section_path: payload, vectorize_status: 'done', created_at: '2026' }] }) };
            }
            if (url === '/api/kb/documents/preview') {
              return { ok: true, status: 200, json: async () => ({ total_chunks: 1, chunks: [{ id: 1, category: payload, section_path: payload, questions: payload, answer: payload, content_type: payload, is_key_clause: false }] }) };
            }
            if (url === '/api/kb/search') {
              return { ok: false, status: 500, json: async () => ({ detail: payload }) };
            }
            return { ok: true, status: 200, json: async () => ({}) };
          }
        });
        await sb.loadMaterials();
        await sb.loadChunks();
        sb.document.getElementById('doc-content').value = '# test';
        await sb.previewDocChunks();
        sb.document.getElementById('search-query').value = 'test';
        await sb.executeSearch();
        const combined = [
          sb.document.getElementById('materials-tbody').innerHTML,
          sb.document.getElementById('chunks-tbody').innerHTML,
          sb.document.getElementById('preview-tbody').innerHTML,
          sb.document.getElementById('search-results').innerHTML,
        ].join('\n');
        results.kb = {
          hasRawImg: combined.includes('<img'),
          hasEscaped: combined.includes('&lt;img'),
        };
      }

      // 5. observability.html
      {
        const sb = createSandbox({
          htmlPath: path.join(STATIC_DIR, 'observability.html'),
          fetchImpl: async () => ({
            ok: true, status: 200,
            json: async () => ({
              cost_block: { present: true, data: { total_requests: 1, total_tokens: 10, total_cost_usd: 0.01, items: [{ intent: payload, total_calls: 1, prompt_tokens: 5, completion_tokens: 5, total_tokens: 10, estimated_cost_usd: 0.01, cost_percentage: 50 }] } },
              eval_trend_block: { present: true, runs: [{ id: 1, triggered_by: payload, dataset_size: 10, metrics: { mrr: 0.9 }, deltas: { mrr: 0 }, created_at: payload }] },
              calibration_block: { present: false, data: null }
            })
          })
        });
        await sb.loadOverview();
        const combined = sb.document.getElementById('cost-tbody').innerHTML + sb.document.getElementById('eval-tbody').innerHTML;
        results.observability = {
          hasRawImg: combined.includes('<img'),
          hasEscaped: combined.includes('&lt;img'),
        };
      }

      // 6. acceptance.js
      {
        const sb = createSandbox({
          htmlPath: path.join(STATIC_DIR, 'acceptance_eval.html'),
          jsPaths: [path.join(STATIC_DIR, 'admin.js'), path.join(STATIC_DIR, 'acceptance.js')],
          fetchImpl: async (url) => {
            if (url === '/api/acceptance/classify') {
              return { ok: true, status: 200, json: async () => ({ labels: [payload], scores: { [payload]: 0.9 } }) };
            }
            return { ok: false, status: 400, json: async () => ({ detail: 'bad job' }) };
          }
        });
        await sb.runAcceptanceJob(payload);
        const modalTitleHtml = sb.document.getElementById('job-modal-title').innerHTML;
        sb.initClassifyDemo('interactive-demo-container', '"><img src=x onerror=alert(1)>', { autoRun: false });
        sb.document.getElementById('demo-input-text').value = 'test';
        sb.document.getElementById('demo-classify-btn').click();
        await new Promise(r => setTimeout(r, 20));
        const demoHtml = sb.document.getElementById('interactive-demo-container').innerHTML;
        const tagsHtml = sb.document.getElementById('demo-hit-tags').innerHTML;
        const barsHtml = sb.document.getElementById('demo-bars-container').innerHTML;
        const combined = modalTitleHtml + demoHtml + tagsHtml + barsHtml;
        results.acceptance_js = {
          hasRawImg: combined.includes('<img'),
          hasEscaped: combined.includes('&lt;img'),
        };
      }

      // 7. index.html order card inline onclick breakout
      {
        const sb = createSandbox({ htmlPath: path.join(STATIC_DIR, 'index.html') });
        const host = sb.document.createElement('div');
        sb.renderOrderSelectorCards(host, [{
          order_id: quotePayload,
          product_name: payload,
          status: '已发货',
          amount: '100元',
          order_time: '2026-09-01',
        }]);
        const card = host.querySelector('.order-card-item');
        const onclickAttr = card ? card.getAttribute('onclick') : '';
        results.index_order_card = {
          hasRawImg: host.innerHTML.includes('<img'),
          interpolatedInOnclick: onclickAttr.includes('alert(1)') || onclickAttr.includes('1001'),
          datasetOrderId: card ? card.dataset.orderId : null,
        };
      }

      return results;
    """)

    for key in ["rag_eval", "review_queue", "acceptance_errors", "kb", "observability", "acceptance_js"]:
        assert res[key]["hasRawImg"] is False, f"{key} rendered unescaped <img tag"
        assert res[key]["hasEscaped"] is True, f"{key} did not escape HTML payload"

    assert res["index_order_card"]["hasRawImg"] is False
    assert res["index_order_card"]["interpolatedInOnclick"] is False, "order_id must not be interpolated inside onclick JS string"
    assert res["index_order_card"]["datasetOrderId"] == "1001');alert(1);//"


# ==============================================================================
# FE-04: index.html SSE error preservation, resume error, ticket_preview, flush
# ==============================================================================

def test_fe04_index_sse_error_preserve_resume_error_and_ticket_preview():
    res = _run_node(r"""
      // Case A: stream emits error + [DONE]
      let sessionLoadsA = 0;
      const sbA = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'index.html'),
        fetchImpl: async (url) => {
          if (url === '/api/conversations') {
            sessionLoadsA++;
            return { ok: true, status: 200, json: async () => [] };
          }
          if (url === '/api/chat/stream') {
            return makeSseResponse([
              'data: {"event_type":"error","conversation_id":10,"error":"上游LLM超时"}\n\n',
              'data: [DONE]\n\n',
            ]);
          }
          return { ok: true, status: 200, json: async () => ({}) };
        }
      });
      sbA.document.getElementById('user-input').value = '查物流';
      await sbA.sendMessage();
      await new Promise(r => setTimeout(r, 40));
      const rowsA = sbA.document.querySelectorAll('#messages-container .msg-row.bot');
      const lastBotA = rowsA[rowsA.length - 1];
      const textBodyA = lastBotA.querySelector('.text-body').innerHTML;
      const hasFeedbackA = !!lastBotA.querySelector('.feedback-bar');

      // Case B: ticket_preview only stream (no text event)
      const sbB = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'index.html'),
        fetchImpl: async (url) => {
          if (url === '/api/conversations') return { ok: true, status: 200, json: async () => [] };
          if (url === '/api/chat/stream') {
            return makeSseResponse([
              'data: {"event_type":"ticket_preview","conversation_id":11,"ticket_type":"售后","description":"屏幕破损"}\n\n',
              'data: [DONE]\n\n',
            ]);
          }
          return { ok: true, status: 200, json: async () => ({}) };
        }
      });
      sbB.document.getElementById('user-input').value = '帮我建工单';
      await sbB.sendMessage();
      await new Promise(r => setTimeout(r, 40));
      const rowsB = sbB.document.querySelectorAll('#messages-container .msg-row.bot');
      const lastBotB = rowsB[rowsB.length - 1];
      const hasTicketCardB = !!lastBotB.querySelector('.ticket-preview-card');
      const textBodyB = lastBotB.querySelector('.text-body').innerHTML;

      // Case C: resumeChatWorkflow receives error event -> shows error and re-enables card buttons
      const sbC = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'index.html'),
        fetchImpl: async (url) => {
          if (url === '/api/conversations') return { ok: true, status: 200, json: async () => [] };
          if (url === '/api/chat/resume') {
            return makeSseResponse([
              'data: {"event_type":"error","conversation_id":11,"error":"工单系统异常"}\n\n',
              'data: [DONE]\n\n',
            ]);
          }
          return { ok: true, status: 200, json: async () => ({}) };
        }
      });
      const hostC = sbC.document.createElement('div');
      sbC.document.getElementById('messages-container').appendChild(hostC);
      sbC.renderTicketPreviewCard(hostC, { conversation_id: 11, ticket_type: '售后', description: '破损' });
      const confirmBtn = hostC.querySelector('.btn-ticket-confirm');
      await sbC.handleTicketDecision(confirmBtn, 11, 'confirm');
      await new Promise(r => setTimeout(r, 40));
      const rowsC = sbC.document.querySelectorAll('#messages-container .msg-row.bot');
      const lastBotC = rowsC[rowsC.length - 1];
      const resumeTextC = lastBotC.querySelector('.text-body').innerHTML;
      const confirmDisabledAfterErr = confirmBtn.disabled;

      // Case D: UTF-8 multibyte split + trailing frame without \n\n
      const enc = new TextEncoder();
      const fullFrame = enc.encode('data: {"event_type":"text","conversation_id":12,"content":"你好世界"}');
      const mid = fullFrame.length - 4; // splits inside multibyte Chinese char
      const sbD = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'index.html'),
        fetchImpl: async (url) => {
          if (url === '/api/conversations') return { ok: true, status: 200, json: async () => [] };
          if (url === '/api/chat/stream') {
            return makeSseResponse([fullFrame.slice(0, mid), fullFrame.slice(mid)]);
          }
          return { ok: true, status: 200, json: async () => ({}) };
        }
      });
      sbD.document.getElementById('user-input').value = 'hi';
      await sbD.sendMessage();
      await new Promise(r => setTimeout(r, 80));
      const rowsD = sbD.document.querySelectorAll('#messages-container .msg-row.bot');
      const textBodyD = rowsD[rowsD.length - 1].querySelector('.text-body').innerHTML;

      return {
        textBodyA, hasFeedbackA, sessionLoadsA,
        hasTicketCardB, textBodyB,
        resumeTextC, confirmDisabledAfterErr,
        textBodyD,
      };
    """)

    assert "上游LLM超时" in res["textBodyA"]
    assert "抱歉，未能获取有效回复" not in res["textBodyA"]
    assert res["hasFeedbackA"] is False
    # At most 1 loadSessions on new conv id + 1 on finishStreaming (not 3 from double finishStreaming)
    assert res["sessionLoadsA"] <= 2

    assert res["hasTicketCardB"] is True
    assert "抱歉，未能获取有效回复" not in res["textBodyB"]
    assert "typing-indicator" not in res["textBodyB"]

    assert "工单系统异常" in res["resumeTextC"]
    assert "已处理完成" not in res["resumeTextC"]
    assert res["confirmDisabledAfterErr"] is False, "Ticket buttons should re-enable on resume error"

    assert "你好世界" in res["textBodyD"]


# ==============================================================================
# FE-05: acceptance_data.html integer leakage overlap counts
# ==============================================================================

def test_fe05_acceptance_data_integer_leakage_counts():
    res = _run_node(r"""
      let payload = {
        present: true,
        dataset: { train_count: 100, val_count: 20, test_count: 20, train_size_bytes: 1024, val_size_bytes: 512, test_size_bytes: 512 },
        leakage_check: { passed: false, train_val_overlap: 3, train_test_overlap: 0, val_test_overlap: 2 },
        training_artifacts: { files: [] },
        onnx_artifacts: { files: [] },
      };
      const sb = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'acceptance_data.html'),
        jsPaths: [path.join(STATIC_DIR, 'admin.js'), path.join(STATIC_DIR, 'acceptance.js')],
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => payload }),
      });

      await sb.loadData();
      const first = {
        tv: sb.document.getElementById('overlap-train-val').innerText,
        tt: sb.document.getElementById('overlap-train-test').innerText,
        vt: sb.document.getElementById('overlap-val-test').innerText,
        tvColor: sb.document.getElementById('overlap-train-val').style.color,
        ttColor: sb.document.getElementById('overlap-train-test').style.color,
      };

      payload = {
        ...payload,
        leakage_check: { passed: true, train_val_overlap: 0, train_test_overlap: 0, val_test_overlap: 0 },
      };
      await sb.loadData();
      const second = {
        tv: sb.document.getElementById('overlap-train-val').innerText,
        tvColor: sb.document.getElementById('overlap-train-val').style.color,
      };
      return { first, second };
    """)

    assert res["first"]["tv"] == "3"
    assert res["first"]["tt"] == "0"
    assert res["first"]["vt"] == "2"
    assert res["first"]["tvColor"] == "var(--danger-dark)"
    assert res["first"]["ttColor"] == "var(--success-dark)"
    assert res["second"]["tv"] == "0"
    assert res["second"]["tvColor"] == "var(--success-dark)"


# ==============================================================================
# FE-06: acceptance_errors.html real report schema filter counts & badges
# ==============================================================================

def test_fe06_acceptance_errors_real_report_schema_filters_and_badges():
    res = _run_node(r"""
      const payload = {
        present: true,
        error_accounting: {
          total_error_samples: 3,
          under_tag_count: 1,
          over_tag_count: 1,
          displaced_count: 1,
          total_penalty: 4,
          confusion_matrix_sum: 0,
          balanced: false,
        },
        total_error_samples: 3,
        error_samples: [
          { text: '问句1', true_labels: ['物流', '运费'], pred_labels: ['物流'], missed_labels: ['运费'], extra_labels: [], under_tag_count: 1, over_tag_count: 0, displaced_count: 0, boundary_note: '物流管货、运费管钱' },
          { text: '问句2', true_labels: ['优惠活动'], pred_labels: ['优惠活动', '价保'], missed_labels: [], extra_labels: ['价保'], under_tag_count: 0, over_tag_count: 1, displaced_count: 0, boundary_note: '—' },
          { text: '问句3', true_labels: ['退换货'], pred_labels: ['保修维修'], missed_labels: ['退换货'], extra_labels: ['保修维修'], under_tag_count: 0, over_tag_count: 0, displaced_count: 1, boundary_note: '修归保修维修、退归退换货' },
        ],
      };
      const sb = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'acceptance_errors.html'),
        jsPaths: [path.join(STATIC_DIR, 'admin.js'), path.join(STATIC_DIR, 'acceptance.js')],
        fetchImpl: async () => ({ ok: true, status: 200, json: async () => payload }),
      });

      await sb.loadErrors();
      const counts = {
        all: sb.document.getElementById('count-filter-all').innerText,
        under: sb.document.getElementById('count-filter-under').innerText,
        over: sb.document.getElementById('count-filter-over').innerText,
        displaced: sb.document.getElementById('count-filter-displaced').innerText,
      };
      const formulaHtml = sb.document.getElementById('formula-check-result').innerHTML;
      const allHtml = sb.document.getElementById('error-samples-tbody').innerHTML;

      sb.filterSamples('under');
      const underHtml = sb.document.getElementById('error-samples-tbody').innerHTML;
      sb.filterSamples('over');
      const overHtml = sb.document.getElementById('error-samples-tbody').innerHTML;
      sb.filterSamples('displaced');
      const displacedHtml = sb.document.getElementById('error-samples-tbody').innerHTML;

      return { counts, formulaHtml, allHtml, underHtml, overHtml, displacedHtml };
    """)

    assert res["counts"] == {"all": "3", "under": "1", "over": "1", "displaced": "1"}
    assert "混淆和 0" in res["formulaHtml"], f"Expected confusion_matrix_sum 0 preserved via ??, got: {res['formulaHtml']}"
    assert "漏打" in res["allHtml"] and "多打" in res["allHtml"] and "错位" in res["allHtml"]
    assert "问句1" in res["underHtml"] and "问句2" not in res["underHtml"]
    assert "问句2" in res["overHtml"] and "问句1" not in res["overHtml"]
    assert "问句3" in res["displacedHtml"] and "问句2" not in res["displacedHtml"]


# ==============================================================================
# FE-07: index.html async race conditions (createNewChat mid-stream & selectSession)
# ==============================================================================

def test_fe07_index_race_conditions_new_chat_and_select_session():
    res = _run_node(r"""
      // Part 1: createNewChat() during active SSE stream aborts and ignores late frames
      let capturedSignal = null;
      let releaseSecondChunk;
      const secondChunkReady = new Promise(r => { releaseSecondChunk = r; });

      const sb1 = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'index.html'),
        fetchImpl: async (url, opts) => {
          if (url === '/api/conversations') return { ok: true, status: 200, json: async () => [] };
          if (url === '/api/chat/stream') {
            capturedSignal = opts.signal;
            const enc = new TextEncoder();
            let step = 0;
            return {
              ok: true, status: 200, statusText: 'OK',
              body: {
                getReader() {
                  return {
                    async read() {
                      if (step === 0) {
                        step = 1;
                        return { done: false, value: enc.encode('data: {"event_type":"text","conversation_id":99,"content":"旧回复"}\n\n') };
                      }
                      if (step === 1) {
                        step = 2;
                        await secondChunkReady;
                        return { done: false, value: enc.encode('data: {"event_type":"text","conversation_id":99,"content":"滞后帧"}\n\n') };
                      }
                      return { done: true, value: undefined };
                    }
                  };
                }
              }
            };
          }
          return { ok: true, status: 200, json: async () => ({}) };
        }
      });

      sb1.document.getElementById('user-input').value = '旧问题';
      const sendPromise = sb1.sendMessage();
      await new Promise(r => setTimeout(r, 20));
      sb1.createNewChat();
      releaseSecondChunk();
      await sendPromise;
      await new Promise(r => setTimeout(r, 30));

      const abortedOnNewChat = !!(capturedSignal && capturedSignal.aborted);
      const convIdAfterNewChat = sb1.__eval('currentConversationId');
      const sessionBadgeAfterNewChat = sb1.document.getElementById('session-display').innerText;

      // Part 2: Overlapping selectSession(1) then selectSession(2) resolving out-of-order
      let resolveS1, resolveS2;
      const p1 = new Promise(r => { resolveS1 = r; });
      const p2 = new Promise(r => { resolveS2 = r; });
      const sb2 = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'index.html'),
        fetchImpl: async (url) => {
          if (url === '/api/conversations/1/messages') {
            await p1;
            return { ok: true, status: 200, json: async () => [{ id: 1, role: 'user', content: '会话1旧消息' }] };
          }
          if (url === '/api/conversations/2/messages') {
            await p2;
            return { ok: true, status: 200, json: async () => [{ id: 2, role: 'user', content: '会话2新消息' }] };
          }
          return { ok: true, status: 200, json: async () => [] };
        }
      });

      const s1Promise = sb2.selectSession(1);
      const s2Promise = sb2.selectSession(2);
      resolveS2();
      await s2Promise;
      resolveS1();
      await s1Promise;

      const finalMessagesHtml = sb2.document.getElementById('messages-container').innerHTML;

      return {
        abortedOnNewChat,
        convIdAfterNewChat,
        sessionBadgeAfterNewChat,
        finalMessagesHtml,
      };
    """)

    assert res["abortedOnNewChat"] is True
    assert res["convIdAfterNewChat"] is None
    assert "自动生成" in res["sessionBadgeAfterNewChat"]
    assert "会话2新消息" in res["finalMessagesHtml"]
    assert "会话1旧消息" not in res["finalMessagesHtml"]


# ==============================================================================
# FE-08, FE-09, FE-17: index.html refund btn, order_time, history citations, order regex
# ==============================================================================

def test_fe08_fe09_fe17_index_refund_modal_order_card_and_history_citations():
    res = _run_node(r"""
      const sb = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'index.html'),
        fetchImpl: async (url) => {
          if (url === '/api/tickets') {
            return { ok: true, status: 200, json: async () => ({ ticket_no: 'TK-9001', conversation_id: 1, ticket_type: '售后', status: '待处理' }) };
          }
          if (url === '/api/chat/stream') {
            return makeSseResponse([
              'data: {"event_type":"text","conversation_id":88,"content":"已收到订单信息"}\n\n',
              'data: [DONE]\n\n',
            ]);
          }
          return { ok: true, status: 200, json: async () => [] };
        }
      });

      // FE-09: order_time renders on order selector card
      const host = sb.document.createElement('div');
      sb.renderOrderSelectorCards(host, [{
        order_id: '1001',
        product_name: '智能猫砂盆',
        amount: '899元',
        status: '已发货',
        order_time: '2026-09-05 12:30:00',
      }]);
      const orderTimeHtml = host.querySelector('.order-card-time').innerHTML;

      // FE-17(a): Historical assistant messages without citations don't render dead .cite-pill buttons
      sb.renderConversationMessages([
        { id: 1, role: 'assistant', content: '根据规则[1]可退货' }
      ]);
      const historyHtml = sb.document.getElementById('messages-container').innerHTML;
      const historyCiteButtons = sb.document.querySelectorAll('#messages-container .cite-pill').length;

      // FE-17(b): Refund modal unlocks order input when no order was detected
      const refundBtn1 = sb.document.createElement('button');
      sb.__eval('currentRefundOrderId = null');
      sb.openRefundModal(refundBtn1);
      const unlockedWhenEmpty = sb.document.getElementById('refund-modal-order-id').readOnly === false;
      const valWhenEmpty = sb.document.getElementById('refund-modal-order-id').value;

      // FE-08: submitRefundModal disables source activeRefundBtn & marks 退款已申请
      sb.document.getElementById('refund-modal-order-id').value = 'ORD-2026099';
      await sb.submitRefundModal();
      const refundBtnDisabled = refundBtn1.disabled;
      const refundBtnHtml = refundBtn1.innerHTML;

      // FE-17(c): Broader order-ID extraction in sendMessage
      sb.__eval('currentRefundOrderId = null');
      sb.document.getElementById('user-input').value = '订单号 ORD-2026888 想退货';
      await sb.sendMessage();
      await new Promise(r => setTimeout(r, 40));
      const extractedOrderId = sb.__eval('currentRefundOrderId');

      return {
        orderTimeHtml,
        historyHtml,
        historyCiteButtons,
        unlockedWhenEmpty,
        valWhenEmpty,
        refundBtnDisabled,
        refundBtnHtml,
        extractedOrderId,
      };
    """)

    assert "2026-09-05 12:30:00" in res["orderTimeHtml"]
    assert "根据规则[1]可退货" in res["historyHtml"]
    assert res["historyCiteButtons"] == 0
    assert res["unlockedWhenEmpty"] is True
    assert res["valWhenEmpty"] == ""
    assert res["refundBtnDisabled"] is True
    assert "退款已申请" in res["refundBtnHtml"]
    assert res["extractedOrderId"] == "ORD-2026888"


# ==============================================================================
# FE-10 & FE-13: rag_eval.html evaluated_at, zero metrics, pagination, non-OK
# ==============================================================================

def test_fe10_fe13_rag_eval_meta_zero_metrics_pagination_and_error_handling():
    res = _run_node(r"""
      const requestedUrls = [];
      let reportResponse = {
        ok: false,
        status: 500,
        json: async () => { throw new SyntaxError('Unexpected token <'); },
      };

      const sb = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'rag_eval.html'),
        fetchImpl: async (url) => {
          requestedUrls.push(url);
          if (url === '/api/rag-eval/report') return reportResponse;
          if (url.startsWith('/api/rag-eval/faith-cases')) {
            const pageMatch = url.match(/page=(\d+)/);
            const page = pageMatch ? Number(pageMatch[1]) : 1;
            return {
              ok: true, status: 200,
              json: async () => ({
                metrics: { current_round_judge_rate: 0, current_round_human_confirmed_rate: 0, ledger_summary: { total: 45, unresolved: 45, resolved: 0, wont_resolve: 0 } },
                total: 45,
                page,
                page_size: 20,
                items: [{ id: page, eval_id: 'E' + page, bucket: 'A_policy', strategy: 'hybrid', query: 'q', answer: 'a', reason: 'r', status: '未解决', seen_count: 1, citations: [], cited_indices: [] }]
              })
            };
          }
          return { ok: true, status: 200, json: async () => ({}) };
        }
      });

      // FE-13: HTTP 500 plain-text must NOT falsely show "尚未生成评测报告产物"
      await sb.fetchReportData();
      const emptyDisplayOn500 = sb.document.getElementById('empty-state-card').style.display;
      const emptyTitleOn500 = sb.document.querySelector('#empty-state-card .empty-title')
        ? sb.document.querySelector('#empty-state-card .empty-title').textContent : '';

      // FE-10: evaluated_at + zero metrics (no 0.98/1.0 fabrication)
      sb.renderReportMeta({ evaluated_at: '2026-09-16 21:25:58', eval_set_size: 300 });
      const evalTime = sb.document.getElementById('meta-eval-time').textContent;

      sb.renderGenerationSection({
        answer_coverage: { hybrid_rerank: 0 },
        faithfulness: { hybrid_rerank: 0 },
        out_of_scope_refusal_rate: 0,
      });
      const genHtml = sb.document.getElementById('generation-chart-box').innerHTML;

      // FE-10: pagination controls
      await sb.fetchCasesData();
      const prevBtn = sb.document.getElementById('btn-cases-prev');
      const nextBtn = sb.document.getElementById('btn-cases-next');
      const prevDisabledP1 = prevBtn ? prevBtn.disabled : null;
      const nextDisabledP1 = nextBtn ? nextBtn.disabled : null;
      if (nextBtn) nextBtn.click();
      await new Promise(r => setTimeout(r, 20));

      return {
        emptyDisplayOn500,
        emptyTitleOn500,
        evalTime,
        genHtml,
        hasPaginationBtns: !!(prevBtn && nextBtn),
        prevDisabledP1,
        nextDisabledP1,
        requestedUrls,
      };
    """)

    assert res["evalTime"] == "2026-09-16 21:25:58"
    assert ">98%<" not in res["genHtml"]
    assert ">100%<" not in res["genHtml"]
    assert res["genHtml"].count(">0%<") == 3
    assert res["genHtml"].count('stroke-dasharray="0, 100"') == 3
    assert res["hasPaginationBtns"] is True
    assert res["prevDisabledP1"] is True
    assert res["nextDisabledP1"] is False
    assert any("page=2" in u for u in res["requestedUrls"])
    # On HTTP 500, should not show misleading "尚未生成评测报告产物"
    assert not (res["emptyDisplayOn500"] == "flex" and "尚未生成评测报告产物" in res["emptyTitleOn500"])


# ==============================================================================
# FE-11: All 4 job pollers handle 'stopped', HTTP 404, timer leaks, double close
# ==============================================================================

def test_fe11_job_pollers_stopped_errors_and_timer_cleanup():
    res = _run_node(r"""
      const out = {};

      // 1. acceptance.js: 'stopped' terminates & unlocks; 404 terminates & unlocks; closeBtn fires onComplete once
      {
        let logMode = 'stopped';
        let completeCalls = 0;
        const sb = createSandbox({
          htmlPath: path.join(STATIC_DIR, 'acceptance.html'),
          jsPaths: [path.join(STATIC_DIR, 'admin.js'), path.join(STATIC_DIR, 'acceptance.js')],
          fetchImpl: async (url) => {
            if (url === '/api/jobs/start') return { ok: true, status: 200, json: async () => ({ job_id: 'j1' }) };
            if (url.startsWith('/api/jobs/j1/logs')) {
              if (logMode === '404') return { ok: false, status: 404, json: async () => ({ detail: 'not found' }) };
              return { ok: true, status: 200, json: async () => ({ status: logMode, logs: ['line1'] }) };
            }
            return { ok: true, status: 200, json: async () => ({}) };
          }
        });
        await sb.runAcceptanceJob('eval-ch10', { heavy: false, onComplete: () => { completeCalls++; } });
        await sb.__tickIntervals();
        const stoppedIntervals = sb.__intervals.size;
        const stoppedCloseDisabled = sb.document.getElementById('job-modal-close').disabled;
        sb.document.getElementById('job-modal-close').click();

        logMode = '404';
        await sb.runAcceptanceJob('eval-ch10', { heavy: false, onComplete: () => { completeCalls++; } });
        await sb.__tickIntervals();
        const err404Intervals = sb.__intervals.size;
        const err404CloseDisabled = sb.document.getElementById('job-modal-close').disabled;

        out.acceptance_js = { stoppedIntervals, stoppedCloseDisabled, completeCalls, err404Intervals, err404CloseDisabled };
      }

      // 2. observability.html: 'stopped' and 404 stop polling & unlock close button
      {
        let logMode = 'stopped';
        const sb = createSandbox({
          htmlPath: path.join(STATIC_DIR, 'observability.html'),
          fetchImpl: async (url) => {
            if (url === '/api/jobs/start') return { ok: true, status: 200, json: async () => ({ job_id: 'j2' }) };
            if (url.startsWith('/api/jobs/j2/logs')) {
              if (logMode === '404') return { ok: false, status: 404, json: async () => ({ detail: 'not found' }) };
              return { ok: true, status: 200, json: async () => ({ status: logMode, logs: ['stopped'] }) };
            }
            return { ok: true, status: 200, json: async () => ({}) };
          }
        });
        await sb.runJob('cost-analysis');
        await sb.__tickIntervals();
        const stoppedIntervals = sb.__intervals.size;
        const stoppedDisabled = sb.document.getElementById('job-modal-close').disabled;

        logMode = '404';
        await sb.runJob('cost-analysis');
        await sb.__tickIntervals();
        const err404Intervals = sb.__intervals.size;
        const err404Disabled = sb.document.getElementById('job-modal-close').disabled;
        out.observability = { stoppedIntervals, stoppedDisabled, err404Intervals, err404Disabled };
      }

      // 3. review_queue.html: 'stopped' and 404 stop polling & unlock close button
      {
        let logMode = 'stopped';
        const sb = createSandbox({
          htmlPath: path.join(STATIC_DIR, 'review_queue.html'),
          fetchImpl: async (url) => {
            if (url === '/api/jobs/start') return { ok: true, status: 200, json: async () => ({ job_id: 'j3' }) };
            if (url.startsWith('/api/jobs/j3/logs')) {
              if (logMode === '404') return { ok: false, status: 404, json: async () => ({ detail: 'not found' }) };
              return { ok: true, status: 200, json: async () => ({ status: logMode, logs: ['stopped'] }) };
            }
            return { ok: true, status: 200, json: async () => ({ total: 0, items: [] }) };
          }
        });
        await sb.runJob('flywheel-pipeline');
        await sb.__tickIntervals();
        const stoppedIntervals = sb.__intervals.size;
        const stoppedDisabled = sb.document.getElementById('job-modal-close').disabled;

        logMode = '404';
        await sb.runJob('flywheel-pipeline');
        await sb.__tickIntervals();
        const err404Intervals = sb.__intervals.size;
        const err404Disabled = sb.document.getElementById('job-modal-close').disabled;
        out.review_queue = { stoppedIntervals, stoppedDisabled, err404Intervals, err404Disabled };
      }

      // 4. rag_eval.html: closeModal clears interval, 'stopped' & 404 stop polling, badge resets on startJob
      {
        let statusVal = 'failed';
        const sb = createSandbox({
          htmlPath: path.join(STATIC_DIR, 'rag_eval.html'),
          fetchImpl: async (url) => {
            if (url === '/api/jobs') return { ok: true, status: 200, json: async () => ({ job_id: 'j4' }) };
            if (url === '/api/jobs/j4/logs') {
              if (statusVal === '404') return { ok: false, status: 404, json: async () => ({ detail: 'missing' }) };
              return { ok: true, status: 200, json: async () => ({ status: statusVal, logs: ['log'] }) };
            }
            if (url === '/api/jobs/j4') {
              if (statusVal === '404') return { ok: false, status: 404, json: async () => ({ detail: 'missing' }) };
              return { ok: true, status: 200, json: async () => ({ status: statusVal }) };
            }
            return { ok: true, status: 200, json: async () => ({}) };
          }
        });
        await sb.startJob('eval-rag');
        await sb.__tickIntervals(); // completes as failed -> badge-red
        const failedClass = sb.document.getElementById('job-status-badge').className;

        // Re-run -> badge must reset to Running / badge-blue
        statusVal = 'running';
        await sb.startJob('eval-rag');
        const resetText = sb.document.getElementById('job-status-badge').textContent;
        const resetClass = sb.document.getElementById('job-status-badge').className;

        // Closing console-modal while running must clear jobPollTimer
        sb.closeModal('console-modal');
        const intervalsAfterClose = sb.__intervals.size;

        // Stopped status stops polling
        statusVal = 'stopped';
        await sb.startJob('eval-rag');
        await sb.__tickIntervals();
        const stoppedIntervals = sb.__intervals.size;

        // 404 stops polling
        statusVal = '404';
        await sb.startJob('eval-rag');
        await sb.__tickIntervals();
        const err404Intervals = sb.__intervals.size;

        out.rag_eval = { failedClass, resetText, resetClass, intervalsAfterClose, stoppedIntervals, err404Intervals };
      }

      return out;
    """)

    assert res["acceptance_js"]["stoppedIntervals"] == 0
    assert res["acceptance_js"]["stoppedCloseDisabled"] is False
    assert res["acceptance_js"]["completeCalls"] == 1
    assert res["acceptance_js"]["err404Intervals"] == 0
    assert res["acceptance_js"]["err404CloseDisabled"] is False

    assert res["observability"] == {
        "stoppedIntervals": 0, "stoppedDisabled": False,
        "err404Intervals": 0, "err404Disabled": False,
    }
    assert res["review_queue"] == {
        "stoppedIntervals": 0, "stoppedDisabled": False,
        "err404Intervals": 0, "err404Disabled": False,
    }
    assert "badge-red" in res["rag_eval"]["failedClass"]
    assert res["rag_eval"]["resetText"] == "Running"
    assert "badge-blue" in res["rag_eval"]["resetClass"]
    assert res["rag_eval"]["intervalsAfterClose"] == 0
    assert res["rag_eval"]["stoppedIntervals"] == 0
    assert res["rag_eval"]["err404Intervals"] == 0


# ==============================================================================
# FE-12: observability.html cost_percentage <= 1 & latest-round-only regression alert
# ==============================================================================

def test_fe12_observability_cost_percentage_and_latest_run_alert():
    res = _run_node(r"""
      let runs = [
        { id: 2, triggered_by: 'manual', dataset_size: 10, metrics: { mrr: 0.90 }, deltas: { mrr: 0.05 }, created_at: '2026-11-02' },
        { id: 1, triggered_by: 'manual', dataset_size: 10, metrics: { mrr: 0.85 }, deltas: { mrr: -0.04 }, created_at: '2026-11-01' },
      ];
      const sb = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'observability.html'),
        fetchImpl: async () => ({
          ok: true, status: 200,
          json: async () => ({
            cost_block: {
              present: true,
              data: {
                total_requests: 10, total_tokens: 1000, total_cost_usd: 1.0,
                items: [
                  { intent: 'refund', total_calls: 8, prompt_tokens: 500, completion_tokens: 400, total_tokens: 900, estimated_cost_usd: 0.985, cost_percentage: 98.5 },
                  { intent: 'tiny', total_calls: 1, prompt_tokens: 3, completion_tokens: 2, total_tokens: 5, estimated_cost_usd: 0.005, cost_percentage: 0.5 },
                  { intent: 'onepct', total_calls: 1, prompt_tokens: 5, completion_tokens: 5, total_tokens: 10, estimated_cost_usd: 0.01, cost_percentage: 1.0 },
                ]
              }
            },
            eval_trend_block: { present: true, runs },
            calibration_block: { present: false, data: null },
          })
        })
      });

      await sb.loadOverview();
      const costHtml = sb.document.getElementById('cost-tbody').innerHTML;
      const alertWhenLatestImproved = sb.document.getElementById('eval-alert').style.display;

      runs = [
        { id: 3, triggered_by: 'manual', dataset_size: 10, metrics: { mrr: 0.80 }, deltas: { mrr: -0.10 }, created_at: '2026-11-03' },
        { id: 2, triggered_by: 'manual', dataset_size: 10, metrics: { mrr: 0.90 }, deltas: { mrr: 0.05 }, created_at: '2026-11-02' },
      ];
      await sb.loadOverview();
      const alertWhenLatestDropped = sb.document.getElementById('eval-alert').style.display;

      return { costHtml, alertWhenLatestImproved, alertWhenLatestDropped };
    """)

    assert "0.5%" in res["costHtml"]
    assert "50.0%" not in res["costHtml"]
    assert "1.0%" in res["costHtml"]
    assert "100.0%" not in res["costHtml"]
    assert res["alertWhenLatestImproved"] == "none"
    assert res["alertWhenLatestDropped"] == "block"


# ==============================================================================
# FE-13: kb.html non-OK & non-JSON error handling
# ==============================================================================

def test_fe13_kb_non_ok_and_non_json_error_handling():
    res = _run_node(r"""
      const sb = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'kb.html'),
        fetchImpl: async () => ({
          ok: false,
          status: 502,
          json: async () => { throw new SyntaxError('Unexpected token <'); },
        })
      });
      await sb.loadMaterials();
      await sb.loadChunks();
      sb.document.getElementById('search-query').value = '运费';
      await sb.executeSearch();
      return {
        materialsHtml: sb.document.getElementById('materials-tbody').innerHTML,
        chunksHtml: sb.document.getElementById('chunks-tbody').innerHTML,
        searchHtml: sb.document.getElementById('search-results').innerHTML,
      };
    """)

    assert "502" in res["materialsHtml"] and "map" not in res["materialsHtml"]
    assert "502" in res["chunksHtml"] and "length" not in res["chunksHtml"]
    assert "502" in res["searchHtml"] and "Unexpected token" not in res["searchHtml"]


# ==============================================================================
# FE-14: acceptance.js & acceptance_eval.html threshold, labels, offline demo
# ==============================================================================

def test_fe14_acceptance_eval_threshold_labels_and_offline_demo():
    html = (STATIC_DIR / "acceptance_eval.html").read_text(encoding="utf-8")
    assert "0.30 ~ 0.70" in html
    assert "0.20 ~ 0.60" not in html

    res = _run_node(r"""
      const calls = [];
      const sb = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'acceptance_eval.html'),
        jsPaths: [path.join(STATIC_DIR, 'admin.js'), path.join(STATIC_DIR, 'acceptance.js')],
        fetchImpl: async (url) => {
          calls.push(url);
          if (url === '/api/acceptance/eval') {
            return {
              ok: true, status: 200,
              json: async () => ({
                present: true,
                summary: { macro_f1: 0.97, micro_f1: 0.98, macro_precision: 0.97, macro_recall: 0.97 },
                threshold: 0.35,
                per_class: [],
                candidate_scores: { '0.30': 0.9742, '0.35': 0.9767, '0.70': 0.9681 },
                confusion_matrices: {},
              })
            };
          }
          if (url === '/api/acceptance/service') {
            return { ok: true, status: 200, json: async () => ({ online: false, status: 'offline' }) };
          }
          if (url === '/api/acceptance/classify') {
            return { ok: false, status: 503, json: async () => ({ detail: 'offline' }) };
          }
          return { ok: true, status: 200, json: async () => ({}) };
        }
      });

      await sb.loadEval();
      await new Promise(r => setTimeout(r, 40));

      const thresholdLabel = sb.document.getElementById('demo-threshold-label').innerText
        || sb.document.getElementById('demo-threshold-label').textContent;
      const scanBarsHtml = sb.document.getElementById('threshold-scan-bars').innerHTML;

      // User types custom text, then refreshes loadEval() -> input must be preserved
      sb.document.getElementById('demo-input-text').value = '用户自定义提问';
      await sb.loadEval();
      const preservedInput = sb.document.getElementById('demo-input-text').value;

      return {
        thresholdLabel,
        scanBarsHtml,
        preservedInput,
        classifyCalls: calls.filter(c => c === '/api/acceptance/classify').length,
      };
    """)

    assert res["thresholdLabel"] == "0.35"
    assert "Micro F1" in res["scanBarsHtml"]
    assert "Macro F1" not in res["scanBarsHtml"]
    assert res["classifyCalls"] == 0, "Should not auto-fire /api/acceptance/classify when service is offline"
    assert res["preservedInput"] == "用户自定义提问"


# ==============================================================================
# FE-15: admin.js uses PAGE_METAS defaults & rel="noopener noreferrer"
# ==============================================================================

def test_fe15_admin_js_page_metas_and_noopener():
    res = _run_node(r"""
      const out = {};
      for (const [pathname, key] of [['/kb', 'kb'], ['/observability', 'observability'], ['/acceptance/eval', 'acceptance']]) {
        const sb = createSandbox({
          jsPaths: [path.join(STATIC_DIR, 'admin.js')],
          pathname,
        });
        const header = sb.document.querySelector('header');
        const docsLink = header.querySelector('a[href="/docs"]');
        out[key] = {
          title: header.querySelector('.brand-title').innerText,
          icon: header.querySelector('.brand-icon').innerText,
          docsRel: docsLink ? docsLink.getAttribute('rel') : null,
        };
      }
      return out;
    """)

    assert "知识库" in res["kb"]["title"]
    assert res["kb"]["icon"] == "📚"
    assert "观测与成本" in res["observability"]["title"]
    assert "实证验收" in res["acceptance"]["title"]
    assert res["kb"]["docsRel"] == "noopener noreferrer"


# ==============================================================================
# FE-16: topic_distribution.html renders 0% instead of -% for zero question_percentage
# ==============================================================================

def test_fe16_topic_distribution_zero_question_percentage():
    res = _run_node(r"""
      const sb = createSandbox({
        htmlPath: path.join(STATIC_DIR, 'topic_distribution.html'),
        jsPaths: [path.join(STATIC_DIR, 'admin.js'), path.join(STATIC_DIR, 'acceptance.js')],
        fetchImpl: async () => ({
          ok: true, status: 200,
          json: async () => ({
            total_classified: 10,
            total_unclassified: 0,
            total_pool: 10,
            top_1_topic: '物流',
            top_3_topics: ['物流'],
            distribution: [
              { category: '物流', count: 10, percentage: 100, question_percentage: 100, tier: 'strict' },
              { category: '评价', count: 0, percentage: 0, question_percentage: 0, tier: 'loose' },
            ]
          })
        })
      });
      await sb.loadDistribution();
      return { tbodyHtml: sb.document.getElementById('distribution-tbody').innerHTML };
    """)

    assert "-%" not in res["tbodyHtml"]
    assert "<td>0%</td>" in res["tbodyHtml"]
