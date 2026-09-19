/**
 * app/static/acceptance.js - Acceptance Suite Shared Logic
 * 
 * Provides:
 * - API Client functions for /api/acceptance/*, /api/topics/distribution, /api/jobs/*
 * - Unified Job execution with heavy-task confirmation and realtime terminal log streaming
 * - Interactive single-sentence multi-label classification tester
 * - Toast and UI utility helpers
 */

(function () {
  const HEAVY_JOBS = new Set([
    "train-ch10",
    "data-prep-ch10",
    "eval-ch10",
    "export-onnx",
    "flywheel-pipeline",
  ]);

  // =========================================================================
  // 1. API Fetch Helpers
  // =========================================================================
  async function fetchJson(url, options = {}) {
    const res = await fetch(url, options);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `请求失败 (HTTP ${res.status})`);
    }
    return await res.json();
  }

  window.fetchAcceptanceOverview = () => fetchJson("/api/acceptance/overview");
  window.fetchAcceptanceEval = () => fetchJson("/api/acceptance/eval");
  window.fetchAcceptanceData = () => fetchJson("/api/acceptance/data");
  window.fetchAcceptanceErrors = () => fetchJson("/api/acceptance/errors");
  window.fetchAcceptanceService = () => fetchJson("/api/acceptance/service");
  window.fetchTopicDistribution = () => fetchJson("/api/topics/distribution");

  window.classifySingleText = async (text) => {
    return await fetchJson("/api/acceptance/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    });
  };

  // =========================================================================
  // 2. Toast Utility
  // =========================================================================
  window.showToast = function (message, type = "info") {
    let toast = document.getElementById("toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "toast";
      toast.className = "toast";
      document.body.appendChild(toast);
    }
    toast.innerText = message;
    toast.style.display = "block";
    if (type === "error") {
      toast.style.background = "#b91c1c";
    } else if (type === "success") {
      toast.style.background = "#047857";
    } else {
      toast.style.background = "#1e293b";
    }
    setTimeout(() => {
      toast.style.display = "none";
    }, 3500);
  };

  // =========================================================================
  // 3. Tri-state Badge Renderer
  // =========================================================================
  window.renderGateBadge = function (status) {
    if (status === "pass") {
      return '<span class="badge badge-pass">✓ PASS 达标</span>';
    } else if (status === "fail") {
      return '<span class="badge badge-fail">✕ FAIL 未达</span>';
    } else {
      return '<span class="badge badge-missing">○ MISSING 待建</span>';
    }
  };

  // =========================================================================
  // 4. Job Modal & Realtime Log Streaming
  // =========================================================================
  let pollInterval = null;

  function ensureJobModal() {
    let modal = document.getElementById("job-modal");
    if (!modal) {
      modal = document.createElement("div");
      modal.id = "job-modal";
      modal.className = "modal-overlay";
      modal.innerHTML = `
        <div class="modal-box">
          <div class="modal-header">
            <span id="job-modal-title" style="display:flex;align-items:center;gap:8px;">
              ⚡ 执行作业
            </span>
            <span id="job-modal-status" class="badge badge-missing">准备中</span>
          </div>
          <div class="console-body" id="job-modal-logs">正在启动作业...</div>
          <div class="modal-footer">
            <button class="btn-secondary" id="job-modal-close" disabled>等待完成</button>
          </div>
        </div>
      `;
      document.body.appendChild(modal);

      const closeBtn = document.getElementById("job-modal-close");
      closeBtn.addEventListener("click", () => {
        closeJobModal();
      });
    }
    return modal;
  }

  window.runAcceptanceJob = async function (jobName, options = {}) {
    const isHeavy = options.heavy || HEAVY_JOBS.has(jobName);

    if (isHeavy) {
      const confirmed = window.confirm(
        `⚠️ 【重型任务提醒】\n\n作业 [${jobName}] 包含模型微调/全量评测/语料处理，需要消耗较多计算资源并可能持续数分钟。\n\n确定立即启动执行吗？`
      );
      if (!confirmed) return;
    }

    const modal = ensureJobModal();
    const titleEl = document.getElementById("job-modal-title");
    const statusEl = document.getElementById("job-modal-status");
    const logsEl = document.getElementById("job-modal-logs");
    const closeBtn = document.getElementById("job-modal-close");

    modal.classList.add("active");
    titleEl.innerHTML = `⚡ 执行作业: <code style="background:#f1f5f9;padding:2px 6px;border-radius:4px;">${jobName}</code>`;
    statusEl.className = "badge badge-missing";
    statusEl.innerText = "运行中 (Running)";
    logsEl.innerText = `[${new Date().toLocaleTimeString()}] 正在启动作业: ${jobName}...\n`;
    closeBtn.disabled = true;
    closeBtn.innerText = "等待执行完成";

    try {
      const startRes = await fetchJson("/api/jobs/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_name: jobName }),
      });

      const jobId = startRes.job_id;
      logsEl.innerText += `[${new Date().toLocaleTimeString()}] 作业已派发，Job ID: ${jobId}\n`;

      if (pollInterval) clearInterval(pollInterval);
      pollInterval = setInterval(async () => {
        try {
          const logData = await fetchJson(`/api/jobs/${jobId}/logs?tail=100`);
          if (logData && logData.logs) {
            logsEl.innerText = logData.logs.join("\n");
            logsEl.scrollTop = logsEl.scrollHeight;
          }

          let status = logData.status;
          if (!status) {
            const statusData = await fetchJson(`/api/jobs/${jobId}`);
            status = statusData.status;
          }

          if (status === "completed" || status === "failed") {
            clearInterval(pollInterval);
            pollInterval = null;

            if (status === "completed") {
              statusEl.className = "badge badge-pass";
              statusEl.innerText = "✓ 执行成功 (Done)";
              closeBtn.innerText = "完成并刷新";
            } else {
              statusEl.className = "badge badge-fail";
              statusEl.innerText = "✕ 执行失败 (Failed)";
              closeBtn.innerText = "关闭";
            }
            closeBtn.disabled = false;

            closeBtn.onclick = () => {
              closeJobModal();
              if (options.onComplete) {
                options.onComplete(status === "completed");
              } else {
                window.location.reload();
              }
            };
          }
        } catch (pollErr) {
          console.error("Job polling error", pollErr);
        }
      }, 1000);
    } catch (err) {
      logsEl.innerText += `\n[ERROR] 启动失败: ${err.message}\n`;
      statusEl.className = "badge badge-fail";
      statusEl.innerText = "启动异常";
      closeBtn.disabled = false;
      closeBtn.innerText = "关闭";
      closeBtn.onclick = () => closeJobModal();
    }
  };

  function closeJobModal() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
    const modal = document.getElementById("job-modal");
    if (modal) modal.classList.remove("active");
  }

  // =========================================================================
  // 5. Single Query Interactive Tester
  // =========================================================================
  window.initClassifyDemo = function (containerId, initialText = "买大了想退") {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.innerHTML = `
      <div class="classify-box">
        <div style="font-weight:600;font-size:14.5px;margin-bottom:8px;color:#1e293b;">
          🎯 单句试分类实时演示 (17 类各自独立过线，过几个打几个)
        </div>
        <div style="font-size:13px;color:var(--text-muted);margin-bottom:12px;">
          输入客户口语原话（如“买大了想退”），现场调用独立推理服务 (:8110)，观察 17 个类目各自的置信度得分及过线状态。
        </div>
        <div class="classify-input-group">
          <input type="text" id="demo-input-text" class="classify-input" value="${initialText}" placeholder="请输入客户咨询文本..." />
          <button class="btn-primary" id="demo-classify-btn">⚡ 立即分类</button>
        </div>
        <div id="demo-results-wrap" style="display:none;">
          <div style="margin-bottom:12px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            <strong>命中权威标签:</strong>
            <div id="demo-hit-tags" class="classify-tags"></div>
          </div>
          <div style="font-size:12.5px;color:var(--text-muted);margin-bottom:8px;">
            17 类独立置信度得分 (判定红线: <code id="demo-threshold-label">0.45</code>)：
          </div>
          <div id="demo-bars-container" class="bar-chart-container"></div>
        </div>
      </div>
    `;

    const inputEl = document.getElementById("demo-input-text");
    const btnEl = document.getElementById("demo-classify-btn");
    const resultsWrap = document.getElementById("demo-results-wrap");
    const hitTagsEl = document.getElementById("demo-hit-tags");
    const barsEl = document.getElementById("demo-bars-container");

    async function doClassify() {
      const text = inputEl.value.trim();
      if (!text) {
        showToast("请输入待分类文本", "error");
        return;
      }
      btnEl.disabled = true;
      btnEl.innerText = "分类中...";
      try {
        const data = await classifySingleText(text);
        resultsWrap.style.display = "block";

        const labels = data.labels || [];
        if (labels.length === 0) {
          hitTagsEl.innerHTML = '<span class="classify-tag" style="background:#f1f5f9;color:#64748b;">(未命中任何类目，无过线标签)</span>';
        } else {
          hitTagsEl.innerHTML = labels.map((l) => `<span class="classify-tag active">🏷️ ${l}</span>`).join(" ");
        }

        const scores = data.scores || {};
        const sortedCats = Object.entries(scores).sort((a, b) => b[1] - a[1]);

        barsEl.innerHTML = sortedCats.map(([cat, score]) => {
          const isHit = labels.includes(cat);
          const pct = Math.min(100, Math.max(0, score * 100)).toFixed(1);
          return `
            <div class="bar-chart-row">
              <div class="bar-label" style="${isHit ? 'color:var(--primary);font-weight:700;' : ''}">
                ${isHit ? '🔥 ' : ''}${cat}
              </div>
              <div class="bar-track">
                <div class="bar-fill ${isHit ? 'top3' : ''}" style="width:${pct}%"></div>
              </div>
              <div class="bar-meta">
                <strong>${score.toFixed(4)}</strong> (${pct}%)
                ${isHit ? '<span class="badge badge-pass" style="margin-left:4px;">过线</span>' : ''}
              </div>
            </div>
          `;
        }).join("");
      } catch (err) {
        showToast(`分类请求失败: ${err.message}`, "error");
      } finally {
        btnEl.disabled = false;
        btnEl.innerText = "⚡ 立即分类";
      }
    }

    btnEl.addEventListener("click", doClassify);
    inputEl.addEventListener("keypress", (e) => {
      if (e.key === "Enter") doClassify();
    });

    // Auto-run once if initial text present
    if (initialText) {
      setTimeout(doClassify, 300);
    }
  };
})();
