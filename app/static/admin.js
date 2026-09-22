/**
 * app/static/admin.js - Unified Administration Shell and Navigation
 * 
 * Provides:
 * - Unified top navigation across all backend modules
 * - Secondary subnav tabs for the acceptance suite (/acceptance*)
 * - Active state auto-detection based on window.location.pathname
 */

(function () {
  const MAIN_NAV_ITEMS = [
    { href: "/admin", label: "🏠 管理首页", key: "admin" },
    { href: "/kb", label: "📚 知识库", key: "kb" },
    { href: "/rag-eval", label: "⚖️ RAG 评测", key: "rag-eval" },
    { href: "/observability", label: "📈 观测与成本", key: "observability" },
    { href: "/review-queue", label: "🔄 飞轮待审", key: "review-queue" },
    { href: "/topic-distribution", label: "📊 主题分布", key: "topic-distribution" },
    { href: "/acceptance", label: "🛡️ 实证验收", key: "acceptance" },
    { href: "/chat", label: "💬 在线客服", key: "chat" },
    { href: "/docs", label: "📖 API 文档", key: "docs", external: true },
  ];

  const ACCEPTANCE_SUBNAV_ITEMS = [
    { href: "/acceptance", label: "🛡️ 验收总览与闸门", key: "overview" },
    { href: "/acceptance/eval", label: "🔬 评测深度剖析", key: "eval" },
    { href: "/acceptance/data", label: "🧬 语料与考卷血缘", key: "data" },
    { href: "/acceptance/errors", label: "📒 错例与错因账本", key: "errors" },
  ];

  const PAGE_METAS = {
    "admin": { icon: "⚙️", title: "后台聚合管理中心", subtitle: "全方位监控知识沉淀、数据飞轮、分类模型验收与生产质量成本" },
    "kb": { icon: "📚", title: "知识库管理工作台", subtitle: "Dense 密集语义向量知识库 · 双写落库与自愈沙盒" },
    "rag-eval": { icon: "⚖️", title: "RAG 质量评估工作台", subtitle: "四策略综合对比 · 忠实度裁判与编造台账" },
    "observability": { icon: "📈", title: "观测与成本看板", subtitle: "全链路 Trace 意图成本测算 · 动态置信度校准" },
    "review-queue": { icon: "🔄", title: "飞轮待审队列", subtitle: "低置信度问题语义聚类 · 运营核准写回知识库" },
    "topic-distribution": { icon: "📊", title: "17 类主题分布看板", subtitle: "多标签分类器识别高频堆积 · 指导知识库优先补充" },
    "acceptance": { icon: "🛡️", title: "九项实证验收总览", subtitle: "零泄漏、17 类 F1 红线、记账平衡与 ONNX 一致性" },
    "chat": { icon: "💬", title: "在线客服体验", subtitle: "多轮 SSE 流式对话 · 角色约束与工单提取" },
  };

  function detectActiveKeys() {
    const path = window.location.pathname;
    let activeMain = "";
    let activeSub = "";

    if (path === "/admin") activeMain = "admin";
    else if (path === "/kb") activeMain = "kb";
    else if (path.startsWith("/rag-eval") || path.startsWith("/rag_eval")) activeMain = "rag-eval";
    else if (path === "/observability") activeMain = "observability";
    else if (path === "/review-queue") activeMain = "review-queue";
    else if (path === "/topic-distribution") activeMain = "topic-distribution";
    else if (path.startsWith("/acceptance")) {
      activeMain = "acceptance";
      if (path === "/acceptance" || path === "/acceptance/") activeSub = "overview";
      else if (path.startsWith("/acceptance/eval")) activeSub = "eval";
      else if (path.startsWith("/acceptance/data")) activeSub = "data";
      else if (path.startsWith("/acceptance/errors")) activeSub = "errors";
    } else if (path === "/chat") activeMain = "chat";

    return { activeMain, activeSub };
  }

  function renderAdminShell(options = {}) {
    const detected = detectActiveKeys();
    const activeMain = options.activeMain || detected.activeMain;
    const activeSub = options.activeSub || detected.activeSub;
    const title = options.title || "AstroBot 后台聚合管理中心";
    const subtitle = options.subtitle || "智能电商客服系统管理中心";
    const icon = options.icon || "⚙️";

    // 1. Render Header
    let headerEl = document.querySelector("header");
    if (!headerEl) {
      headerEl = document.createElement("header");
      document.body.insertBefore(headerEl, document.body.firstChild);
    }

    const mainNavHtml = MAIN_NAV_ITEMS.map((item) => {
      const isActive = item.key === activeMain ? "active" : "";
      const target = item.external ? 'target="_blank"' : "";
      return `<a href="${item.href}" class="btn-nav ${isActive}" ${target}>${item.label}</a>`;
    }).join("\n");

    headerEl.innerHTML = `
      <a href="/admin" class="brand">
        <div class="brand-icon">${icon}</div>
        <div>
          <div class="brand-title">${title}</div>
          <div class="brand-subtitle">${subtitle}</div>
        </div>
      </a>
      <div class="header-links">
        ${mainNavHtml}
      </div>
    `;

    // 2. Render Secondary Subnav for Acceptance pages
    if (activeMain === "acceptance") {
      let subnavEl = document.getElementById("acceptance-subnav");
      if (!subnavEl) {
        subnavEl = document.createElement("div");
        subnavEl.id = "acceptance-subnav";
        subnavEl.className = "subnav-bar";
        headerEl.parentNode.insertBefore(subnavEl, headerEl.nextSibling);
      }

      const subnavHtml = ACCEPTANCE_SUBNAV_ITEMS.map((item) => {
        const isActive = item.key === activeSub ? "active" : "";
        return `<a href="${item.href}" class="subnav-tab ${isActive}">${item.label}</a>`;
      }).join("\n");

      subnavEl.innerHTML = subnavHtml;
    }
  }

  // Export to window
  window.renderAdminShell = renderAdminShell;

  // Auto-run if script is loaded
  document.addEventListener("DOMContentLoaded", function () {
    // If header has data-auto-render="true" or page hasn't manually called it
    if (!window._adminShellRendered) {
      window._adminShellRendered = true;
      renderAdminShell();
    }
  });
})();
