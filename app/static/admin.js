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

  function detectActiveKeys() {
    const path = window.location.pathname;
    let activeMain = "";
    let activeSub = "";

    if (path === "/admin") activeMain = "admin";
    else if (path === "/kb") activeMain = "kb";
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
    const title = options.title || document.title.replace("AstroBot - ", "");
    const subtitle = options.subtitle || "AstroBot 智能电商客服系统管理中心";
    const icon = options.icon || "🛡️";

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
