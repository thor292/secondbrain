// Shared helpers for every SecondBrain page: DOM building, formatting, the top bar and small animations.
const SB = (() => {
  const SVG_NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs = {}, ...children) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === "class") n.className = v;
      else if (k === "style" && typeof v === "object") Object.assign(n.style, v);
      else if (k.startsWith("on") && typeof v === "function") n[k] = v;
      else n.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) if (c != null && c !== false) n.append(c instanceof Node ? c : document.createTextNode(c));
    return n;
  }
  function svg(tag, attrs = {}, ...children) {
    const n = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
    for (const c of children.flat()) if (c != null) n.append(c);
    return n;
  }

  const money = (v) => (v < 0 ? "−$" : "$") + Math.abs(Math.round(v || 0)).toLocaleString();
  const shortMoney = (v) => {
    const a = Math.abs(v || 0), s = v < 0 ? "−$" : "$";
    return a >= 1e6 ? s + (a / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M" : a >= 1e3 ? s + Math.round(a / 1e3) + "k" : s + Math.round(a);
  };
  const pct = (v) => (v == null ? "—" : Math.round(v * 100) + "%");
  const fmtDay = (iso) => (iso ? new Date(String(iso).slice(0, 10) + "T00:00:00").toLocaleDateString(undefined, { month: "short", day: "numeric" }) : "—");
  const addDays = (iso, n) => { const d = new Date(iso + "T00:00:00"); d.setDate(d.getDate() + n); return d.toISOString().slice(0, 10); };

  // Numbers count up when they appear: the page feels alive.
  function countUp(node, to, format = (x) => Math.round(x).toLocaleString(), ms = 1100, from = 0) {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches || !isFinite(to)) { node.textContent = format(to); return; }
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min(1, (now - start) / ms), e = 1 - Math.pow(1 - t, 3);
      node.textContent = format(from + (to - from) * e);
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function kpi(label, value, opts = {}) {
    const v = el("div", { class: "v " + (opts.cls || "") });
    if (typeof value === "number") countUp(v, value, opts.format, opts.ms); else v.textContent = value;
    return el("div", { class: `kpi rise ${opts.glow || ""}`, title: opts.tip, style: opts.delay ? { animationDelay: opts.delay + "ms" } : null },
      el("div", { class: "k" }, opts.icon ? el("span", {}, opts.icon) : null, label), v,
      opts.d ? el("div", { class: "d" }, opts.d) : null, opts.spark || null);
  }

  function sparkline(values, { color = "var(--cyan)", w = 120, h = 40, fill = true } = {}) {
    const max = Math.max(1, ...values), min = Math.min(0, ...values), n = values.length;
    const x = (i) => (n <= 1 ? 0 : (i / (n - 1)) * w), y = (v) => h - 2 - ((v - min) / (max - min || 1)) * (h - 6);
    const pts = values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
    const id = "g" + Math.random().toString(36).slice(2, 8);
    return svg("svg", { viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: "none", class: "spark" },
      svg("defs", {}, svg("linearGradient", { id, x1: 0, y1: 0, x2: 0, y2: 1 },
        svg("stop", { offset: "0%", "stop-color": color, "stop-opacity": 0.45 }), svg("stop", { offset: "100%", "stop-color": color, "stop-opacity": 0 }))),
      fill ? svg("polygon", { points: `0,${h} ${pts} ${w},${h}`, fill: `url(#${id})` }) : null,
      svg("polyline", { points: pts, fill: "none", stroke: color, "stroke-width": 2, "vector-effect": "non-scaling-stroke" }));
  }

  const LOGO = `<svg viewBox="0 0 32 32" aria-hidden="true"><defs><linearGradient id="sbg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#8b7bff"/><stop offset="1" stop-color="#33d6f2"/></linearGradient></defs>
    <circle cx="16" cy="16" r="14" fill="none" stroke="url(#sbg)" stroke-width="2"/>
    <g stroke="url(#sbg)" stroke-width="1.6"><line x1="16" y1="9" x2="9" y2="20"/><line x1="16" y1="9" x2="23" y2="20"/><line x1="9" y1="20" x2="23" y2="20"/></g>
    <g fill="url(#sbg)"><circle cx="16" cy="9" r="3"/><circle cx="9" cy="20" r="2.6"/><circle cx="23" cy="20" r="2.6"/></g></svg>`;

  // The same top bar on every page.
  function nav(active, ...right) {
    const links = [["/", "Chat", "chat"], ["/explore", "Explore", "explore"], ["/graph-view", "Graph", "graph"], ["/hq", "HQ", "hq"]];
    const logo = el("a", { class: "sb-logo", href: "/" });
    logo.innerHTML = LOGO + "<span>Second<span class='grad-text'>Brain</span></span>";
    const bar = el("nav", { class: "sb-nav" }, logo,
      el("div", { class: "sb-links" }, links.map(([href, label, key]) => el("a", { href, class: key === active ? "on" : "" }, label))),
      el("div", { class: "right" }, right));
    document.body.prepend(bar);
    return bar;
  }

  const store = {
    get(k) { try { return JSON.parse(sessionStorage.getItem(k)); } catch { return null; } },
    set(k, v) { try { v == null ? sessionStorage.removeItem(k) : sessionStorage.setItem(k, JSON.stringify(v)); } catch {} },
  };

  return { el, svg, money, shortMoney, pct, fmtDay, addDays, countUp, kpi, sparkline, nav, store, LOGO };
})();
