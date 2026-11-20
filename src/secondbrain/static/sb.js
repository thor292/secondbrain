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

  // Rupees the Indian way: ₹12,34,567 exact; ₹45,600 / ₹4.2 L / ₹1.3 Cr for headlines.
  const inGroup = (n) => Math.round(n).toLocaleString("en-IN");
  const money = (v) => (v < 0 ? "−₹" : "₹") + inGroup(Math.abs(v || 0));
  const shortMoney = (v) => {
    const a = Math.abs(v || 0), s = v < 0 ? "−₹" : "₹";
    if (a >= 1e7) return s + (a / 1e7).toFixed(a >= 1e9 ? 0 : 1).replace(/\.0$/, "") + " Cr";
    if (a >= 1e5) return s + (a / 1e5).toFixed(a >= 1e6 ? 0 : 1).replace(/\.0$/, "") + " L";
    return s + inGroup(a >= 1e4 ? Math.round(a / 100) * 100 : a);
  };
  const pct = (v) => (v == null ? "—" : Math.round(v * 100) + "%");
  const fmtDay = (iso) => (iso ? new Date(String(iso).slice(0, 10) + "T00:00:00").toLocaleDateString("en-IN", { month: "short", day: "numeric" }) : "—");
  // The festivals that move Indian retail (they follow the lunar calendar, so dates are listed).
  const FESTIVALS = [["2024-10-29", "Dhanteras"], ["2024-11-01", "Diwali"], ["2025-03-14", "Holi"], ["2025-10-18", "Dhanteras"],
    ["2025-10-20", "Diwali"], ["2026-03-04", "Holi"], ["2026-11-06", "Dhanteras"], ["2026-11-08", "Diwali"], ["2027-10-29", "Diwali"]];
  const festivalsBetween = (from, to) => FESTIVALS.filter(([d]) => d >= from && d <= to).map(([day, name]) => ({ day, name }));
  // "Dhanteras today · Diwali in 2 days" for a brain's date; empty when nothing is near.
  function festivalNote(iso, within = 45) {
    if (!iso) return "";
    const t = new Date(String(iso).slice(0, 10) + "T00:00:00");
    return FESTIVALS.map(([d, name]) => [name, Math.round((new Date(d + "T00:00:00") - t) / 864e5)])
      .filter(([, n]) => n >= 0 && n <= within)
      .map(([name, n]) => (n === 0 ? `${name} today` : n === 1 ? `${name} tomorrow` : `${name} in ${n} days`)).join(" · ");
  }
  const addDays = (iso, n) => { const d = new Date(iso + "T00:00:00"); d.setDate(d.getDate() + n); return d.toISOString().slice(0, 10); };

  // Numbers count up when they appear: the page feels alive.
  function countUp(node, to, format = (x) => Math.round(x).toLocaleString("en-IN"), ms = 1100, from = 0) {
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
    const links = [["/", "Chat", "chat"], ["/explore", "Explore", "explore"], ["/graph-view", "Graph", "graph"], ["/commerce", "Live Commerce", "commerce"], ["/hq", "HQ", "hq"]];
    const logo = el("a", { class: "sb-logo", href: "/" });
    logo.innerHTML = LOGO + "<span>Second<span class='grad-text'>Brain</span></span>";
    const bar = el("nav", { class: "sb-nav" }, logo,
      el("div", { class: "sb-links" }, links.map(([href, label, key]) => el("a", { href, class: key === active ? "on" : "" }, label))),
      el("div", { class: "right" }, right));
    document.body.prepend(bar);
    commerceConfig().then((c) => {   // the TimescaleDB page, where this copy has it
      if (!c.tsdb) return;
      const hq = bar.querySelector('.sb-links a[href="/hq"]');
      hq.before(el("a", { href: "/timescale", class: active === "timescale" ? "on" : "" }, "TimescaleDB"));
    });
    return bar;
  }

  // ---------- live notices ----------
  // The page logs in to the live broker with this person's own login. The broker only
  // delivers notices for brains they may see; messages carry ids, so the page reloads
  // from the API, which checks permissions again.
  const MQTT_JS = "https://cdn.jsdelivr.net/npm/mqtt@5.10.1/dist/mqtt.min.js";
  let liveClient = null;
  function loadScript(src) {
    if (window.mqtt) return Promise.resolve();
    return new Promise((ok, fail) => document.head.append(el("script", { src, onload: ok, onerror: fail })));
  }
  function liveBadge(state) {
    let b = document.querySelector(".sb-nav .live-badge");
    if (!b) { b = el("span", { class: "live-badge" }); document.querySelector(".sb-nav .right")?.prepend(b); }
    b.className = "live-badge " + state;
    b.textContent = state === "on" ? "Live" : "Reconnecting…";
    b.title = state === "on" ? "Connected: new signals and knowledge appear the moment they happen" : "Live notices paused";
  }
  // The admin token: typed once per tab, or handed over by the app when the page is opened on the demo
  // machine itself (demo mode, local only).
  let tokenP = null;
  function adminToken() {
    const have = (() => { try { return sessionStorage.getItem("sb-admin-token"); } catch { return null; } })();
    if (have) return Promise.resolve(have);
    return tokenP ||= fetch("/demo/admin-token").then((r) => (r.ok ? r.json() : {})).catch(() => ({})).then((d) => {
      tokenP = null;
      if (d.token) { try { sessionStorage.setItem("sb-admin-token", d.token); } catch {} }
      return d.token || null;
    });
  }

  // A read from the Live Commerce engine that waits out a busy engine (504, or no answer yet) instead of
  // failing: only a stopped engine (or 90 s of busy) gives up, so a page never falls back to its demo feed
  // just because the engine was in the middle of a big batch.
  async function engineFetch(url, { timeout = 8000, onWait } = {}) {
    const until = Date.now() + 90000;
    for (let n = 0; ; n++) {
      const ctl = new AbortController(), t = setTimeout(() => ctl.abort(), timeout);
      try {
        let tok = null; try { tok = sessionStorage.getItem("sb-admin-token"); } catch {}
        const r = await fetch(url, { signal: ctl.signal, headers: tok ? { "X-Admin-Token": tok } : {} });
        if (r.ok) return await r.json();
        if (r.status !== 504) { const e = new Error(r.statusText); e.status = r.status; throw e; }
      } catch (e) {
        if (e.name !== "AbortError") throw e;
      } finally { clearTimeout(t); }
      if (Date.now() > until) throw new Error("The engine stayed busy");
      onWait?.(n);
      await new Promise((res) => setTimeout(res, 1500));
    }
  }

  // Live Commerce topics sit under a prefix the app reports (a second copy of the demo uses its own).
  let commercePrefix = "sb/commerce", configP = null;
  function commerceConfig() {
    return configP ||= fetch("/commerce/config").then((r) => (r.ok ? r.json() : {})).catch(() => ({}))
      .then((c) => { if (c.live_prefix) commercePrefix = c.live_prefix; return c; });
  }
  // "sb/commerce/signals/dmart/x" -> "signals/dmart/x"; null for anything else.
  const commerceRest = (topic) => (topic.startsWith(commercePrefix + "/") ? topic.slice(commercePrefix.length + 1) : null);

  async function live(credentialsUrl, headers, onMessage) {
    if (liveClient) { liveClient.end(true); liveClient = null; }
    await commerceConfig();
    try {
      const res = await fetch(credentialsUrl, { headers });
      if (!res.ok) return null;  // live notices switched off: the page still works
      const c = await res.json();
      await loadScript(MQTT_JS);
      const client = window.mqtt.connect(c.url, { username: c.username, password: c.password, reconnectPeriod: 3000 });
      client.on("connect", () => { client.subscribe(c.topics, { qos: 1 }); liveBadge("on"); });
      client.on("offline", () => liveBadge("off"));
      client.on("message", (topic, payload) => {
        let m; try { m = JSON.parse(new TextDecoder().decode(payload)); } catch { return; }
        onMessage(topic, m);
      });
      liveClient = client;
      return client;
    } catch { return null; }
  }
  // Many notices in a burst (e.g. the time machine) become one summary toast.
  function batcher(ms, flush) {
    let items = [], timer = null;
    return (item) => { items.push(item); clearTimeout(timer); timer = setTimeout(() => { const all = items; items = []; flush(all); }, ms); };
  }
  function toast(icon, title, text, { tone = "", ms = 6000, onclick } = {}) {
    let stack = document.querySelector(".toasts");
    if (!stack) { stack = el("div", { class: "toasts", "aria-live": "polite" }); document.body.append(stack); }
    const t = el("div", { class: `toast ${tone}`, onclick: () => { onclick?.(); t.remove(); } },
      el("div", { class: "toast-icon" }, icon),
      el("div", {}, el("div", { class: "toast-title" }, title), text ? el("div", { class: "toast-text" }, text) : null),
      el("div", { class: "toast-bar", style: { animationDuration: ms + "ms" } }));
    stack.append(t);
    setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 400); }, ms);
    while (stack.children.length > 4) stack.firstChild.remove();
  }

  const store = {
    get(k) { try { return JSON.parse(sessionStorage.getItem(k)); } catch { return null; } },
    set(k, v) { try { v == null ? sessionStorage.removeItem(k) : sessionStorage.setItem(k, JSON.stringify(v)); } catch {} },
  };

  return { el, svg, money, shortMoney, pct, fmtDay, addDays, festivalNote, festivalsBetween, countUp, kpi, sparkline, nav, store, LOGO, live, toast, batcher, commerceConfig, commerceRest, adminToken, engineFetch };
})();
