"use strict";
// Dashboard chủ shop: chỉ hiển thị số đếm "ý kiến" (review × chủ đề × cảm xúc)
// do backend trả về trong data.seller_summary. Không có điểm tổng hợp hay lời khuyên.

const SD_SENTS = [
  { id: "positive", label: "Tích cực",  lower: "tích cực",  color: "#008300", ink: "#ffffff" },
  { id: "neutral",  label: "Trung lập", lower: "trung lập", color: "#a8a79f", ink: "#0b0b0b" },
  { id: "negative", label: "Tiêu cực",  lower: "tiêu cực",  color: "#eb6834", ink: "#0b0b0b" },
];
const SD_SENT_BY_ID = Object.fromEntries(SD_SENTS.map(s => [s.id, s]));

// Thang một màu cho bảng nhiệt; `white` là bậc bắt đầu dùng chữ trắng.
const SD_RAMPS = {
  positive: { bg: ["#d3ecd0", "#a6d9a1", "#6fc069", "#2f9e2f", "#006f00"], white: 4 },
  neutral:  { bg: ["#ecebe6", "#d6d5cd", "#b9b8b0", "#8f8e87", "#6b6a65"], white: 4 },
  negative: { bg: ["#fde3d6", "#f9c3a8", "#f29a70", "#eb6834", "#c24e1f"], white: 4 },
};

const SD_PAGE_SIZE = 15;
const SD_LOW_PERIOD = 5;   // cột có ít hơn ngần này ý kiến thì làm mờ
const SD_DENSE_COLS = 14;  // quá ngần này cột thì bỏ nhãn trong cột

const SD_HEAT_TOPICS = 8;  // bảng nhiệt chỉ lấy ngần này chủ đề nhiều ý kiến nhất

// Các cách gom theo thời gian, tính ngay trên trình duyệt từ ngày của từng ý
// kiến. "year" là trục thời gian thật; "month" lọc được theo năm; "day" lọc
// được theo năm và tháng; "weekday" gộp tất cả. Thứ: 0 = thứ Hai … 6 = Chủ nhật.
const SD_WEEKDAYS = ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "CN"];
const SD_DIMS = [
  { id: "year",    label: "Năm",              name: "năm",              tick: k => k,                    full: k => `Năm ${k}` },
  { id: "month",   label: "Tháng",            name: "tháng trong năm",  tick: k => `Th${k}`,             full: k => `Tháng ${k}` },
  { id: "day",     label: "Ngày trong tháng", name: "ngày trong tháng", tick: k => k,                    full: k => `Ngày ${k}` },
  { id: "weekday", label: "Ngày trong tuần",  name: "ngày trong tuần",  tick: k => SD_WEEKDAYS[Number(k)], full: k => SD_WEEKDAYS[Number(k)] },
];
const SD_DIM_BY_ID = Object.fromEntries(SD_DIMS.map(d => [d.id, d]));

const SD_ICONS = {
  reviews:  '<path d="M6 3h9l4 4v14H6z"/><path d="M15 3v4h4"/><path d="M9 12h7"/><path d="M9 16h5"/>',
  opinions: '<path d="M4 5h12v8H9l-5 4z"/><path d="M19 9v7l2 2"/><path d="M19 16h-7"/>',
  positive: '<path d="M7 11v9H4v-9z"/><path d="M7 11l4-8a2 2 0 012 2v4h5a2 2 0 012 2.3l-1 6A2 2 0 0117 20H7"/>',
  neutral:  '<circle cx="12" cy="12" r="9"/><path d="M8 12h8"/>',
  negative: '<path d="M7 13V4H4v9z"/><path d="M7 13l4 8a2 2 0 002-2v-4h5a2 2 0 002-2.3l-1-6A2 2 0 0017 4H7"/>',
  star:     '<path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/>',
};
function sdIcon(name) {
  return `<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${SD_ICONS[name]}</svg>`;
}

const sdState = {
  summary: null,
  productInfo: {},
  sort: "negative",
  mode: "count",
  heat: "negative",
  dim: "year",
  year: "all",   // lọc năm cho chế độ "month" và "day"
  month: "all",  // lọc tháng cho chế độ "day"
  dated: [],     // ý kiến có ngày viết, đã tách sẵn năm/tháng/ngày/thứ
  table: "all",
  topic: "all",
  stars: "all",
  query: "",
  page: 1,
};

function sdEl(id) { return document.getElementById(id); }
function sdPct(part, total) { return total > 0 ? Math.round((part / total) * 100) : 0; }
function sdNum(v) { return Number(v || 0).toLocaleString("vi-VN"); }
function sdDot(color) { return `<span class="sd-dot" style="background:${color}"></span>`; }

function sdDateLabel(iso) {
  if (!iso) return "—";
  const [y, m, d] = iso.split("-");
  return `${d}/${m}/${y}`;
}

// Tô nổi từ khóa đang tìm; escape từng đoạn để không chèn HTML từ review.
function sdHighlight(text, query) {
  const q = query.trim().toLowerCase();
  if (!q) return escHtml(text);
  const lower = text.toLowerCase();
  let out = "", from = 0, at;
  while ((at = lower.indexOf(q, from)) !== -1) {
    out += escHtml(text.slice(from, at)) + `<mark>${escHtml(text.slice(at, at + q.length))}</mark>`;
    from = at + q.length;
  }
  return out + escHtml(text.slice(from));
}

// ─── Bộ chọn cảm xúc dùng chung ────────────────────────────────────────────
function sdRenderSelector(containerId, current, withAll) {
  const options = (withAll ? [{ id: "all", label: "Tất cả", color: null }] : []).concat(SD_SENTS);
  sdEl(containerId).innerHTML = options.map(o => `
    <button type="button" class="sd-chip" data-value="${o.id}" aria-pressed="${o.id === current}">
      ${o.color ? sdDot(o.color) : ""}${o.label}
    </button>`).join("");
}

function sdBindSelector(containerId, onPick) {
  sdEl(containerId).addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-value]");
    if (btn) onPick(btn.dataset.value);
  });
}

// ─── Tổng quan ─────────────────────────────────────────────────────────────
function sdRenderOverview() {
  const s = sdState.summary;
  const pi = sdState.productInfo || {};
  const totalOps = s.total_opinions || 0;

  const reviewCount = Number(pi.review_count) || 0;
  const rating = Number(pi.rating_average) || 0;
  const avg = s.total_reviews ? (totalOps / s.total_reviews).toFixed(1).replace(".", ",") : "0";

  const tiles = [
    {
      icon: "reviews", label: "Review đã phân tích", value: sdNum(s.total_reviews),
      sub: reviewCount > 0 ? `trên ${sdNum(reviewCount)} review của sản phẩm` : "review có nội dung hợp lệ",
    },
    { icon: "opinions", label: "Tổng ý kiến", value: sdNum(totalOps), sub: `trung bình ${avg} ý kiến mỗi review` },
    ...SD_SENTS.map(sent => {
      const d = s.sentiments[sent.id] || { opinions: 0, reviews: 0 };
      return {
        icon: sent.id, cls: `is-${sent.id}`, label: sent.label,
        value: `${sdPct(d.opinions, totalOps)}%`,
        sub: `${sdNum(d.opinions)} ý kiến · trong ${sdNum(d.reviews)} review`,
      };
    }),
    {
      icon: "star", label: "Sao trung bình trên Tiki",
      value: rating > 0 ? `${rating.toFixed(1).replace(".", ",")} / 5` : "—",
      sub: "theo trang sản phẩm",
    },
  ];
  sdEl("sd-kpis").innerHTML = tiles.map(t => `
    <div class="kpi ${t.cls || ""}">
      <div class="kpi-top"><span class="kpi-icon">${sdIcon(t.icon)}</span>${t.label}</div>
      <div class="kpi-val">${t.value}</div>
      <div class="kpi-sub">${t.sub}</div>
    </div>`).join("");

  sdEl("sd-split-title").textContent = `Tỷ lệ ba cảm xúc trên ${sdNum(totalOps)} ý kiến`;
  sdEl("sd-split").innerHTML = SD_SENTS.map(sent => {
    const n = (s.sentiments[sent.id] || {}).opinions || 0;
    return n ? `<div class="sd-seg" style="flex:${n} 1 0px;background:${sent.color}"
                     title="${sent.label}: ${sdNum(n)} ý kiến (${sdPct(n, totalOps)}%)"></div>` : "";
  }).join("");
}

// ─── Cảm xúc theo chủ đề ───────────────────────────────────────────────────
function sdRenderTopics() {
  const s = sdState.summary;
  const key = sdState.sort;
  const ratio = sdState.mode === "ratio";
  const maxTotal = Math.max(1, ...s.topics.map(t => t.total));

  const rows = s.topics.slice().sort((a, b) => {
    if (a.low_data !== b.low_data) return a.low_data ? 1 : -1;
    const va = ratio ? a[key] / a.total : a[key];
    const vb = ratio ? b[key] / b.total : b[key];
    return vb - va || b.total - a.total;
  });

  sdRenderSelector("sd-sort", sdState.sort, false);
  sdEl("sd-mode").innerHTML = [["count", "Số lượng"], ["ratio", "Tỷ lệ %"]].map(([id, label]) =>
    `<button type="button" class="sd-chip" data-value="${id}" aria-pressed="${id === sdState.mode}">${label}</button>`
  ).join("");

  if (!rows.length) {
    sdEl("sd-topics").innerHTML = `<p class="sd-empty">Chưa có ý kiến nào được nhận diện.</p>`;
    return;
  }

  sdEl("sd-topics").innerHTML = rows.map(t => {
    const base = ratio ? t.total : maxTotal;
    const segs = SD_SENTS.map(sent => {
      const n = t[sent.id];
      return n ? `<span class="sd-bar-seg" style="width:${(n / base * 100).toFixed(2)}%;background:${sent.color}"></span>` : "";
    }).join("");
    // Con số của cảm xúc đang sắp xếp được tô đậm, để không phải nhớ thứ tự màu.
    const values = SD_SENTS.map(sent => {
      const text = ratio ? `${sdPct(t[sent.id], t.total)}%` : sdNum(t[sent.id]);
      return `<span class="${sent.id === key ? "is-key" : ""}">${text}</span>`;
    }).join("");
    const tip = SD_SENTS.map(sent => `${sent.label}: ${t[sent.id]}`).join(", ");
    const classes = ["sd-topic-row", t.low_data ? "is-low" : "", t.aspect === sdState.topic ? "is-active" : ""].join(" ");
    return `
      <button type="button" class="${classes}" data-aspect="${escHtml(t.aspect)}"
              aria-pressed="${t.aspect === sdState.topic}" title="${escHtml(t.display_name)} — ${tip}">
        <span class="sd-topic-name">${escHtml(t.display_name)}</span>
        <span class="sd-bar">${segs}</span>
        <span class="sd-topic-val">${values}</span>
      </button>`;
  }).join("");
}

// ─── Tỷ lệ ba cảm xúc theo kỳ (cột xếp chồng 100%) ─────────────────────────
// Tách ngày "YYYY-MM-DD" (đã theo giờ Việt Nam từ server) thành các thành phần.
function sdParseDated(opinions) {
  return opinions.filter(o => o.date).map(o => {
    const [y, m, d] = o.date.split("-").map(Number);
    const weekday = (new Date(Date.UTC(y, m - 1, d)).getUTCDay() + 6) % 7;
    return { y, m, d, weekday, aspect: o.aspect, sentiment: o.sentiment };
  });
}

function sdYearFilter() { return sdState.dim === "month" || sdState.dim === "day" ? sdState.year : "all"; }
function sdMonthFilter() { return sdState.dim === "day" ? sdState.month : "all"; }

// Trục của biểu đồ theo chế độ và bộ lọc đang chọn.
function sdTrendKeys() {
  const dated = sdState.dated;
  if (!dated.length) return [];
  const range = (from, to) => Array.from({ length: to - from + 1 }, (_, i) => String(from + i));
  if (sdState.dim === "year") {
    const years = dated.map(o => o.y);
    return range(Math.min(...years), Math.max(...years));
  }
  if (sdState.dim === "month") return range(1, 12);
  if (sdState.dim === "weekday") return range(0, 6);
  const year = sdYearFilter(), month = sdMonthFilter();
  if (month === "all") return range(1, 31);
  // Chưa chọn năm thì lấy năm nhuận để tháng 2 có đủ 29 ngày.
  return range(1, new Date(year === "all" ? 2024 : Number(year), Number(month), 0).getDate());
}

function sdTrend() {
  const keys = sdTrendKeys();
  const year = sdYearFilter(), month = sdMonthFilter();
  const field = { year: "y", month: "m", day: "d", weekday: "weekday" }[sdState.dim];
  const empty = () => ({ positive: 0, neutral: 0, negative: 0 });
  const totals = {}, byTopic = {};
  for (const o of sdState.dated) {
    if (year !== "all" && o.y !== Number(year)) continue;
    if (month !== "all" && o.m !== Number(month)) continue;
    const key = String(o[field]);
    (totals[key] = totals[key] || empty())[o.sentiment] += 1;
    const topic = (byTopic[o.aspect] = byTopic[o.aspect] || {});
    (topic[key] = topic[key] || empty())[o.sentiment] += 1;
  }
  const buckets = keys.map(key => {
    const c = totals[key] || empty();
    return { key, ...c, total: c.positive + c.neutral + c.negative };
  });
  const heatmap = sdState.summary.topics.slice(0, SD_HEAT_TOPICS).map(t => {
    const row = { aspect: t.aspect, display_name: t.display_name };
    for (const sent of SD_SENTS) row[sent.id] = keys.map(key => ((byTopic[t.aspect] || {})[key] || empty())[sent.id]);
    return row;
  });
  return { keys, buckets, heatmap, total: buckets.reduce((sum, b) => sum + b.total, 0) };
}

// Tên chế độ kèm bộ lọc, dùng cho tiêu đề hai biểu đồ.
function sdTrendName() {
  const year = sdYearFilter(), month = sdMonthFilter();
  if (sdState.dim === "month") return year === "all" ? "tháng trong năm" : `tháng trong năm ${year}`;
  if (sdState.dim === "day") {
    if (month !== "all") return `ngày trong tháng ${month}${year === "all" ? "" : "/" + year}`;
    return year === "all" ? "ngày trong tháng" : `ngày trong tháng, năm ${year}`;
  }
  return SD_DIM_BY_ID[sdState.dim].name;
}

function sdRenderTrendControls() {
  sdEl("sd-dim").innerHTML = SD_DIMS.map(d =>
    `<button type="button" class="sd-chip" data-value="${d.id}" aria-pressed="${d.id === sdState.dim}">${d.label}</button>`
  ).join("");

  const perYear = {};
  for (const o of sdState.dated) perYear[o.y] = (perYear[o.y] || 0) + 1;
  const yearSel = sdEl("sd-year");
  yearSel.innerHTML = `<option value="all">Tất cả năm</option>` +
    Object.keys(perYear).sort().reverse().map(y => `<option value="${y}">${y} (${sdNum(perYear[y])} ý kiến)</option>`).join("");
  yearSel.value = sdState.year;

  const monthSel = sdEl("sd-month");
  monthSel.innerHTML = `<option value="all">Tất cả tháng</option>` +
    Array.from({ length: 12 }, (_, i) => `<option value="${i + 1}">Tháng ${i + 1}</option>`).join("");
  monthSel.value = sdState.month;

  sdEl("sd-year-wrap").classList.toggle("hidden", !(sdState.dim === "month" || sdState.dim === "day"));
  sdEl("sd-month-wrap").classList.toggle("hidden", sdState.dim !== "day");
  document.querySelectorAll(".sd-period-name").forEach(n => { n.textContent = sdTrendName(); });
}

function sdRenderPeriods() {
  const dim = SD_DIM_BY_ID[sdState.dim];
  const trend = sdTrend();
  const periods = trend.buckets;
  const el = sdEl("sd-months");
  sdRenderTrendControls();

  if (periods.length && !trend.total) {
    el.innerHTML = `<p class="sd-empty">Không có ý kiến nào trong khoảng thời gian đã chọn.</p>`;
    return;
  }
  if (!periods.length) {
    el.innerHTML = `<p class="sd-empty">Review của sản phẩm này không có ngày viết nên chưa vẽ được xu hướng.</p>`;
    return;
  }
  // Tiêu cực nằm trên cùng, tích cực sát trục.
  const order = [SD_SENT_BY_ID.negative, SD_SENT_BY_ID.neutral, SD_SENT_BY_ID.positive];
  const maxTotal = Math.max(1, ...periods.map(p => p.total));
  const dense = periods.length > SD_DENSE_COLS ? " is-dense" : "";
  el.style.setProperty("--sd-cols", periods.length);
  el.innerHTML = `
    <div class="sd-cols${dense}">
      ${periods.map(p => `
        <div class="sd-col${p.total < SD_LOW_PERIOD ? " is-low" : ""}"${p.total ? "" : ` title="${dim.full(p.key)}: chưa có ý kiến"`}>
          ${order.map(sent => {
            const n = p[sent.id];
            if (!n) return "";
            const pct = sdPct(n, p.total);
            return `<div class="sd-col-seg" style="flex:${n} 1 0px;background:${sent.color};color:${sent.ink}"
                         title="${dim.full(p.key)} · ${sent.label}: ${n} ý kiến (${pct}%)">${pct >= 9 ? pct + "%" : ""}</div>`;
          }).join("")}
        </div>`).join("")}
    </div>
    <div class="sd-cols sd-vol${dense}" aria-hidden="true">
      ${periods.map(p => `<span style="height:${p.total ? Math.max(4, Math.round(p.total / maxTotal * 100)) : 0}%"></span>`).join("")}
    </div>
    <div class="sd-cols sd-axis${dense}">
      ${periods.map(p => `<div title="${dim.full(p.key)}: ${sdNum(p.total)} ý kiến"><strong>${dim.tick(p.key)}</strong><span>${sdNum(p.total)}${periods.length > 8 ? "" : " ý kiến"}</span></div>`).join("")}
    </div>`;
}

// ─── Bảng nhiệt chủ đề × kỳ ────────────────────────────────────────────────
function sdRenderHeat() {
  const trend = sdTrend();
  const dim = SD_DIM_BY_ID[sdState.dim];
  const heat = { months: trend.keys, rows: trend.heatmap };
  const sent = SD_SENT_BY_ID[sdState.heat];
  sdRenderSelector("sd-heat-sel", sdState.heat, false);
  sdEl("sd-heat-label").textContent = sent.lower;

  const el = sdEl("sd-heat");
  if (!heat.months.length || !heat.rows.length) {
    el.innerHTML = `<p class="sd-empty">Chưa đủ dữ liệu theo thời gian.</p>`;
    return;
  }
  if (!trend.total) {
    el.innerHTML = `<p class="sd-empty">Không có ý kiến nào trong khoảng thời gian đã chọn.</p>`;
    return;
  }
  const ramp = SD_RAMPS[sent.id];
  const max = Math.max(1, ...heat.rows.flatMap(r => r[sent.id]));
  const cell = (v, period, name) => {
    const tip = escHtml(`${name} · ${dim.full(period)}: ${v} ý kiến ${sent.lower}`);
    if (!v) return `<td class="sd-heat-zero" title="${tip}">0</td>`;
    const step = Math.min(4, Math.ceil((v / max) * 5) - 1);
    const fg = step >= ramp.white ? "#ffffff" : "#0b0b0b";
    return `<td style="background:${ramp.bg[step]};color:${fg}" title="${tip}">${v}</td>`;
  };
  el.innerHTML = `
    <table class="sd-heat-table">
      <thead><tr><th></th>${heat.months.map(m => `<th title="${dim.full(m)}">${dim.id === "weekday" ? dim.tick(m).replace("Thứ ", "T") : dim.tick(m)}</th>`).join("")}</tr></thead>
      <tbody>
        ${heat.rows.map(r => `
          <tr><th scope="row">${escHtml(r.display_name)}</th>
          ${r[sent.id].map((v, i) => cell(v, heat.months[i], r.display_name)).join("")}</tr>`).join("")}
      </tbody>
    </table>`;
}

// ─── Bảng "Khách nói gì" ───────────────────────────────────────────────────
function sdFilteredOpinions() {
  const q = sdState.query.trim().toLowerCase();
  return (sdState.summary.opinions || []).filter(o =>
    (sdState.table === "all" || o.sentiment === sdState.table) &&
    (sdState.topic === "all" || o.aspect === sdState.topic) &&
    (sdState.stars === "all" || String(o.rating) === sdState.stars) &&
    (!q || o.sentence.toLowerCase().includes(q))
  );
}

function sdRenderTopicFilter() {
  const sel = sdEl("sd-topic-filter");
  sel.innerHTML = `<option value="all">Tất cả chủ đề</option>` +
    sdState.summary.topics.map(t =>
      `<option value="${escHtml(t.aspect)}">${escHtml(t.display_name)} (${t.total})</option>`).join("");
  sel.value = sdState.topic;
}

function sdRenderTable() {
  sdRenderSelector("sd-table-sel", sdState.table, true);

  const activeTopic = sdState.summary.topics.find(t => t.aspect === sdState.topic);
  sdEl("sd-active-filter").innerHTML = activeTopic ? `
    <button type="button" class="sd-filter-chip" data-clear="topic" aria-label="Bỏ lọc chủ đề ${escHtml(activeTopic.display_name)}">
      Đang lọc: ${escHtml(activeTopic.display_name)}
      <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12"/><path d="M18 6L6 18"/></svg>
    </button>` : "";

  const rows = sdFilteredOpinions();
  const pages = Math.max(1, Math.ceil(rows.length / SD_PAGE_SIZE));
  sdState.page = Math.min(Math.max(1, sdState.page), pages);
  const start = (sdState.page - 1) * SD_PAGE_SIZE;
  const pageRows = rows.slice(start, start + SD_PAGE_SIZE);

  sdEl("sd-table-body").innerHTML = pageRows.length
    ? pageRows.map(o => {
        const sent = SD_SENT_BY_ID[o.sentiment];
        return `
          <tr>
            <td class="sd-td-topic">${escHtml(o.display_name)}</td>
            <td class="sd-td-sent">${sdDot(sent.color)}${sent.label}</td>
            <td class="sd-td-text">${sdHighlight(o.sentence, sdState.query)}</td>
            <td class="sd-td-num">${o.rating ? o.rating + " sao" : "—"}</td>
            <td class="sd-td-num">${sdDateLabel(o.date)}</td>
          </tr>`;
      }).join("")
    : `<tr><td colspan="5" class="sd-empty">Không có ý kiến nào khớp bộ lọc.</td></tr>`;

  sdEl("sd-table-count").textContent = rows.length
    ? `Đang xem ${start + 1}–${start + pageRows.length} trên ${sdNum(rows.length)} ý kiến`
    : "0 ý kiến";
  sdEl("sd-prev").disabled = sdState.page <= 1;
  sdEl("sd-next").disabled = sdState.page >= pages;
}

// Đổi chủ đề đang lọc: cập nhật cả hàng được tô nổi, ô chọn và bảng.
function sdSetTopic(aspect) {
  sdState.topic = aspect;
  sdState.page = 1;
  sdEl("sd-topic-filter").value = aspect;
  sdRenderTopics();
  sdRenderTable();
}

function sdExportCsv() {
  if (!sdState.summary) return;
  const quote = v => `"${String(v == null ? "" : v).replace(/"/g, '""')}"`;
  const lines = [["Chủ đề", "Mã chủ đề", "Cảm xúc", "Khách viết", "Số sao", "Ngày", "Mã review"].map(quote).join(",")]
    .concat(sdFilteredOpinions().map(o => [
      o.display_name, o.aspect, SD_SENT_BY_ID[o.sentiment].label, o.sentence, o.rating || "", o.date, o.review_id,
    ].map(quote).join(",")));
  // BOM để Excel đọc đúng tiếng Việt.
  const blob = new Blob(["﻿" + lines.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `y-kien-${(sdState.productInfo || {}).product_id || "san-pham"}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ─── Điểm vào ──────────────────────────────────────────────────────────────
function renderSellerDashboard(data) {
  const summary = data && data.seller_summary;
  if (!summary) throw new Error("Kết quả phân tích thiếu phần tổng hợp ý kiến");
  Object.assign(sdState, {
    summary, productInfo: data.product_info || {},
    sort: "negative", mode: "count", heat: "negative",
    dim: "year", year: "all", month: "all", dated: sdParseDated(summary.opinions || []),
    table: "all", topic: "all", stars: "all", query: "", page: 1,
  });
  sdEl("sd-search").value = "";
  sdEl("sd-star-filter").value = "all";
  sdRenderOverview();
  sdRenderTopics();
  sdRenderPeriods();
  sdRenderHeat();
  sdRenderTopicFilter();
  sdRenderTable();
}

function initSellerDashboard() {
  const ready = () => !!sdState.summary;
  sdBindSelector("sd-sort", v => { if (ready()) { sdState.sort = v; sdRenderTopics(); } });
  sdBindSelector("sd-mode", v => { if (ready()) { sdState.mode = v; sdRenderTopics(); } });
  sdBindSelector("sd-heat-sel", v => { if (ready()) { sdState.heat = v; sdRenderHeat(); } });
  const rerenderTrend = () => { sdRenderPeriods(); sdRenderHeat(); };
  sdBindSelector("sd-dim", v => { if (ready()) { sdState.dim = v; rerenderTrend(); } });
  sdEl("sd-year").addEventListener("change", (e) => { if (ready()) { sdState.year = e.target.value; rerenderTrend(); } });
  sdEl("sd-month").addEventListener("change", (e) => { if (ready()) { sdState.month = e.target.value; rerenderTrend(); } });
  sdBindSelector("sd-table-sel", v => { if (ready()) { sdState.table = v; sdState.page = 1; sdRenderTable(); } });

  sdEl("sd-topics").addEventListener("click", (e) => {
    const row = e.target.closest("button[data-aspect]");
    if (!row || !ready()) return;
    // Bấm lại chủ đề đang chọn thì bỏ lọc.
    const next = row.dataset.aspect === sdState.topic ? "all" : row.dataset.aspect;
    sdSetTopic(next);
    if (next !== "all") sdEl("sec-opinions").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  sdEl("sd-active-filter").addEventListener("click", (e) => {
    if (ready() && e.target.closest("button[data-clear]")) sdSetTopic("all");
  });
  sdEl("sd-topic-filter").addEventListener("change", (e) => { if (ready()) sdSetTopic(e.target.value); });
  sdEl("sd-star-filter").addEventListener("change", (e) => {
    if (!ready()) return;
    sdState.stars = e.target.value; sdState.page = 1; sdRenderTable();
  });
  sdEl("sd-search").addEventListener("input", (e) => {
    if (!ready()) return;
    sdState.query = e.target.value; sdState.page = 1; sdRenderTable();
  });
  sdEl("sd-prev").addEventListener("click", () => { if (ready()) { sdState.page -= 1; sdRenderTable(); } });
  sdEl("sd-next").addEventListener("click", () => { if (ready()) { sdState.page += 1; sdRenderTable(); } });
  sdEl("sd-export").addEventListener("click", sdExportCsv);
}

initSellerDashboard();
