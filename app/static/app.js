"use strict";
// Khung trang: chuyển màn hình, tiến trình phân tích, thẻ sản phẩm, chatbot.
// Phần dashboard nằm ở seller_dashboard.js.

let currentProductId = null;

// ─── Tiện ích ──────────────────────────────────────────────────────────────
function escHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Tiki chặn hotlink ảnh nên phải đi qua proxy của server.
function proxyImg(url) {
  return url ? `/api/proxy-image?url=${encodeURIComponent(url)}` : "";
}

// ─── Màn hình: trang chủ / đang phân tích / kết quả ────────────────────────
function setView(view) {
  document.getElementById("landing").classList.toggle("hidden", view !== "landing");
  document.getElementById("loading-view").classList.toggle("hidden", view !== "loading");
  document.getElementById("result-root").classList.toggle("hidden", view !== "result");
  document.getElementById("chat-fab").classList.toggle("hidden", view !== "result");
  if (view !== "result") setChatOpen(false);
}

function showError(message) {
  document.getElementById("error-text").textContent = message;
  document.getElementById("error-banner").classList.remove("hidden");
}
function hideError() {
  document.getElementById("error-banner").classList.add("hidden");
}

// ─── Tiến trình phân tích ──────────────────────────────────────────────────
// Mốc phần trăm khớp với các bước server báo về trong DashboardService.
function setProgress(pct, msg) {
  const value = Math.max(0, Math.min(100, Number(pct) || 0));
  document.getElementById("load-bar").style.width = `${value}%`;
  document.getElementById("load-bar-wrap").setAttribute("aria-valuenow", String(Math.round(value)));
  document.getElementById("load-pct").textContent = `${Math.round(value)}%`;
  if (msg) document.getElementById("load-msg").textContent = msg;

  const active = value < 30 ? 1 : value < 82 ? 2 : 3;
  document.querySelectorAll("#load-steps li").forEach(li => {
    const step = Number(li.dataset.step);
    li.classList.toggle("is-done", step < active);
    li.classList.toggle("is-active", step === active);
  });
}

// ─── Render: Product Hero ──────────────────────────────────────────────────
function renderHeader(data) {
  const pi  = data.product_info || {};
  
  // Tên sản phẩm
  const inputUrl = (document.getElementById("product-url") || {}).value || "";
  let productName = pi.name && pi.name !== "nan" && pi.name !== "None" ? pi.name : "";
  if (!productName || productName === "—") {
    const m2 = inputUrl.match(/tiki\.vn\/(.+?)-p\d+/);
    if (m2) productName = m2[1].replace(/-/g, " ").replace(/\b\w/g, l => l.toUpperCase());
  }
  if (!productName || productName === "—") productName = "Sản phẩm #" + (pi.product_id || "");

  const nameEl = document.getElementById("product-name");
  if (nameEl) nameEl.textContent = productName;

  // Product ID + meta chips
  const chips = [];
  
  if (pi.product_id && pi.product_id !== "nan" && pi.product_id !== "None") {
    chips.push('<span class="meta-chip product-id">🔖 Mã SP: ' + escHtml(String(pi.product_id)) + '</span>');
  } else {
    chips.push('<span class="meta-chip product-id">🔖 Mã SP: Không xác định</span>');
  }
  
  if (pi.brand_name && pi.brand_name !== "nan" && pi.brand_name !== "None" && pi.brand_name !== "") {
    chips.push('<span class="meta-chip brand">🏢 ' + escHtml(pi.brand_name) + '</span>');
  }
  
  if (pi.seller_name && pi.seller_name !== "nan" && pi.seller_name !== "None" && pi.seller_name !== "") {
    chips.push('<span class="meta-chip">🏪 ' + escHtml(pi.seller_name) + '</span>');
  }
  
  if (pi.seller_is_official || pi.is_official) {
    chips.push('<span class="meta-chip official">✓ Official Store</span>');
  }
  
  if (pi.price && Number(pi.price) > 0) {
    const priceFormatted = Number(pi.price).toLocaleString("vi-VN") + " ₫";
    chips.push('<span class="meta-chip price">💰 ' + priceFormatted + '</span>');
  }
  
  if (pi.original_price && Number(pi.original_price) > Number(pi.price)) {
    const origFormatted = Number(pi.original_price).toLocaleString("vi-VN") + " ₫";
    const discount = Math.round((1 - Number(pi.price) / Number(pi.original_price)) * 100);
    chips.push('<span class="meta-chip original-price">~~ ' + origFormatted + ' ~~</span>');
    chips.push('<span class="meta-chip discount">-' + discount + '%</span>');
  }
  
  if (pi.sold_quantity && Number(pi.sold_quantity) > 0) {
    const soldFormatted = Number(pi.sold_quantity).toLocaleString("vi-VN");
    chips.push('<span class="meta-chip sold">✓ Đã bán: ' + soldFormatted + '</span>');
  } else if (pi.sold_text && pi.sold_text !== "nan" && pi.sold_text !== "") {
    chips.push('<span class="meta-chip sold">✓ ' + escHtml(pi.sold_text) + '</span>');
  }
  
  if (pi.rating_average && Number(pi.rating_average) > 0) {
    chips.push('<span class="meta-chip rating">⭐ ' + Number(pi.rating_average).toFixed(1) + ' sao</span>');
  }
  
  if (pi.review_count && Number(pi.review_count) > 0) {
    chips.push('<span class="meta-chip review-count">📝 ' + Number(pi.review_count).toLocaleString("vi-VN") + ' đánh giá</span>');
  }
  
  const linkHref = (pi.product_url && pi.product_url.startsWith("https")) ? pi.product_url : inputUrl;
  if (linkHref && linkHref.includes("tiki.vn")) {
    chips.push('<a class="meta-chip link" href="' + escHtml(linkHref) + '" target="_blank" rel="noopener">🔗 Xem trên Tiki</a>');
  }

  const metaEl = document.getElementById("product-meta");
  if (metaEl) {
    metaEl.innerHTML = chips.length ? chips.join("") : '<span style="color:var(--text-3)">⚠️ Không có thông tin chi tiết</span>';
  }

  // Ảnh sản phẩm
  const imgWrap = document.getElementById("product-img-wrap");
  if (imgWrap) {
    const rawUrl = pi.thumbnail_url || pi.image_url || "";
    if (rawUrl && rawUrl !== "nan" && rawUrl !== "") {
      imgWrap.innerHTML = '<img id="main-product-img" src="' + escHtml(proxyImg(rawUrl)) + '" alt="product" loading="lazy" onerror="this.parentElement.innerHTML=\'<span class=product-image-placeholder>🛍️</span>\'"/>';
    } else {
      imgWrap.innerHTML = '<span class="product-image-placeholder">🛍️</span>';
    }
  }

  // Gallery
  const extraImgs = (pi.images || []).filter(img => img && img !== "nan" && img !== "").slice(1, 5);
  const galleryWrap = document.getElementById("product-gallery-wrap");
  if (galleryWrap && extraImgs.length) {
    galleryWrap.innerHTML = '<div class="product-gallery">' +
      extraImgs.map(u => '<div class="gallery-thumb" onclick="swapMainImg(this,\'' + escHtml(proxyImg(u)) + '\')"><img src="' + escHtml(proxyImg(u)) + '" loading="lazy"/></div>').join("") +
    '</div>';
  } else if (galleryWrap) {
    galleryWrap.innerHTML = "";
  }

  // Short description
  const shortDesc = pi.short_description || pi.description || "";
  const shortDescEl = document.getElementById("product-short-desc");
  if (shortDescEl && shortDesc && shortDesc !== "nan" && shortDesc !== "") {
    shortDescEl.style.display = "block";
    shortDescEl.innerHTML = escHtml(shortDesc.substring(0, 200)) + (shortDesc.length > 200 ? "..." : "");
  } else if (shortDescEl) {
    shortDescEl.style.display = "none";
  }

  // Gia + so sao Tiki
  const price = Number(pi.price) || 0;
  const origPx = Number(pi.original_price) || 0;
  const disc = (origPx > price && origPx > 0 && price > 0) ? Math.round((1 - price / origPx) * 100) : null;
  const rating = pi.rating_average ? Number(pi.rating_average).toFixed(1) : null;
  const soldTxt = pi.sold_text || (pi.sold_quantity > 0 ? Number(pi.sold_quantity).toLocaleString("vi-VN") : "");

  const scoresEl = document.getElementById("product-scores");
  if (scoresEl) {
    let priceHtml = '';
    if (price > 0) {
      priceHtml = `
        <div class="price-current">
          <span class="price-value">${price.toLocaleString("vi-VN")}đ</span>
          ${disc ? '<span class="discount-badge">-' + disc + '%</span>' : ''}
        </div>
      `;
      if (origPx > price && origPx > 0) {
        priceHtml += '<div class="price-original">' + origPx.toLocaleString("vi-VN") + 'đ</div>';
      }
    }
    
    scoresEl.innerHTML = `
      <div class="product-price-panel">
        ${priceHtml}
        ${soldTxt ? '<div class="sold-count">✓ Đã bán: ' + soldTxt + '</div>' : ''}
        <div style="margin-top:10px;padding-top:10px;border-top:1px solid var(--line)">
          <div style="font-size:11px;color:var(--text-3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px">Đánh giá Tiki</div>
          <div style="font-size:26px;font-weight:800;color:var(--amber)">${rating ? '⭐ ' + rating : '—'}<span style="font-size:14px;color:var(--text-3)">/5</span></div>
        </div>
      </div>
    `;
  }
}

function swapMainImg(thumbEl, newSrc) {
  const mainImg = document.getElementById("main-product-img");
  if (mainImg) mainImg.src = newSrc;
  document.querySelectorAll(".gallery-thumb").forEach(t => t.classList.remove("active"));
  thumbEl.classList.add("active");
}

// ─── Chatbot ───────────────────────────────────────────────────────────────
function setChatOpen(open) {
  const panel = document.getElementById("chat-panel");
  const fab = document.getElementById("chat-fab");
  panel.classList.toggle("hidden", !open);
  document.body.classList.toggle("chat-open", open);
  fab.setAttribute("aria-expanded", String(open));
  fab.setAttribute("aria-label", open ? "Đóng trợ lý review" : "Mở trợ lý review");
  if (open) document.getElementById("chat-question").focus();
}

function resetChatFor(data) {
  const nextId = (data.product_info || {}).product_id || null;
  if (nextId !== currentProductId) {
    // Sản phẩm mới: xóa hội thoại cũ để câu trả lời không lẫn sản phẩm.
    document.getElementById("chat-output").innerHTML =
      '<div class="assistant-message muted">Hãy đặt câu hỏi về review của sản phẩm đang xem.</div>';
  }
  currentProductId = nextId;
}

function answerHtml(text) {
  return escHtml(text).replace(/\n/g, "<br>");
}

function aspectLabel(aspect) {
  const labels = {
    "PRODUCT#SAFETY": "Độ an toàn",
    "PRODUCT#MATERIAL": "Chất liệu",
    "PRODUCT#SCENT": "Mùi sản phẩm",
    "PRODUCT#SIZE": "Kích thước",
    "PRODUCT#ABSORPTION": "Khả năng thấm hút",
    "PRODUCT#FUNCTION": "Công năng",
    "PRODUCT#QUALITY": "Chất lượng",
    "PRODUCT#VALUE": "Độ đáng tiền",
    "PRODUCT#DURABILITY": "Độ bền",
    "PRODUCT#COMFORT": "Độ thoải mái",
    "PRICE#AFFORDABILITY": "Giá cả",
    "DELIVERY#SPEED": "Giao hàng",
    "DELIVERY#PACKAGING": "Bao bì",
    "DELIVERY#ACCURACY": "Giao đúng/đủ hàng",
    "SELLER#AUTHENTICITY": "Tính chính hãng",
  };
  return labels[aspect] || aspect;
}

async function askAssistant(question) {
  if (!currentProductId) {
    showError("Hãy phân tích một sản phẩm trước khi hỏi trợ lý.");
    return;
  }
  const output = document.getElementById("chat-output");
  const pendingId = `chat-pending-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  output.insertAdjacentHTML("beforeend", `<div class="user-message">${escHtml(question)}</div>`);
  output.insertAdjacentHTML("beforeend", `<div class="assistant-message muted" id="${pendingId}">Đang tìm bằng chứng...</div>`);
  output.scrollTop = output.scrollHeight;

  try {
    const resp = await fetch("/api/assistant/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product_id: currentProductId, question, use_llm: true }),
    });
    if (!resp.ok) {
      let msg = "Không thể trả lời câu hỏi này";
      try { msg = (await resp.json()).detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    const data = await resp.json();
    const structured = data.structured_answer || {};
    const citations = (structured.citations && structured.citations.length)
      ? structured.citations
      : (data.evidence || []).map(e => ({ aspect: e.aspect_label || e.aspect || "Bằng chứng", evidence: e.text || "" }));
    const evidenceHtml = citations.slice(0, 3).map(e =>
      `<li><span>${escHtml(aspectLabel(e.aspect || "Bằng chứng"))}</span>${escHtml((e.evidence || "").slice(0, 180))}</li>`
    ).join("");
    document.getElementById(pendingId)?.remove();
    output.insertAdjacentHTML("beforeend", `
      <div class="assistant-message">
        <p>${answerHtml(data.answer || "Chưa có câu trả lời.")}</p>
        ${evidenceHtml ? `<div class="chat-evidence-title">Bằng chứng tham khảo</div><ul class="chat-evidence">${evidenceHtml}</ul>` : ""}
      </div>`);
  } catch (err) {
    document.getElementById(pendingId)?.remove();
    output.insertAdjacentHTML("beforeend", `<div class="assistant-message error">${escHtml(err.message)}</div>`);
  } finally {
    output.scrollTop = output.scrollHeight;
  }
}

// ─── Phân tích (tiến trình qua SSE) ────────────────────────────────────────
async function analyze() {
  const url = document.getElementById("product-url").value.trim();
  if (!url) {
    showError("Hãy dán link sản phẩm Tiki vào ô phía trên.");
    document.getElementById("product-url").focus();
    return;
  }

  const btn = document.getElementById("analyze-btn");
  const previousView = document.getElementById("result-root").classList.contains("hidden") ? "landing" : "result";
  btn.disabled = true;
  hideError();
  setProgress(0, "Gửi yêu cầu tới server");
  setView("loading");
  document.getElementById("loading-view").scrollIntoView({ block: "start" });

  let es = null;
  try {
    const startResp = await fetch("/api/analyze/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ product_url: url }),
    });
    if (!startResp.ok) {
      let errMsg = "Không thể bắt đầu phân tích";
      try { errMsg = (await startResp.json()).detail || errMsg; } catch (_) {}
      throw new Error(errMsg);
    }
    const { job_id } = await startResp.json();

    const data = await new Promise((resolve, reject) => {
      es = new EventSource(`/api/analyze/progress/${job_id}`);
      // Sản phẩm nhiều review: tải review, chạy PhoBERT và ghi vector có thể mất vài phút.
      const timeout = setTimeout(() => {
        reject(new Error("Phân tích quá 10 phút. Hãy thử lại."));
      }, 10 * 60 * 1000);

      es.onmessage = (event) => {
        let payload;
        try { payload = JSON.parse(event.data); } catch (_) { return; }
        setProgress(payload.progress ?? 0, payload.message ?? "");
        if (payload.status === "done") {
          clearTimeout(timeout);
          resolve(payload.data);
        } else if (payload.status === "error") {
          clearTimeout(timeout);
          reject(new Error(payload.error || "Lỗi không xác định từ server"));
        }
      };
      es.onerror = () => {
        clearTimeout(timeout);
        reject(new Error("Mất kết nối với server. Hãy kiểm tra server còn chạy không."));
      };
    });

    renderAll(data);
  } catch (err) {
    setView(previousView);
    showError(`Không phân tích được: ${err.message}`);
  } finally {
    if (es) { try { es.close(); } catch (_) {} }
    btn.disabled = false;
  }
}

function renderAll(data) {
  if (!data) throw new Error("Server không trả về dữ liệu");
  renderHeader(data);
  resetChatFor(data);
  renderSellerDashboard(data);
  setView("result");
  document.getElementById("result-root").scrollIntoView({ block: "start" });
  updateActiveSection();
}

// ─── Điều hướng trong trang ────────────────────────────────────────────────
const navLinks = Array.from(document.querySelectorAll(".section-nav a"));
function updateActiveSection() {
  const offset = 90; // chiều cao thanh điều hướng dính
  let current = navLinks[0];
  for (const a of navLinks) {
    const section = document.querySelector(a.getAttribute("href"));
    if (section && section.getBoundingClientRect().top - offset <= 0) current = a;
  }
  navLinks.forEach(a => a.classList.toggle("is-active", a === current));
}
let scrollTicking = false;
window.addEventListener("scroll", () => {
  if (scrollTicking) return;
  scrollTicking = true;
  requestAnimationFrame(() => { updateActiveSection(); scrollTicking = false; });
}, { passive: true });

// ─── Sự kiện ───────────────────────────────────────────────────────────────
document.getElementById("search-form").addEventListener("submit", (e) => {
  e.preventDefault();
  analyze();
});
document.getElementById("error-close").addEventListener("click", hideError);

document.querySelectorAll(".sample-card").forEach(card => {
  card.addEventListener("click", () => {
    // Server nhận cả mã sản phẩm trần trong ô link.
    document.getElementById("product-url").value = card.dataset.productId;
    analyze();
  });
});

document.getElementById("chat-fab").addEventListener("click", () => {
  setChatOpen(document.getElementById("chat-panel").classList.contains("hidden"));
});
document.getElementById("chat-close").addEventListener("click", () => setChatOpen(false));

document.querySelectorAll(".quick-questions button").forEach(btn => {
  btn.addEventListener("click", () => askAssistant(btn.dataset.question || btn.textContent.trim()));
});
document.getElementById("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = document.getElementById("chat-question");
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  askAssistant(question);
});
