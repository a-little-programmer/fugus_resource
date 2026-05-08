const form = document.querySelector("#search-form");
const queryInput = document.querySelector("#query");
const topKInput = document.querySelector("#top-k");
const thresholdInput = document.querySelector("#threshold");
const resultsBody = document.querySelector("#results");
const messageEl = document.querySelector("#message");
const statusEl = document.querySelector("#status");
const summaryEl = document.querySelector("#summary");

const sourceLabels = {
  exact: "精确匹配",
  ambiguous_alias: "别名多义",
  expanded_exact: "缩写展开",
  ambiguous_abbreviation: "缩写多义",
  embedding: "向量召回",
};

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const status = await response.json();
    statusEl.textContent = `${status.entities} 个实体，${status.name_vectors} 个名称向量，索引后端 ${status.index_backend}`;
    thresholdInput.value = status.threshold;
  } catch (error) {
    statusEl.textContent = "索引状态加载失败";
  }
}

async function search(query) {
  const params = new URLSearchParams({
    q: query,
    top_k: topKInput.value,
    threshold: thresholdInput.value,
  });
  messageEl.textContent = "检索中...";
  summaryEl.textContent = "";
  resultsBody.innerHTML = `<tr class="empty"><td colspan="6">正在计算候选</td></tr>`;

  try {
    const response = await fetch(`/api/search?${params.toString()}`);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.message || "检索失败");
    }
    renderResults(payload);
  } catch (error) {
    messageEl.textContent = error.message;
    resultsBody.innerHTML = `<tr class="empty"><td colspan="6">请求失败</td></tr>`;
  }
}

function renderResults(payload) {
  messageEl.textContent = translateMessage(payload.message);
  const results = payload.results || [];
  summaryEl.textContent = results.length ? `${results.length} 个候选` : "0 个候选";

  if (!results.length) {
    resultsBody.innerHTML = `<tr class="empty"><td colspan="6">没有结果</td></tr>`;
    return;
  }

  resultsBody.innerHTML = results.map((result) => {
    const sourceClass = result.source === "embedding" ? "embedding" : "";
    const ambiguousClass = result.ambiguous ? "warning" : "";
    const status = result.ambiguous
      ? "需要选择"
      : result.low_confidence
        ? "低置信度"
        : "可用";
    const statusClass = result.ambiguous ? "warn" : result.low_confidence ? "warn" : "";

    return `
      <tr>
        <td>
          <strong>${escapeHtml(result.standard_name_cn || "")}</strong>
        </td>
        <td class="latin">${escapeHtml(result.scientific_name || "")}</td>
        <td><span class="badge ${sourceClass} ${ambiguousClass}">${escapeHtml(sourceLabels[result.source] || result.source)}</span></td>
        <td><span class="score">${Number(result.score).toFixed(4)}</span></td>
        <td>${escapeHtml(result.matched_name || "")}</td>
        <td><span class="status ${statusClass}">${status}</span></td>
      </tr>
    `;
  }).join("");
}

function translateMessage(message) {
  const messages = {
    "Exact match found.": "已命中库内精确别名。",
    "Ambiguous exact alias; choose one candidate.": "别名对应多个实体，需要人工选择。",
    "Ambiguous Latin abbreviation; embedding search skipped.": "拉丁缩写存在多义候选，已跳过向量排序。",
    "No exact match found; showing most similar embedding candidates.": "未找到精确匹配，以下为最相似候选。",
  };
  if (message && message.startsWith("Latin abbreviation expanded")) {
    return "已通过库内拉丁名规则展开缩写。";
  }
  return messages[message] || message || "";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (query) {
    search(query);
  }
});

loadStatus();
