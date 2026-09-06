const data = window.EXPERIMENT_DASHBOARD_DATA;

const state = {
  tab: "rag",
  datasets: new Set(),
  models: new Set(),
  mode: "all",
  rag: "all",
  search: "",
  editorEnabled: false
};

const multiSelects = [];

const els = {
  reportMeta: document.getElementById("reportMeta"),
  metrics: document.getElementById("metrics"),
  datasetMulti: document.getElementById("datasetMulti"),
  datasetTrigger: document.getElementById("datasetTrigger"),
  datasetMenu: document.getElementById("datasetMenu"),
  modelMulti: document.getElementById("modelMulti"),
  modelTrigger: document.getElementById("modelTrigger"),
  modelMenu: document.getElementById("modelMenu"),
  modeFilter: document.getElementById("modeFilter"),
  ragFilter: document.getElementById("ragFilter"),
  searchFilter: document.getElementById("searchFilter"),
  ragPairs: document.getElementById("ragPairs"),
  modePairs: document.getElementById("modePairs"),
  matrix: document.getElementById("matrix"),
  runsTable: document.getElementById("runsTable"),
  summaryBars: document.getElementById("summaryBars"),
  qualityPanel: document.getElementById("qualityPanel"),
  ragCount: document.getElementById("ragCount"),
  modeCount: document.getElementById("modeCount"),
  matrixCount: document.getElementById("matrixCount"),
  runsCount: document.getElementById("runsCount"),
  dialog: document.getElementById("detailDialog"),
  detailContent: document.getElementById("detailContent"),
  closeDialog: document.getElementById("closeDialog")
};

function fmt(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(digits);
  return String(value);
}

function signedFmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(digits)}`;
}

function signedPctPoints(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const points = value * 100;
  const sign = points > 0 ? "+" : "";
  return `${sign}${points.toFixed(1)} pp`;
}

function average(values) {
  const clean = values.map(Number).filter(value => Number.isFinite(value));
  if (!clean.length) return null;
  return clean.reduce((sum, value) => sum + value, 0) / clean.length;
}

function pct(value) {
  if (value === null || value === undefined) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function deltaBadge(label, value, digits = 2) {
  const number = Number(value || 0);
  const cls = number > 0 ? "good" : number < 0 ? "bad" : "";
  const sign = number > 0 ? "+" : "";
  return `<span class="badge ${cls}">${label}: ${sign}${number.toFixed(digits)}</span>`;
}

function effectBadge(label, effect) {
  const cls = effect === "helped" ? "good" : effect === "hurt" ? "bad" : "";
  return `<span class="badge ${cls}">${escapeHtml(label)}: ${escapeHtml(effect || "same")}</span>`;
}

function successEffectBadge(effect) {
  return effectBadge("success", effect);
}

function firstTryEffectBadge(effect) {
  return effectBadge("first try", effect);
}

function boolBadge(value, label) {
  return `<span class="badge ${value ? "good" : "bad"}">${label}: ${value ? "yes" : "no"}</span>`;
}

function strategyBadge(record) {
  if (!record.rag_enabled) return "";
  const cls = record.rag_strategy_category === "strategy_used"
    ? "good"
    : record.rag_strategy_category === "strategy_ignored"
      ? "warn"
      : "bad";
  const shortLabel = {
    strategy_used: "used",
    strategy_ignored: "ignored",
    unclear_failed: "unclear"
  }[record.rag_strategy_category] || "unclear";
  return `<span class="badge ${cls}">strategy: ${escapeHtml(shortLabel)}</span>`;
}

function textMatch(record) {
  const q = state.search.trim().toLowerCase();
  if (!q) return true;
  const haystack = [
    record.dataset,
    record.provider,
    record.model,
    record.provider_model,
    record.query_mode,
    record.rag_strategy_label,
    record.feature_notes,
    record.seeding_notes,
    record.target_feature,
    record.visualization_goal
  ].join(" ").toLowerCase();
  return haystack.includes(q);
}

function recordPasses(record, includeMode = true, includeRag = true) {
  if (!setPasses(state.datasets, record.dataset)) return false;
  if (!setPasses(state.models, record.provider_model)) return false;
  if (includeMode && state.mode !== "all" && record.query_mode !== state.mode) return false;
  if (includeRag && !recordMatchesRagCategory(record)) return false;
  return textMatch(record);
}

function setPasses(selected, value) {
  return selected.size === 0 || selected.has(value);
}

function recordMatchesRagCategory(record) {
  return state.rag === "all" || record.rag_strategy_category === state.rag;
}

function ragPairMatchesRagCategory(pair) {
  if (state.rag === "all") return true;
  if (state.rag === "no_rag") return true;
  return pair.rag_on_strategy_category === state.rag;
}

function modePairMatchesRagCategory(pair) {
  if (state.rag === "all") return true;
  if (state.rag === "no_rag") return pair.rag_enabled === false;
  if (!pair.rag_enabled) return false;
  return (
    pair.explorative.rag_strategy_category === state.rag ||
    pair.feature_aware.rag_strategy_category === state.rag
  );
}

function pairTextMatches(...records) {
  return records.some(record => textMatch(record));
}

function ragPairPasses(pair) {
  if (!setPasses(state.datasets, pair.dataset)) return false;
  if (!setPasses(state.models, pair.provider_model)) return false;
  if (state.mode !== "all" && pair.query_mode !== state.mode) return false;
  if (!ragPairMatchesRagCategory(pair)) return false;
  return pairTextMatches(pair.rag_off, pair.rag_on);
}

function modePairPasses(pair) {
  if (!setPasses(state.datasets, pair.dataset)) return false;
  if (!setPasses(state.models, pair.provider_model)) return false;
  if (!modePairMatchesRagCategory(pair)) return false;
  return pairTextMatches(pair.explorative, pair.feature_aware);
}

function filteredRecords() {
  return data.records.filter(record => recordPasses(record));
}

function filteredPrimaryRecords() {
  return data.primary_records.filter(record => recordPasses(record));
}

function filteredRagPairs() {
  return data.rag_pairs.filter(pair => ragPairPasses(pair));
}

function filteredModePairs() {
  return data.mode_pairs.filter(pair => modePairPasses(pair));
}

function averageField(records, field) {
  return average(records.map(record => record[field]));
}

function successRate(records) {
  if (!records.length) return null;
  return records.filter(record => record.succeeded).length / records.length;
}

function successNote(records) {
  return `${records.filter(record => record.succeeded).length} / ${records.length} succeeded`;
}

function firstTryRate(records) {
  if (!records.length) return null;
  return records.filter(record => record.first_try_success).length / records.length;
}

function firstTryNote(records) {
  return `${records.filter(record => record.first_try_success).length} / ${records.length} first try`;
}

function strategyAdoptionRate(records) {
  const used = records.filter(record => record.rag_strategy_category === "strategy_used").length;
  const ignored = records.filter(record => record.rag_strategy_category === "strategy_ignored").length;
  const assessable = used + ignored;
  if (!assessable) return null;
  return used / assessable;
}

function strategyAdoptionNote(records) {
  const used = records.filter(record => record.rag_strategy_category === "strategy_used").length;
  const ignored = records.filter(record => record.rag_strategy_category === "strategy_ignored").length;
  const unclear = records.filter(record => record.rag_strategy_category === "unclear_failed").length;
  return `${used} used / ${ignored} ignored / ${unclear} unclear`;
}

function effectDeltaNote(pairs, field) {
  const helped = pairs.filter(pair => pair[field] === "helped").length;
  const same = pairs.filter(pair => pair[field] === "same").length;
  const hurt = pairs.filter(pair => pair[field] === "hurt").length;
  return `${helped} helped / ${same} same / ${hurt} hurt`;
}

function populateFilters() {
  const datasets = data.dimensions.datasets;
  const models = data.dimensions.provider_models.map(item => item.label);
  const modes = ["all", ...data.dimensions.query_modes];
  buildMultiSelect({
    multi: els.datasetMulti,
    trigger: els.datasetTrigger,
    menu: els.datasetMenu,
    values: datasets,
    selected: state.datasets,
    allLabel: "All datasets",
    singularLabel: "dataset",
    pluralLabel: "datasets"
  });
  buildMultiSelect({
    multi: els.modelMulti,
    trigger: els.modelTrigger,
    menu: els.modelMenu,
    values: models,
    selected: state.models,
    allLabel: "All models",
    singularLabel: "model",
    pluralLabel: "models"
  });
  fillSelect(els.modeFilter, modes, "All modes");
}

function buildMultiSelect(config) {
  multiSelects.push(config);
  config.trigger.addEventListener("click", event => {
    event.stopPropagation();
    const shouldOpen = !config.multi.classList.contains("open");
    closeMultiSelects(config);
    config.multi.classList.toggle("open", shouldOpen);
    config.trigger.setAttribute("aria-expanded", String(shouldOpen));
  });
  config.menu.addEventListener("click", event => event.stopPropagation());
  renderMultiSelect(config);
}

function renderMultiSelect(config) {
  const selected = config.selected;
  const selectedValues = [...selected];
  const allChecked = selected.size === 0;
  config.trigger.textContent = allChecked
    ? config.allLabel
    : selected.size === 1
      ? selectedValues[0]
      : `${selected.size} ${config.pluralLabel} selected`;
  config.trigger.title = allChecked ? config.allLabel : selectedValues.join(", ");
  config.menu.innerHTML = [
    optionHtml("__all__", config.allLabel, allChecked, true),
    ...config.values.map(value => optionHtml(value, value, selected.has(value), false))
  ].join("");
  config.menu.querySelectorAll("input[type='checkbox']").forEach(input => {
    input.addEventListener("change", () => {
      const value = input.dataset.value;
      if (value === "__all__") {
        selected.clear();
      } else if (input.checked) {
        selected.add(value);
      } else {
        selected.delete(value);
      }
      renderMultiSelect(config);
      renderActive();
    });
  });
}

function optionHtml(value, label, checked, isAll) {
  return `
    <label class="multi-option ${isAll ? "all" : ""}">
      <input type="checkbox" data-value="${escapeAttr(value)}" ${checked ? "checked" : ""}>
      <span>${escapeHtml(label)}</span>
    </label>
  `;
}

function closeMultiSelects(except = null) {
  for (const config of multiSelects) {
    if (config === except) continue;
    config.multi.classList.remove("open");
    config.trigger.setAttribute("aria-expanded", "false");
  }
}

function fillSelect(select, values, allLabel) {
  select.innerHTML = values.map(value => {
    const label = value === "all" ? allLabel : value;
    return `<option value="${escapeAttr(value)}">${escapeHtml(label)}</option>`;
  }).join("");
}

function renderMeta() {
  const editor = state.editorEnabled ? " - editor active" : "";
  els.reportMeta.textContent = `Generated ${new Date(data.generated_at).toLocaleString()} - primary: ${data.primary_strategy}${editor}`;
}

function renderMetrics() {
  const records = filteredRecords();
  const primaryRecords = filteredPrimaryRecords();
  const ragPairs = filteredRagPairs();
  const modePairs = filteredModePairs();
  const ragDeltaSuccess = average(ragPairs.map(pair => pair.delta_success));
  const ragDeltaFirstTry = average(ragPairs.map(pair => pair.delta_first_try_success));
  const ragDeltaSeeding = average(ragPairs.map(pair => pair.delta_seeding));
  const ragDeltaCoverage = average(ragPairs.map(pair => pair.delta_observed_feature_coverage));
  const modeDeltaSeeding = average(modePairs.map(pair => pair.delta_seeding));
  const metrics = [
    {label: "Raw runs", value: records.length, note: "matching filters"},
    {label: "Strategy adoption", value: pct(strategyAdoptionRate(primaryRecords)), note: strategyAdoptionNote(primaryRecords)},
    {label: "Success", value: pct(successRate(primaryRecords)), note: successNote(primaryRecords)},
    {label: "First-try success", value: pct(firstTryRate(primaryRecords)), note: firstTryNote(primaryRecords)},
    {label: "Avg coverage", value: pct(averageField(primaryRecords, "observed_feature_coverage")), note: "selected conditions"},
    {label: "Avg seeding", value: fmt(averageField(primaryRecords, "seeding_score")), note: "selected conditions"},
    {label: "RAG Δ success", value: signedPctPoints(ragDeltaSuccess), note: effectDeltaNote(ragPairs, "success_effect")},
    {label: "RAG Δ first-try", value: signedPctPoints(ragDeltaFirstTry), note: effectDeltaNote(ragPairs, "first_try_effect")},
    {label: "RAG Δ coverage", value: signedPctPoints(ragDeltaCoverage), note: `${ragPairs.length} paired comparisons`},
    {label: "RAG Δ seeding", value: signedFmt(ragDeltaSeeding), note: `${ragPairs.length} paired comparisons`},
    {label: "Mode Δ seeding", value: signedFmt(modeDeltaSeeding), note: `${modePairs.length} paired comparisons`}
  ];
  els.metrics.innerHTML = metrics.map(metric => `
    <div class="metric">
      <span class="label">${escapeHtml(metric.label)}</span>
      <span class="value">${escapeHtml(metric.value)}</span>
      <span class="note">${escapeHtml(metric.note)}</span>
    </div>
  `).join("");
}

function renderRagPairs() {
  const pairs = filteredRagPairs();
  els.ragCount.textContent = `${pairs.length} pairs`;
  els.ragPairs.innerHTML = pairs.length ? pairs.map(pair => pairHtml({
    title: `${pair.dataset} - ${pair.provider_model}`,
    subtitle: pair.query_mode,
    deltas: [
      successEffectBadge(pair.success_effect),
      firstTryEffectBadge(pair.first_try_effect),
      deltaBadge("features", pair.delta_features, 1),
      deltaBadge("coverage", pair.delta_observed_feature_coverage || 0, 2),
      deltaBadge("seeding", pair.delta_seeding, 1)
    ],
    leftLabel: "RAG off",
    rightLabel: "RAG on",
    left: pair.rag_off,
    right: pair.rag_on
  })).join("") : emptyHtml();
}

function renderModePairs() {
  const pairs = filteredModePairs();
  els.modeCount.textContent = `${pairs.length} pairs`;
  els.modePairs.innerHTML = pairs.length ? pairs.map(pair => pairHtml({
    title: `${pair.dataset} - ${pair.provider_model}`,
    subtitle: pair.rag_enabled ? "RAG on" : "RAG off",
    deltas: [
      deltaBadge("features", pair.delta_features, 1),
      deltaBadge("coverage", pair.delta_observed_feature_coverage || 0, 2),
      deltaBadge("seeding", pair.delta_seeding, 1)
    ],
    leftLabel: "Explorative",
    rightLabel: "Feature aware",
    left: pair.explorative,
    right: pair.feature_aware
  })).join("") : emptyHtml();
}

function pairHtml(config) {
  return `
    <article class="pair">
      <div class="pair-header">
        <div class="pair-title">
          <strong>${escapeHtml(config.title)}</strong>
          <span>${escapeHtml(config.subtitle)}</span>
        </div>
        <div class="delta">${config.deltas.join("")}</div>
      </div>
      <div class="pair-body">
        ${resultHtml(config.leftLabel, config.left)}
        ${resultHtml(config.rightLabel, config.right)}
      </div>
    </article>
  `;
}

function resultHtml(label, record) {
  return `
    <div class="result">
      <div class="result-head">
        <strong>${escapeHtml(label)}</strong>
        <span class="delta">
          ${boolBadge(record.succeeded, "success")}
          ${strategyBadge(record)}
        </span>
      </div>
      ${thumbHtml(record)}
      <div class="score-row">
        ${scoreHtml("Features", `${fmt(record.features_recognized, 0)} / ${fmt(record.observed_feature_max, 0)}`)}
        ${scoreHtml("Coverage", pct(record.observed_feature_coverage))}
        ${scoreHtml("Seeding", fmt(record.seeding_score, 1))}
        ${scoreHtml("Attempts", fmt(record.attempts, 0))}
      </div>
      <p class="notes">${escapeHtml(firstUsefulNote(record))}</p>
      <div class="run-actions">
        <button class="run-link" type="button" data-action="detail" data-record="${escapeAttr(record.id)}">Details</button>
        ${editorButtonHtml(record)}
      </div>
    </div>
  `;
}

function editorButtonHtml(record, label = "Edit") {
  if (!state.editorEnabled) return "";
  return `<button class="edit-link" type="button" data-action="edit" data-record="${escapeAttr(record.id)}">${escapeHtml(label)}</button>`;
}

function thumbHtml(record) {
  if (!record.has_image || !record.image_url) {
    return `<div class="thumb"><span class="missing">No viewport image</span></div>`;
  }
  return `<button class="thumb" type="button" data-record="${escapeAttr(record.id)}"><img src="${escapeAttr(record.image_url)}" alt="${escapeAttr(record.dataset)} ${escapeAttr(record.provider_model)} ${escapeAttr(record.query_mode)}"></button>`;
}

function scoreHtml(label, value) {
  return `<div class="score"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function firstUsefulNote(record) {
  return record.feature_notes || record.seeding_notes || record.target_feature || "";
}

function renderMatrix() {
  const filtered = data.primary_records.filter(record => recordPasses(record));
  const byDataset = groupBy(filtered, record => record.dataset);
  els.matrixCount.textContent = `${filtered.length} conditions`;
  els.matrix.innerHTML = Object.keys(byDataset).sort().map(dataset => `
    <section class="dataset-band">
      <h3>${escapeHtml(dataset)}</h3>
      <div class="matrix-grid">
        ${byDataset[dataset].map(record => miniHtml(record)).join("")}
      </div>
    </section>
  `).join("") || emptyHtml();
}

function miniHtml(record) {
  return `
    <article class="mini">
      ${thumbHtml(record)}
      <div class="mini-meta">
        <strong>${escapeHtml(record.provider_model)}</strong>
        <span>${escapeHtml(record.query_mode)} - ${escapeHtml(record.rag_strategy_label || (record.rag_enabled ? "RAG on" : "No RAG"))}</span>
        <span>F ${fmt(record.features_recognized, 0)} | S ${fmt(record.seeding_score, 1)}</span>
      </div>
    </article>
  `;
}

function renderRunsTable() {
  const filtered = data.records.filter(record => recordPasses(record));
  els.runsCount.textContent = `${filtered.length} runs`;
  const header = ["Dataset", "Model", "Mode", "RAG category", "Success", "Features", "Seeding", "Attempts", "Image", "Actions"];
  const rows = filtered.map(record => `
    <tr>
      <td>${escapeHtml(record.dataset)}</td>
      <td>${escapeHtml(record.provider_model)}</td>
      <td>${escapeHtml(record.query_mode)}</td>
      <td>${escapeHtml(record.rag_strategy_label || (record.rag_enabled ? "RAG on" : "No RAG"))}</td>
      <td>${record.succeeded ? "yes" : "no"}</td>
      <td>${fmt(record.features_recognized, 0)}</td>
      <td>${fmt(record.seeding_score, 1)}</td>
      <td>${fmt(record.attempts, 0)}</td>
      <td>${record.has_image ? "yes" : "no"}</td>
      <td>
        <div class="run-actions">
          <button class="run-link" type="button" data-action="detail" data-record="${escapeAttr(record.id)}">Details</button>
          ${editorButtonHtml(record)}
        </div>
      </td>
    </tr>
  `).join("");
  els.runsTable.innerHTML = `<thead><tr>${header.map(value => `<th>${value}</th>`).join("")}</tr></thead><tbody>${rows}</tbody>`;
}

function renderSummary() {
  const scopes = new Set(["primary_by_rag_strategy", "primary_by_mode", "primary_by_provider", "primary_by_dataset"]);
  const rows = data.group_summaries.filter(row => scopes.has(row.scope));
  els.summaryBars.innerHTML = rows.map(row => `
    <div class="bar-row">
      <div class="bar-head">
        <strong>${escapeHtml(cleanScope(row.scope))}: ${escapeHtml(row.group)}</strong>
        <span>${pct(row.success_rate)} success - ${fmt(row.avg_seeding_all, 1)} seeding</span>
      </div>
      <div class="bar-track"><div class="bar-fill" style="width: ${Math.max(0, Math.min(100, row.avg_seeding_all * 10))}%"></div></div>
    </div>
  `).join("");

  els.qualityPanel.innerHTML = [
    qualityItem("Missing conditions", data.totals.missing_conditions),
    qualityItem("Duplicate conditions", data.totals.duplicate_conditions),
    qualityItem("Runs without images", data.totals.runs_without_images),
    qualityItem("RAG pairs", data.totals.rag_pairs),
    qualityItem("Mode pairs", data.totals.mode_pairs)
  ].join("");
}

function cleanScope(scope) {
  return scope.replace("primary_by_", "").replaceAll("_", " ");
}

function qualityItem(label, value) {
  const cls = value === 0 ? "good" : "warn";
  return `<div class="quality-item"><span class="badge ${cls}">${escapeHtml(label)}: ${escapeHtml(value)}</span></div>`;
}

function openDetail(id) {
  const record = data.records.find(item => item.id === id) || data.primary_records.find(item => item.id === id);
  if (!record) return;
  els.detailContent.innerHTML = `
    <h2>${escapeHtml(record.dataset)} - ${escapeHtml(record.provider_model)}</h2>
    <p class="report-meta">${escapeHtml(record.query_mode)} - ${escapeHtml(record.rag_strategy_label || (record.rag_enabled ? "RAG on" : "No RAG"))} - ${escapeHtml(record.saved_at || "")}</p>
    ${record.has_image ? `<img class="detail-image" src="${escapeAttr(record.image_url)}" alt="">` : `<div class="thumb"><span class="missing">No viewport image</span></div>`}
    <div class="score-row">
      ${scoreHtml("Success", record.succeeded ? "yes" : "no")}
      ${scoreHtml("Features", `${fmt(record.features_recognized, 0)} / ${fmt(record.observed_feature_max, 0)}`)}
      ${scoreHtml("Coverage", pct(record.observed_feature_coverage))}
      ${scoreHtml("Seeding", fmt(record.seeding_score, 1))}
    </div>
    <h3>Feature Notes</h3>
    <p class="notes">${escapeHtml(record.feature_notes || "-")}</p>
    <h3>Seeding Notes</h3>
    <p class="notes">${escapeHtml(record.seeding_notes || "-")}</p>
    <h3>Target Feature</h3>
    <p class="notes">${escapeHtml(record.target_feature || "-")}</p>
    <p>${record.session_url ? `<a href="${escapeAttr(record.session_url)}">Session JSON</a>` : ""} ${record.code_url ? `<a href="${escapeAttr(record.code_url)}">Generated code</a>` : ""}</p>
    <div class="detail-actions">${editorButtonHtml(record, "Edit experiment")}</div>
  `;
  const editButton = els.detailContent.querySelector("[data-action='edit']");
  if (editButton) {
    editButton.addEventListener("click", () => openEditor(record.id));
  }
  showDialog();
}

function openEditor(id) {
  const record = data.records.find(item => item.id === id) || data.primary_records.find(item => item.id === id);
  if (!record || !state.editorEnabled) return;
  els.detailContent.innerHTML = editFormHtml(record);
  const form = els.detailContent.querySelector("#editForm");
  const cancel = els.detailContent.querySelector("#cancelEdit");
  form.addEventListener("submit", event => {
    event.preventDefault();
    saveExperimentEdits(record, form);
  });
  cancel.addEventListener("click", () => openDetail(record.id));
  showDialog();
}

function editFormHtml(record) {
  return `
    <form class="edit-form" id="editForm">
      <div>
        <h2>${escapeHtml(record.dataset)} - ${escapeHtml(record.provider_model)}</h2>
        <p class="report-meta">${escapeHtml(record.query_mode)} - ${escapeHtml(record.rag_strategy_label || (record.rag_enabled ? "RAG on" : "No RAG"))} - ${escapeHtml(record.session_file || record.session_rel_path)}</p>
      </div>
      <div class="form-grid">
        <label>
          <span>Result</span>
          <select name="succeeded">
            <option value="true" ${record.succeeded ? "selected" : ""}>Succeeded</option>
            <option value="false" ${!record.succeeded ? "selected" : ""}>Failed</option>
          </select>
        </label>
        <label>
          <span>Attempts</span>
          <input name="attempts" type="number" min="0" max="100000" step="1" value="${escapeAttr(fmt(record.attempts, 0))}">
        </label>
        <label>
          <span>Features recognized</span>
          <input name="features_recognized" type="number" min="0" max="100000" step="1" value="${escapeAttr(fmt(record.features_recognized, 0))}">
        </label>
        <label>
          <span>Seeding score</span>
          <input name="seeding_score" type="number" min="0" max="100" step="0.1" value="${escapeAttr(fmt(record.seeding_score, 1))}">
        </label>
        <label>
          <span>Colormap used</span>
          <select name="colormap_used">
            <option value="true" ${record.colormap_used ? "selected" : ""}>Yes</option>
            <option value="false" ${!record.colormap_used ? "selected" : ""}>No</option>
          </select>
        </label>
        <label>
          <span>Suggested seeding used</span>
          <select name="suggested_seeding_used">
            <option value="" ${record.suggested_seeding_used === null || record.suggested_seeding_used === undefined ? "selected" : ""}>Unclear</option>
            <option value="true" ${record.suggested_seeding_used === true ? "selected" : ""}>Yes</option>
            <option value="false" ${record.suggested_seeding_used === false ? "selected" : ""}>No</option>
          </select>
        </label>
        <label class="wide">
          <span>Feature notes</span>
          <textarea name="feature_notes">${escapeHtml(record.feature_notes || "")}</textarea>
        </label>
        <label class="wide">
          <span>Seeding notes</span>
          <textarea name="seeding_notes">${escapeHtml(record.seeding_notes || "")}</textarea>
        </label>
      </div>
      <div class="detail-actions">
        <button class="primary-button" type="submit">Save JSON</button>
        <button class="secondary-button" type="button" id="cancelEdit">Cancel</button>
      </div>
      <p class="edit-status" id="editStatus"></p>
    </form>
  `;
}

async function saveExperimentEdits(record, form) {
  const button = form.querySelector("button[type='submit']");
  const status = form.querySelector("#editStatus");
  button.disabled = true;
  status.classList.remove("error");
  status.textContent = "Saving...";
  const formData = new FormData(form);
  const suggested = formData.get("suggested_seeding_used");
  const experiment = {
    succeeded: formData.get("succeeded") === "true",
    attempts: formData.get("attempts"),
    features_recognized: formData.get("features_recognized"),
    seeding_score: formData.get("seeding_score"),
    colormap_used: formData.get("colormap_used") === "true",
    suggested_seeding_used: suggested === "" ? null : suggested === "true",
    feature_notes: formData.get("feature_notes"),
    seeding_notes: formData.get("seeding_notes")
  };

  try {
    const response = await fetch("/api/experiment", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        session_rel_path: record.session_rel_path,
        experiment
      })
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || "The JSON could not be saved.");
    }
    status.textContent = "Saved. Refreshing...";
    window.setTimeout(() => window.location.reload(), 350);
  } catch (error) {
    status.textContent = error.message || String(error);
    status.classList.add("error");
    button.disabled = false;
  }
}

function showDialog() {
  if (!els.dialog.open) {
    els.dialog.showModal();
  }
}

async function detectEditor() {
  if (!["http:", "https:"].includes(window.location.protocol)) return;
  try {
    const response = await fetch("/api/editor/status", {cache: "no-store"});
    if (!response.ok) return;
    const payload = await response.json();
    state.editorEnabled = Boolean(payload.editor_enabled);
    renderMeta();
    renderActive();
  } catch (error) {
    state.editorEnabled = false;
  }
}

function groupBy(items, fn) {
  return items.reduce((acc, item) => {
    const key = fn(item);
    (acc[key] ||= []).push(item);
    return acc;
  }, {});
}

function emptyHtml() {
  return `<div class="empty">No matching records.</div>`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, char => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  })[char]);
}

function escapeAttr(value) {
  return escapeHtml(value);
}

function renderActive() {
  renderMetrics();
  renderRagPairs();
  renderModePairs();
  renderMatrix();
  renderRunsTable();
  renderSummary();
  document.querySelectorAll("[data-record]").forEach(button => {
    button.addEventListener("click", event => {
      const id = event.currentTarget.getAttribute("data-record");
      const action = event.currentTarget.getAttribute("data-action") || "detail";
      if (!id) return;
      if (action === "edit") {
        openEditor(id);
      } else {
        openDetail(id);
      }
    });
  });
}

function wireEvents() {
  els.modeFilter.addEventListener("change", event => {
    state.mode = event.target.value;
    renderActive();
  });
  els.ragFilter.addEventListener("change", event => {
    state.rag = event.target.value;
    renderActive();
  });
  els.searchFilter.addEventListener("input", event => {
    state.search = event.target.value;
    renderActive();
  });
  document.querySelectorAll(".tabs button").forEach(button => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tabs button").forEach(item => item.classList.toggle("active", item === button));
      document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.remove("active"));
      document.getElementById(`tab-${button.dataset.tab}`).classList.add("active");
    });
  });
  els.closeDialog.addEventListener("click", () => els.dialog.close());
  document.addEventListener("click", () => closeMultiSelects());
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") closeMultiSelects();
  });
}

populateFilters();
renderMeta();
renderActive();
wireEvents();
detectEditor();
