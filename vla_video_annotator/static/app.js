const views = ["head", "left", "right"];
const palette = ["#ff4a58", "#32b870", "#2474e8", "#d97706", "#7c3aed", "#0891b2"];
const colors = { manip_obj: "#ff4a58", target: "#32b870" };
const LANG_KEY = "vla-video-annotator:language";
const i18n = {
  zh: {
    language: "界面语言",
    datasetLocation: "数据集位置",
    outputDir: "Result 输出目录",
    modelBackend: "模型后端",
    refreshDataset: "刷新数据集",
    taskQueue: "后台任务列表",
    queueGpuPlaceholder: "GPU 编号，如 0,1,2；留空使用上方 GPU",
    clearFinished: "清理完成项",
    headEmpty: "head 空",
    leftEmpty: "left 空",
    rightEmpty: "right 空",
    overwriteExisting: "覆盖已有结果",
    labels: "标注类别",
    promptMode: "自动标注方式",
    brushSize: "画笔大小",
    previewAll: "预览起始帧",
    runAll: "提交三视角任务",
    saveManual: "保存手动调整",
    releaseModel: "释放模型显存",
    queueStatus: "队列状态",
    prevEpisode: "上一集",
    nextEpisode: "下一集",
    enqueueEpisode: "加入任务列表",
    activeLabel: "当前点选类别",
    tools: "绘制工具",
    point: "点",
    box: "框",
    paint: "画笔",
    erase: "橡皮",
    clearCurrent: "清除当前点",
    prevPrompt: "上一个点",
    nextPrompt: "下一个点",
    preview: "预览",
    predict: "推理",
    result: "结果",
    noTimeline: "暂无 server 变更记录",
    textPromptPlaceholder: "在这里输入 Text prompt",
    imageLoadFailed: "图像加载失败，已重试 {retries} 次\n{url}",
    promptJumped: "{label}: 已跳到{direction}标注帧",
    noPromptFrame: "{label}: 没有{direction}标注帧",
    nextSegment: "下一段",
    prevSegment: "上一段",
    lastEpisode: "已经是最后一集",
    firstEpisode: "已经是第一集",
    selectEpisodeFirst: "请先选择 episode",
    selectAtLeastOneView: "至少选择一个视角",
    enqueueSubmitted: "{episode} {views} 已提交后台任务: {queueId}",
    episodeQueued: "{episode} 已加入后台任务: {queueId}",
    viewSubmitted: "{view}: 从 Raw {frame}f 提交传播任务",
    previewStart: "{view}: 预览起始帧 {frame}f",
    modelReleased: "模型已释放: {status}",
    manualSaved: "手动调整已保存到 mask_int.npy / boxes_xywh.npy",
    modelStatus: "模型状态: {status} | 后端: {backend}\nPoint/Box 建议用 SAM2.1；Text 可选 SAM3.1；队列 GPU 可填 0,1,2 并行三视角",
    noEpisodes: "没有找到 episode_*.mp4",
    refreshed: "已刷新: {videoRoot}\nResult 输出目录: {outputRoot}\nText 自动推理输入位置：左侧“标注类别”每行的文本框",
    annotated: "annotated",
    missing: "missing",
    noVideo: "no video",
    annotation: "annotation",
    defaultGpu: "默认",
  },
  en: {
    language: "Language",
    datasetLocation: "Dataset location",
    outputDir: "Result output directory",
    modelBackend: "Model backend",
    refreshDataset: "Refresh dataset",
    taskQueue: "Background tasks",
    queueGpuPlaceholder: "GPU IDs, e.g. 0,1,2; empty uses the GPU above",
    clearFinished: "Clear finished",
    headEmpty: "head empty",
    leftEmpty: "left empty",
    rightEmpty: "right empty",
    overwriteExisting: "Overwrite existing results",
    labels: "Labels",
    promptMode: "Auto annotation mode",
    brushSize: "Brush size",
    previewAll: "Preview start frame",
    runAll: "Submit all views",
    saveManual: "Save manual edits",
    releaseModel: "Release model memory",
    queueStatus: "Queue status",
    prevEpisode: "Previous",
    nextEpisode: "Next",
    enqueueEpisode: "Add to queue",
    activeLabel: "Active label",
    tools: "Drawing tool",
    point: "Point",
    box: "Box",
    paint: "Brush",
    erase: "Eraser",
    clearCurrent: "Clear current prompt",
    prevPrompt: "Previous prompt",
    nextPrompt: "Next prompt",
    preview: "Preview",
    predict: "Infer",
    result: "Result",
    noTimeline: "No server timeline yet",
    textPromptPlaceholder: "Enter text prompt",
    imageLoadFailed: "Image load failed after {retries} retries\n{url}",
    promptJumped: "{label}: jumped to the {direction} prompt frame",
    noPromptFrame: "{label}: no {direction} prompt frame",
    nextSegment: "next",
    prevSegment: "previous",
    lastEpisode: "Already at the last episode",
    firstEpisode: "Already at the first episode",
    selectEpisodeFirst: "Select an episode first",
    selectAtLeastOneView: "Select at least one view",
    enqueueSubmitted: "{episode} {views} submitted to queue: {queueId}",
    episodeQueued: "{episode} added to queue: {queueId}",
    viewSubmitted: "{view}: submitted propagation from Raw {frame}f",
    previewStart: "{view}: preview start frame {frame}f",
    modelReleased: "Model released: {status}",
    manualSaved: "Manual edits saved to mask_int.npy / boxes_xywh.npy",
    modelStatus: "Model status: {status} | backend: {backend}\nPoint/Box: SAM2.1 recommended; Text: optional SAM3.1; queue GPUs can be 0,1,2 for parallel views",
    noEpisodes: "No episode_*.mp4 files found",
    refreshed: "Refreshed: {videoRoot}\nResult output directory: {outputRoot}\nText prompts are entered in the label rows on the left",
    annotated: "annotated",
    missing: "missing",
    noVideo: "no video",
    annotation: "annotation",
    defaultGpu: "Default",
  },
};

const state = {
  config: null,
  episodes: [],
  episode: null,
  activeLabel: "manip_obj",
  promptMode: "point",
  tool: "box",
  rawFrames: { head: 0, left: 0, right: 0 },
  resultFrames: { head: 0, left: 0, right: 0 },
  frameCounts: { head: 0, left: 0, right: 0 },
  sizes: {
    head: { width: 640, height: 480 },
    left: { width: 640, height: 480 },
    right: { width: 640, height: 480 },
  },
  points: {},
  boxes: {},
  serverTimeline: [],
  frameLoadTimers: {},
  language: localStorage.getItem(LANG_KEY) || "zh",
};

const $ = (id) => document.getElementById(id);

function t(key, vars = {}) {
  const dict = i18n[state.language] || i18n.zh;
  return (dict[key] || i18n.zh[key] || key).replace(/\{(\w+)\}/g, (_, name) => vars[name] ?? "");
}

function applyI18n() {
  document.documentElement.lang = state.language === "en" ? "en" : "zh-CN";
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.placeholder = t(node.dataset.i18nPlaceholder);
  });
  document.querySelectorAll(".label-text").forEach((input) => {
    input.placeholder = t("textPromptPlaceholder");
  });
  const languageSelect = $("languageSelect");
  if (languageSelect) languageSelect.value = state.language;
}

function setLanguage(language) {
  state.language = language === "en" ? "en" : "zh";
  localStorage.setItem(LANG_KEY, state.language);
  applyI18n();
  renderTimeline();
  if (state.episode) renderEpisodeAnnotationStatus();
}

function setStatus(text) {
  $("status").textContent = text;
}

async function api(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function activeColor(label = state.activeLabel) {
  if (!colors[label]) colors[label] = palette[Object.keys(colors).length % palette.length];
  return colors[label];
}

function key(view, label = state.activeLabel) {
  return `${view}:${label}`;
}

function frameKey(view, label = state.activeLabel, frame = state.resultFrames[view]) {
  return `${view}:${label}:${frame}`;
}

function encodePath(path) {
  return encodeURIComponent(path);
}

function currentViewPath(view) {
  return state.episode?.views?.[view]?.path || "";
}

function draftDatasetKey() {
  return state.config?.video_root || $("videoRoot")?.value?.trim() || "";
}

function draftKey(episode = state.episode?.episode, chunk = state.episode?.chunk) {
  if (!episode) return "";
  return `vla-video-annotator:draft:${draftDatasetKey()}:${chunk || "chunk-000"}:${episode}`;
}

function saveDraft() {
  const key = draftKey();
  if (!key) return;
  localStorage.setItem(key, JSON.stringify({
    points: state.points,
    boxes: state.boxes,
    rawFrames: state.rawFrames,
    resultFrames: state.resultFrames,
  }));
}

function loadDraft() {
  const key = draftKey();
  if (!key) return;
  try {
    const draft = JSON.parse(localStorage.getItem(key) || "{}");
    state.points = draft.points || {};
    state.boxes = draft.boxes || {};
    state.rawFrames = { ...state.rawFrames, ...(draft.rawFrames || {}) };
    state.resultFrames = { ...state.resultFrames, ...(draft.resultFrames || {}) };
  } catch {
    state.points = {};
    state.boxes = {};
  }
  saveDraft();
}

function addTimeline(action, view, label, frame, detail = "") {
  saveDraft();
}

function renderTimeline() {
  const root = $("promptTimeline");
  if (!root) return;
  root.innerHTML = "";
  if (!state.serverTimeline.length) {
    root.textContent = t("noTimeline");
    return;
  }
  for (const item of state.serverTimeline) {
    const row = document.createElement("div");
    row.className = "timeline-item static";
    row.textContent = `${item.time || ""} ${item.title || ""} ${item.detail || ""}`;
    root.appendChild(row);
  }
}

async function loadServerTimeline() {
  try {
    const data = await api("/api/server_timeline");
    state.serverTimeline = data.timeline || [];
  } catch {
    state.serverTimeline = [];
  }
  renderTimeline();
}

function labelRows() {
  return [...$("labelList").querySelectorAll(".label-item")]
    .map((row, idx) => ({
      enabled: row.querySelector(".label-enabled").checked,
      name: row.querySelector(".label-name").value.trim(),
      text: row.querySelector(".label-text").value.trim(),
      obj_id: idx + 1,
    }))
    .filter((x) => x.name);
}

function refreshActiveLabelSelect() {
  const labels = labelRows();
  const box = $("activeLabelButtons");
  const previous = state.activeLabel;
  state.activeLabel = labels.some((x) => x.name === previous) ? previous : labels[0]?.name || "";
  box.innerHTML = "";
  for (const label of labels) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `label-choice${label.name === state.activeLabel ? " active" : ""}`;
    btn.textContent = label.name;
    btn.style.setProperty("--label-color", activeColor(label.name));
    btn.addEventListener("click", () => {
      state.activeLabel = label.name;
      refreshActiveLabelSelect();
    });
    box.appendChild(btn);
  }
  loadMasksForAll();
  drawAll();
}

function initLabels(labels) {
  const box = $("labelList");
  box.innerHTML = "";
  labels.forEach((name) => {
    const row = document.createElement("div");
    row.className = "label-item";
    row.innerHTML = `
      <input class="label-enabled" type="checkbox" checked>
      <span class="swatch"></span>
      <span class="label-title">${name}</span>
      <input class="label-name" type="hidden" value="${name}">
      <input class="label-text" type="text" value="${name}" aria-label="${name} text prompt" placeholder="${t("textPromptPlaceholder")}">
    `;
    const swatch = row.querySelector(".swatch");
    const nameInput = row.querySelector(".label-name");
    swatch.style.backgroundColor = activeColor(name);
    row.querySelectorAll("input").forEach((input) => {
      input.addEventListener("input", () => {
        swatch.style.backgroundColor = activeColor(nameInput.value.trim());
        refreshActiveLabelSelect();
      });
      input.addEventListener("change", refreshActiveLabelSelect);
    });
    box.appendChild(row);
  });
  refreshActiveLabelSelect();
}

function buildViews() {
  buildViewGrid("viewGrid", "raw");
  buildViewGrid("resultGrid", "result");
}

function buildViewGrid(gridId, kind) {
  const grid = $(gridId);
  const tpl = $("viewTemplate");
  grid.innerHTML = "";
  views.forEach((view) => {
    const node = tpl.content.firstElementChild.cloneNode(true);
    node.dataset.view = view;
    node.dataset.kind = kind;
    node.querySelector("h2").textContent = `${view}`;
    node.querySelector(".predict-view").addEventListener("click", () => enqueueViews([view]));
    node.querySelector(".preview-view").addEventListener("click", () => runPreview([view]));
    node.querySelector(".overlay-view").addEventListener("click", () => showOverlay(view));
    const slider = node.querySelector(".frame-slider");
    const number = node.querySelector(".frame-number");
    slider.addEventListener("input", (event) => setViewFrame(kind, view, event.target.value, { defer: true }));
    slider.addEventListener("change", (event) => setViewFrame(kind, view, event.target.value));
    slider.addEventListener("wheel", (event) => wheelFrame(event, kind, view), { passive: false });
    number.addEventListener("change", (event) => setViewFrame(kind, view, event.target.value));
    number.addEventListener("wheel", (event) => wheelFrame(event, kind, view), { passive: false });
    const stage = node.querySelector(".stage");
    stage.addEventListener("pointerdown", (event) => pointerDown(event, kind, view));
    stage.addEventListener("pointermove", (event) => pointerMove(event, kind, view));
    stage.addEventListener("pointerup", (event) => pointerUp(event, kind, view));
    stage.addEventListener("pointerleave", (event) => pointerUp(event, kind, view));
    if (kind === "raw") node.querySelector(".overlay-view").style.display = "none";
    grid.appendChild(node);
  });
}

function wheelFrame(event, kind, view) {
  event.preventDefault();
  const current = kind === "raw" ? state.rawFrames[view] : state.resultFrames[view];
  const direction = event.deltaY < 0 ? 1 : -1;
  setViewFrame(kind, view, current + direction);
}

function panel(view, kind = "raw") {
  return document.querySelector(`.view-panel[data-kind="${kind}"][data-view="${view}"]`);
}

function panels(view) {
  return [panel(view, "raw"), panel(view, "result")].filter(Boolean);
}

function frameOf(kind, view) {
  return kind === "result" ? state.resultFrames[view] : state.rawFrames[view];
}

function setFrameOf(kind, view, frame) {
  if (kind === "result") state.resultFrames[view] = frame;
  else state.rawFrames[view] = frame;
}

function stageRect(kind, view) {
  return panel(view, kind).querySelector(".stage").getBoundingClientRect();
}

function eventPoint(event, kind, view) {
  const rect = stageRect(kind, view);
  const size = state.sizes[view];
  const scale = Math.min(rect.width / size.width, rect.height / size.height);
  const drawW = size.width * scale;
  const drawH = size.height * scale;
  const ox = (rect.width - drawW) / 2;
  const oy = (rect.height - drawH) / 2;
  const x = (event.clientX - rect.left - ox) / scale;
  const y = (event.clientY - rect.top - oy) / scale;
  return {
    x: Math.max(0, Math.min(size.width - 1, x)),
    y: Math.max(0, Math.min(size.height - 1, y)),
  };
}

function resizeCanvases(view) {
  const size = state.sizes[view];
  panels(view).forEach((p) => {
    p.querySelectorAll("canvas").forEach((canvas) => {
      if (canvas.width !== size.width || canvas.height !== size.height) {
        canvas.width = size.width;
        canvas.height = size.height;
      }
    });
  });
}

function setImageWithRetry(img, url, retries = 4) {
  const token = String(Date.now()) + Math.random().toString(16).slice(2);
  img.dataset.loadToken = token;
  const load = (attempt) => {
    if (img.dataset.loadToken !== token) return;
    img.onerror = () => {
      if (img.dataset.loadToken !== token) return;
      if (attempt >= retries) {
        setStatus(t("imageLoadFailed", { retries, url }));
        return;
      }
      setTimeout(() => {
        const sep = url.includes("?") ? "&" : "?";
        load(attempt + 1);
        img.src = `${url}${sep}retry=${attempt + 1}&t=${Date.now()}`;
      }, 250 + attempt * 250);
    };
    img.onload = () => {
      if (img.dataset.loadToken === token) img.onerror = null;
    };
  };
  load(0);
  img.src = url;
}

async function setViewInfo(view) {
  const info = state.episode.views[view];
  if (!info) {
    panels(view).forEach((p) => { p.querySelector(".meta").textContent = "missing"; });
    return;
  }
  const meta = await api(`/api/video_info?path=${encodePath(info.path)}`);
  state.sizes[view] = { width: meta.width, height: meta.height };
  state.frameCounts[view] = Math.max(1, meta.frame_count || Math.round(meta.duration * meta.fps) || 1);
  panels(view).forEach((p) => {
    const slider = p.querySelector(".frame-slider");
    const input = p.querySelector(".frame-number");
    slider.max = state.frameCounts[view] - 1;
    input.max = state.frameCounts[view] - 1;
    p.querySelector(".meta").textContent = `${meta.width}x${meta.height} ${Math.round(meta.fps || 0)}fps`;
  });
  resizeCanvases(view);
  setViewFrame("raw", view, 0);
  setViewFrame("result", view, 0);
  api("/api/prepare_frames", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path: info.path }),
  }).catch(() => {});
}

function scheduleFrameLoad(kind, view, frame) {
  const timerKey = `${kind}:${view}`;
  clearTimeout(state.frameLoadTimers[timerKey]);
  state.frameLoadTimers[timerKey] = setTimeout(() => {
    setViewFrame(kind, view, frame);
  }, 140);
}

function setViewFrame(kind, view, rawFrame, options = {}) {
  const maxFrame = Math.max(0, state.frameCounts[view] - 1);
  const frame = Math.max(0, Math.min(maxFrame, Number(rawFrame) || 0));
  setFrameOf(kind, view, frame);
  const p = panel(view, kind);
  p.querySelector(".frame-slider").value = frame;
  p.querySelector(".frame-number").value = frame;
  p.querySelector(".points").textContent = `${frame}f`;
  if (options.defer) {
    if (kind === "raw") drawView(view);
    scheduleFrameLoad(kind, view, frame);
    return;
  }
  const path = currentViewPath(view);
  if (path) setImageWithRetry(p.querySelector(".frame"), `/api/frame?path=${encodePath(path)}&frame=${frame}&t=${Date.now()}`);
  if (kind === "result") {
    showOverlay(view).catch(() => {
      p.querySelector(".overlay").classList.remove("visible");
      clearMask(view);
    });
  } else {
    drawView(view);
  }
}

function drawAll() {
  views.forEach(drawView);
}

function drawView(view) {
  resizeCanvases(view);
  drawRaw(view);
  drawResult(view);
}

function drawRaw(view) {
  const p = panel(view, "raw");
  const draw = p.querySelector(".draw");
  const mask = p.querySelector(".mask");
  draw.classList.toggle("editing", (state.tool === "point" || state.tool === "box") && state.promptMode === "point");
  mask.classList.remove("editing");
  const ctx = draw.getContext("2d");
  ctx.clearRect(0, 0, draw.width, draw.height);
  const frame = state.rawFrames[view];
  for (const row of labelRows()) {
    ctx.fillStyle = activeColor(row.name);
    ctx.lineWidth = 3;
    for (const pt of state.points[key(view, row.name)] || []) {
      if (pt.frame !== frame) continue;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = pt.label === 0 ? "#fff" : activeColor(row.name);
      ctx.stroke();
    }
    const b = state.boxes[frameKey(view, row.name, frame)];
    if (b) {
      const active = row.name === state.activeLabel;
      ctx.strokeStyle = activeColor(row.name);
      ctx.lineWidth = active ? 4 : 2;
      ctx.strokeRect(b.x, b.y, b.w, b.h);
      ctx.font = "14px sans-serif";
      ctx.fillStyle = activeColor(row.name);
      ctx.fillText(row.name, b.x + 4, Math.max(14, b.y - 6));
    }
  }
  p.querySelector(".points").textContent = `${frame}f`;
}

function drawResult(view) {
  const p = panel(view, "result");
  const draw = p.querySelector(".draw");
  const mask = p.querySelector(".mask");
  draw.classList.remove("editing");
  mask.classList.toggle("editing", state.tool === "paint" || state.tool === "erase");
  const ctx = draw.getContext("2d");
  ctx.clearRect(0, 0, draw.width, draw.height);
  p.querySelector(".points").textContent = `${state.resultFrames[view]}f`;
}

let drawing = null;

function pointerDown(event, kind, view) {
  event.preventDefault();
  event.currentTarget.setPointerCapture?.(event.pointerId);
  const pt = eventPoint(event, kind, view);
  if (kind === "raw" && state.tool === "point" && state.promptMode === "point") {
    const k = key(view);
    state.points[k] ||= [];
    const pointLabel = event.shiftKey ? 0 : 1;
    state.points[k].push({ ...pt, frame: state.rawFrames[view], label: pointLabel });
    addTimeline("point", view, state.activeLabel, state.rawFrames[view], pointLabel ? "+" : "-");
    drawRaw(view);
    return;
  }
  if (kind === "raw" && state.tool === "box" && state.promptMode === "point") {
    drawing = { kind, view, label: state.activeLabel, start: pt, type: "box", frame: state.rawFrames[view] };
    state.boxes[frameKey(view, drawing.label, drawing.frame)] = { x: pt.x, y: pt.y, w: 0, h: 0 };
    drawRaw(view);
    return;
  }
  if (kind !== "result") return;
  if (state.tool === "paint" || state.tool === "erase") {
    drawing = { kind, view, label: state.activeLabel, type: state.tool };
    paint(event, view);
  }
}

function pointerMove(event, kind, view) {
  if (drawing) event.preventDefault();
  if (!drawing || drawing.view !== view || drawing.kind !== kind) return;
  if (drawing.type === "box") {
    const pt = eventPoint(event, kind, view);
    const x = Math.min(drawing.start.x, pt.x);
    const y = Math.min(drawing.start.y, pt.y);
    state.boxes[frameKey(view, drawing.label, drawing.frame)] = {
      x,
      y,
      w: Math.abs(pt.x - drawing.start.x),
      h: Math.abs(pt.y - drawing.start.y),
    };
    if (kind === "raw") drawRaw(view);
    else drawResult(view);
    return;
  }
  paint(event, view);
}

function pointerUp(event, kind, view) {
  if (drawing) event.preventDefault();
  event.currentTarget.releasePointerCapture?.(event.pointerId);
  if (drawing && drawing.view === view && drawing.kind === kind && drawing.type !== "box") paint(event, view);
  if (drawing && drawing.view === view && drawing.kind === kind && drawing.type === "box") {
    const b = state.boxes[frameKey(view, drawing.label, drawing.frame)];
    if (b && b.w >= 2 && b.h >= 2) {
      addTimeline("box", view, drawing.label, drawing.frame, `${Math.round(b.w)}x${Math.round(b.h)}`);
    }
  }
  drawing = null;
}

function paint(event, view) {
  const pt = eventPoint(event, "result", view);
  const canvas = panel(view, "result").querySelector(".mask");
  resizeCanvases(view);
  const ctx = canvas.getContext("2d");
  ctx.globalCompositeOperation = state.tool === "erase" ? "destination-out" : "source-over";
  ctx.fillStyle = activeColor(state.activeLabel) + "99";
  ctx.beginPath();
  ctx.arc(pt.x, pt.y, Number($("brushSize").value), 0, Math.PI * 2);
  ctx.fill();
  ctx.globalCompositeOperation = "source-over";
}

function clearCurrent() {
  const frame = state.rawFrames.head || 0;
  views.forEach((view) => {
    const rawFrame = state.rawFrames[view];
    const resultFrame = state.resultFrames[view];
    const k = key(view);
    state.points[k] = (state.points[k] || []).filter((p) => p.frame !== rawFrame);
    delete state.boxes[frameKey(view, state.activeLabel, rawFrame)];
    delete state.boxes[frameKey(view, state.activeLabel, resultFrame)];
    clearMask(view);
    drawView(view);
  });
  addTimeline("clear", "all", state.activeLabel, frame);
  saveDraft();
}

function boxFramesForLabel(view, label) {
  const prefix = `${view}:${label}:`;
  return Object.keys(state.boxes)
    .filter((k) => k.startsWith(prefix))
    .map((k) => Number(k.slice(prefix.length)))
    .filter((frame) => Number.isInteger(frame))
    .sort((a, b) => a - b);
}

function pointFramesForLabel(view, label) {
  return [...new Set((state.points[key(view, label)] || []).map((p) => Number(p.frame)).filter((frame) => Number.isInteger(frame)))]
    .sort((a, b) => a - b);
}

function promptFrameForLabel(view, label) {
  const current = state.rawFrames[view];
  const boxFrames = boxFramesForLabel(view, label);
  const pointFrames = pointFramesForLabel(view, label);
  if (boxFrames.includes(current) || pointFrames.includes(current)) return current;
  if (boxFrames.length) return boxFrames[0];
  if (pointFrames.length) return pointFrames[0];
  return current;
}

function promptSegmentsForLabel(view, label) {
  const frames = [...new Set([...boxFramesForLabel(view, label), ...pointFramesForLabel(view, label)])].sort((a, b) => a - b);
  return frames.map((frame) => {
    const box = state.boxes[frameKey(view, label, frame)];
    return {
      frame_index: frame,
      points: (state.points[key(view, label)] || []).filter((p) => p.frame === frame),
      box_xywh: box ? [box.x, box.y, box.w, box.h] : null,
    };
  });
}

function promptFramesForLabel(view, label) {
  return [...new Set([...boxFramesForLabel(view, label), ...pointFramesForLabel(view, label)])].sort((a, b) => a - b);
}

function jumpPromptSegment(direction) {
  if (!state.activeLabel) return;
  let moved = 0;
  for (const view of views) {
    const frames = promptFramesForLabel(view, state.activeLabel);
    if (!frames.length) continue;
    const current = state.rawFrames[view] || 0;
    const target = direction > 0
      ? frames.find((frame) => frame > current)
      : [...frames].reverse().find((frame) => frame < current);
    if (target == null) continue;
    setViewFrame("raw", view, target);
    moved += 1;
  }
  if (moved) {
    setStatus(t("promptJumped", { label: state.activeLabel, direction: direction > 0 ? t("nextSegment") : t("prevSegment") }));
  } else {
    setStatus(t("noPromptFrame", { label: state.activeLabel, direction: direction > 0 ? t("nextSegment") : t("prevSegment") }));
  }
}

function jumpEpisode(direction) {
  const current = state.episode?.episode;
  if (!current || !state.episodes.length) return;
  const idx = state.episodes.findIndex((ep) => ep.episode === current);
  if (idx < 0) return;
  const nextIdx = Math.max(0, Math.min(state.episodes.length - 1, idx + direction));
  if (nextIdx === idx) {
    setStatus(direction > 0 ? t("lastEpisode") : t("firstEpisode"));
    return;
  }
  selectEpisode(state.episodes[nextIdx].episode).catch((e) => setStatus(String(e)));
}

function payloadFor(view) {
  const frame = state.rawFrames[view];
  return {
    episode: state.episode.episode,
    chunk: state.episode.chunk,
    view,
    prompt_mode: state.promptMode,
    labels: labelRows().map((label) => {
      const labelFrame = state.promptMode === "point" ? promptFrameForLabel(view, label.name) : frame;
      const box = state.boxes[frameKey(view, label.name, labelFrame)];
      const segments = state.promptMode === "point" ? promptSegmentsForLabel(view, label.name) : [];
      return {
        ...label,
        frame_index: labelFrame,
        points: (state.points[key(view, label.name)] || []).filter((p) => p.frame === labelFrame),
        box_xywh: box ? [box.x, box.y, box.w, box.h] : null,
        segments,
      };
    }),
  };
}

async function runPredict(viewList = views) {
  await saveSettings();
  saveDraft();
  for (const view of viewList) {
    setStatus(t("viewSubmitted", { view, frame: state.rawFrames[view] }));
    const { job_id } = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFor(view)),
    });
    await watchJob(job_id, view);
  }
}

async function enqueueViews(viewList = views) {
  await saveSettings();
  saveDraft();
  if (!state.episode) throw new Error(t("selectEpisodeFirst"));
  const body = {
    episode: state.episode.episode,
    chunk: state.episode.chunk,
    prompt_mode: state.promptMode,
    overwrite: $("queueOverwrite").checked,
    gpu_ids: parseQueueGpuIds(),
    views: viewList.map((view) => ({
      ...payloadFor(view),
      empty: false,
    })),
  };
  const res = await api("/api/enqueue_episode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  setStatus(t("enqueueSubmitted", { episode: state.episode.episode, views: viewList.join(","), queueId: res.queue_id }));
  renderQueueStatus(res.queue);
  ensureQueuePolling();
}

async function runPreview(viewList = views) {
  await saveSettings();
  saveDraft();
  for (const view of viewList) {
    setStatus(t("previewStart", { view, frame: state.rawFrames[view] }));
    const { job_id } = await api("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFor(view)),
    });
    await watchJob(job_id, view);
  }
}

async function watchJob(jobId, view) {
  for (;;) {
    const job = await api(`/api/job?id=${jobId}`);
    setStatus(`${view}: ${job.status} ${Math.round((job.progress || 0) * 100)}%\n${job.message || ""}`);
    if (job.status === "done") {
      $("paths").textContent = Object.values(job.metadata.files).join("\n");
      state.resultFrames[view] = state.rawFrames[view];
      syncControls("result", view);
      await showOverlay(view);
      await loadEpisodes(state.episode.episode, { preserveCurrent: true });
      return;
    }
    if (job.status === "error") throw new Error(job.message);
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

function selectedQueueViews() {
  return [...document.querySelectorAll(".queue-view")]
    .filter((input) => input.checked)
    .map((input) => input.value);
}

function emptyQueueViews() {
  return new Set(
    [...document.querySelectorAll(".empty-view")]
      .filter((input) => input.checked)
      .map((input) => input.value)
  );
}

function parseQueueGpuIds() {
  return $("queueGpuIds").value
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean)
    .map((x) => Number(x))
    .filter((x) => Number.isInteger(x));
}

async function enqueueEpisode() {
  await saveSettings();
  saveDraft();
  if (!state.episode) throw new Error(t("selectEpisodeFirst"));
  const selectedViews = selectedQueueViews();
  if (!selectedViews.length) throw new Error(t("selectAtLeastOneView"));
  const emptyViews = emptyQueueViews();
  const body = {
    episode: state.episode.episode,
    chunk: state.episode.chunk,
    prompt_mode: state.promptMode,
    overwrite: $("queueOverwrite").checked,
    gpu_ids: parseQueueGpuIds(),
    views: selectedViews.map((view) => ({
      ...payloadFor(view),
      empty: emptyViews.has(view),
    })),
  };
  const res = await api("/api/enqueue_episode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  setStatus(t("episodeQueued", { episode: state.episode.episode, queueId: res.queue_id }));
  renderQueueStatus(res.queue);
  ensureQueuePolling();
}

function renderQueueStatus(queue) {
  const root = $("queueStatus");
  if (!queue) {
    root.textContent = "";
    updateViewQueueStates(null);
    return;
  }
  const header = `Queue: ${queue.status} | total ${queue.total} | queued ${queue.queued} | dispatched ${queue.dispatched || 0} | running ${queue.running} | done ${queue.done} | error ${queue.errors}`;
  root.innerHTML = `<div>${header}</div>`;
  for (const item of (queue.items || []).slice(0, 120)) {
    const viewsText = Object.entries(item.view_status || {})
      .map(([view, viewStatus]) => {
        const gpu = viewStatus.gpu_id == null ? "" : `@${viewStatus.gpu_id}`;
        return `${view}:${viewStatus.status}${gpu} ${Math.round((viewStatus.progress || 0) * 100)}%`;
      })
      .join(" | ");
    const row = document.createElement("div");
    row.className = "batch-episode";
    row.innerHTML = `<span>${item.episode}</span><span>${item.status}</span><span class="batch-views">${viewsText || item.message || ""}</span>`;
    root.appendChild(row);
  }
  updateViewQueueStates(queue);
}

function updateViewQueueStates(queue) {
  const currentEpisode = state.episode?.episode;
  const latest = {};
  for (const item of queue?.items || []) {
    if (item.episode !== currentEpisode) continue;
    for (const [view, viewStatus] of Object.entries(item.view_status || {})) {
      latest[view] = viewStatus;
    }
  }
  for (const view of views) {
    panels(view).forEach((p) => {
      const node = p.querySelector(".queue-state");
      if (!node) return;
      const st = latest[view];
      node.textContent = st ? `${st.status} ${Math.round((st.progress || 0) * 100)}%` : "";
      node.dataset.status = st?.status || "";
    });
  }
}

let queuePoller = null;

function ensureQueuePolling() {
  if (queuePoller) return;
  queuePoller = setInterval(async () => {
    try {
      const queue = await api("/api/task_queue");
      renderQueueStatus(queue);
      if (queue.status === "idle" && queue.queued === 0 && queue.running === 0) {
        clearInterval(queuePoller);
        queuePoller = null;
        await loadEpisodes(state.episode?.episode, { preserveCurrent: true });
      }
    } catch (e) {
      setStatus(String(e));
    }
  }, 1500);
}

async function clearFinishedQueue() {
  const queue = await api("/api/clear_finished_queue", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  renderQueueStatus(queue);
}

async function releaseModel() {
  const res = await api("/api/release_model", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  setStatus(t("modelReleased", { status: res.model_status }));
}

function syncControls(kind, view) {
  const p = panel(view, kind);
  const frame = frameOf(kind, view);
  p.querySelector(".frame-slider").value = frame;
  p.querySelector(".frame-number").value = frame;
}

async function showOverlay(view) {
  const frame = state.resultFrames[view];
  const p = panel(view, "result");
  const path = currentViewPath(view);
  if (path) setImageWithRetry(p.querySelector(".frame"), `/api/frame?path=${encodePath(path)}&frame=${frame}&t=${Date.now()}`);
  const img = p.querySelector(".overlay");
  setImageWithRetry(img, `/api/overlay?episode=${state.episode.episode}&view=${view}&frame=${frame}&t=${Date.now()}`);
  img.classList.add("visible");
  await loadMask(view);
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  return {
    r: parseInt(clean.slice(0, 2), 16),
    g: parseInt(clean.slice(2, 4), 16),
    b: parseInt(clean.slice(4, 6), 16),
  };
}

function drawRleToCanvas(view, rle, label = state.activeLabel) {
  const canvas = panel(view, "result").querySelector(".mask");
  canvas.width = rle.size[1];
  canvas.height = rle.size[0];
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(canvas.width, canvas.height);
  const rgb = hexToRgb(activeColor(label));
  let value = 0;
  let idx = 0;
  for (const count of rle.counts) {
    for (let i = 0; i < count; i += 1) {
      if (value) {
        const off = idx * 4;
        img.data[off] = rgb.r;
        img.data[off + 1] = rgb.g;
        img.data[off + 2] = rgb.b;
        img.data[off + 3] = 120;
      }
      idx += 1;
    }
    value = 1 - value;
  }
  ctx.putImageData(img, 0, 0);
}

function clearMask(view) {
  const canvas = panel(view, "result").querySelector(".mask");
  canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
}

async function loadMask(view) {
  if (!state.episode || !state.activeLabel) return;
  try {
    const frame = state.resultFrames[view];
    const rle = await api(`/api/mask_rle?episode=${state.episode.episode}&view=${view}&label=${encodeURIComponent(state.activeLabel)}&frame=${frame}`);
    drawRleToCanvas(view, rle);
  } catch {
    clearMask(view);
  }
}

function loadMasksForAll() {
  views.forEach((view) => loadMask(view));
}

function viewMark(ep, view) {
  if (!ep.views?.[view]) return `${view[0]}-`;
  return `${view[0]}${ep.annotations?.[view]?.done ? "+" : "!"}`;
}

function episodeOptionText(ep) {
  const marks = views.map((view) => viewMark(ep, view)).join(" ");
  const done = ep.annotation_summary?.done ?? 0;
  const total = ep.annotation_summary?.total ?? views.length;
  return `${ep.episode} [${marks}] ${done}/${total}`;
}

function renderEpisodeAnnotationStatus() {
  const box = $("episodeAnnotationStatus");
  if (!state.episode) {
    box.textContent = "";
    return;
  }
  box.innerHTML = "";
  for (const view of views) {
    const item = document.createElement("div");
    const status = state.episode.annotations?.[view];
    const hasVideo = Boolean(state.episode.views?.[view]);
    const done = Boolean(status?.done);
    item.className = `episode-status-row ${done ? "done" : "missing"}`;
    const detail = !hasVideo
      ? t("noVideo")
      : done
        ? t("annotated")
        : `${t("missing")} ${(status?.missing_files || []).join(", ") || t("annotation")}`;
    item.textContent = `${view}: ${detail}`;
    box.appendChild(item);
  }
}

function canvasRle(canvas) {
  const img = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
  const counts = [];
  let last = 0;
  let run = 0;
  for (let i = 3; i < img.length; i += 4) {
    const value = img[i] > 0 ? 1 : 0;
    if (value === last) run += 1;
    else {
      counts.push(run);
      run = 1;
      last = value;
    }
  }
  counts.push(run);
  return { size: [canvas.height, canvas.width], counts };
}

async function saveManual() {
  for (const view of views) {
    const canvas = panel(view, "result").querySelector(".mask");
    const frame = state.resultFrames[view];
    const body = {
      episode: state.episode.episode,
      view,
      frame_index: frame,
      label: state.activeLabel,
      mask_rle: canvasRle(canvas),
    };
    const b = state.boxes[frameKey(view)];
    if (b) body.box_xywh = [b.x, b.y, b.w, b.h];
    const res = await api("/api/manual_edit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const overlay = panel(view, "result").querySelector(".overlay");
    overlay.src = res.overlay_url;
    overlay.classList.add("visible");
  }
  setStatus(t("manualSaved"));
}

async function loadConfig() {
  state.config = await api("/api/config");
  $("videoRoot").value = state.config.video_root;
  $("outputRoot").value = state.config.output_root;
  $("modelBackend").value = state.config.model_backend || "sam21";
  $("checkpointPath").value = state.config.checkpoint || "";
  initGpuSelect(state.config.gpus || [], state.config.gpu_id);
  initLabels(state.config.default_labels);
  setStatus(t("modelStatus", { status: state.config.model_status || "unknown", backend: state.config.model_backend || "sam21" }));
}

function initGpuSelect(gpus, selected) {
  const sel = $("gpuSelect");
  sel.innerHTML = "";
  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = t("defaultGpu");
  sel.appendChild(defaultOption);
  for (const gpu of gpus) {
    const option = document.createElement("option");
    option.value = String(gpu.index);
    option.textContent = `${gpu.index}: ${gpu.name}`;
    sel.appendChild(option);
  }
  sel.value = selected == null ? "" : String(selected);
}

function applyBackendDefaultCheckpoint() {
  const backend = $("modelBackend").value;
  const defaults = state.config?.default_checkpoints || {};
  const current = $("checkpointPath").value.trim();
  const knownDefaults = new Set(Object.values(defaults).filter(Boolean));
  if (!current || knownDefaults.has(current)) {
    $("checkpointPath").value = defaults[backend] || "";
  }
}

async function saveSettings() {
  state.config = await api("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      video_root: $("videoRoot").value.trim(),
      output_root: $("outputRoot").value.trim(),
      gpu_id: $("gpuSelect").value === "" ? null : Number($("gpuSelect").value),
      model_backend: $("modelBackend").value,
      checkpoint: $("checkpointPath").value.trim(),
      bpe_path: state.config?.bpe_path || "",
    }),
  });
  $("modelBackend").value = state.config.model_backend || "sam21";
  $("checkpointPath").value = state.config.checkpoint || "";
}

async function loadEpisodes(preferredEpisode = state.episode?.episode || null, options = {}) {
  const data = await api("/api/episodes");
  const previousEpisode = state.episode?.episode || null;
  state.episodes = data.episodes;
  const sel = $("episodeSelect");
  const toolbarSel = $("toolbarEpisodeSelect");
  sel.innerHTML = "";
  toolbarSel.innerHTML = "";
  for (const ep of state.episodes) {
    const option = document.createElement("option");
    option.value = ep.episode;
    option.textContent = episodeOptionText(ep);
    sel.appendChild(option);
    toolbarSel.appendChild(option.cloneNode(true));
  }
  const selected = preferredEpisode && state.episodes.some((ep) => ep.episode === preferredEpisode)
    ? preferredEpisode
    : state.episodes[0]?.episode;
  if (selected) {
    sel.value = selected;
    toolbarSel.value = selected;
    if (options.preserveCurrent && selected === previousEpisode) {
      state.episode = state.episodes.find((x) => x.episode === selected);
      renderEpisodeAnnotationStatus();
      api("/api/task_queue").then(renderQueueStatus).catch(() => {});
    } else {
      await selectEpisode(selected);
    }
  } else {
    state.episode = null;
    setStatus(t("noEpisodes"));
  }
}

async function refreshDataset() {
  const current = state.episode?.episode || $("episodeSelect").value || null;
  await saveSettings();
  await loadEpisodes(current);
  setStatus(t("refreshed", { videoRoot: $("videoRoot").value, outputRoot: $("outputRoot").value }));
}

async function selectEpisode(name) {
  state.episode = state.episodes.find((x) => x.episode === name);
  $("episodeSelect").value = name;
  $("toolbarEpisodeSelect").value = name;
  state.points = {};
  state.boxes = {};
  state.rawFrames = { head: 0, left: 0, right: 0 };
  state.resultFrames = { head: 0, left: 0, right: 0 };
  renderEpisodeAnnotationStatus();
  for (const view of views) {
    panels(view).forEach((p) => {
      p.querySelector(".overlay").classList.remove("visible");
      p.querySelector(".frame").removeAttribute("src");
      p.querySelectorAll("canvas").forEach((canvas) => canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height));
    });
    await setViewInfo(view);
  }
  loadDraft();
  for (const view of views) {
    setViewFrame("raw", view, state.rawFrames[view] || 0);
    setViewFrame("result", view, state.resultFrames[view] || 0);
  }
  api("/api/task_queue").then(renderQueueStatus).catch(() => {});
}

document.addEventListener("DOMContentLoaded", async () => {
  applyI18n();
  $("languageSelect").addEventListener("change", (event) => setLanguage(event.target.value));
  buildViews();
  applyI18n();
  document.querySelectorAll(".prompt-mode").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".prompt-mode").forEach((x) => x.classList.remove("active"));
      btn.classList.add("active");
      state.promptMode = btn.dataset.mode;
      drawAll();
    });
  });
  document.querySelectorAll(".tool").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tool").forEach((x) => x.classList.remove("active"));
      btn.classList.add("active");
      state.tool = btn.dataset.tool;
      drawAll();
    });
  });
  $("episodeSelect").addEventListener("change", (event) => selectEpisode(event.target.value).catch((e) => setStatus(String(e))));
  $("toolbarEpisodeSelect").addEventListener("change", (event) => selectEpisode(event.target.value).catch((e) => setStatus(String(e))));
  $("prevEpisode").addEventListener("click", () => jumpEpisode(-1));
  $("nextEpisode").addEventListener("click", () => jumpEpisode(1));
  $("prevPromptPoint").addEventListener("click", () => jumpPromptSegment(-1));
  $("nextPromptPoint").addEventListener("click", () => jumpPromptSegment(1));
  $("refreshDataset").addEventListener("click", () => refreshDataset().catch((e) => setStatus(String(e))));
  $("previewAll").addEventListener("click", () => runPreview().catch((e) => setStatus(String(e))));
  $("runAll").addEventListener("click", () => enqueueViews().catch((e) => setStatus(String(e))));
  $("enqueueEpisode").addEventListener("click", () => enqueueEpisode().catch((e) => setStatus(String(e))));
  $("clearFinishedQueue").addEventListener("click", () => clearFinishedQueue().catch((e) => setStatus(String(e))));
  $("releaseModel").addEventListener("click", () => releaseModel().catch((e) => setStatus(String(e))));
  $("saveManual").addEventListener("click", () => saveManual().catch((e) => setStatus(String(e))));
  $("clearCurrent").addEventListener("click", clearCurrent);
  $("modelBackend").addEventListener("change", applyBackendDefaultCheckpoint);
  await loadConfig();
  await loadServerTimeline();
  await loadEpisodes();
  renderQueueStatus(await api("/api/task_queue"));
});
