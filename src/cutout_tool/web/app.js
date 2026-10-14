const state = { token: "", file: null, task: null, timer: null, objectUrls: [] };
const $ = (id) => document.getElementById(id);

function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-Cutout-Dev-Token", state.token);
  return fetch(path, { ...options, headers });
}

function showConnection(text, good = false) {
  $("connection").textContent = text;
  $("connection").className = good ? "muted good" : "muted";
}

async function connect() {
  state.token = $("token").value;
  if (!state.token) return showConnection("请输入 token");
  try {
    const response = await api("/api/health");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "连接失败");
    showConnection(`已连接 · ${data.model}`, true);
    await refreshHistory();
  } catch (error) {
    showConnection(error.message);
  }
}

function setFile(file) {
  if (!file) return;
  state.file = file;
  $("selected").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  $("process").disabled = false;
}

function setTask(task) {
  state.task = task;
  $("resultPanel").hidden = false;
  $("taskTitle").textContent = task.original_name || "处理结果";
  $("status").textContent = task.status;
  $("status").className = `status ${task.status}`;
  $("stage").textContent = task.stage || "";
  $("error").textContent = task.error || "";
  if (task.status === "succeeded") showResult(task);
}

async function processFile() {
  if (!state.file) return;
  $("process").disabled = true;
  const body = new FormData();
  body.append("file", state.file);
  const response = await api("/api/tasks", { method: "POST", body });
  const task = await response.json();
  if (!response.ok) {
    $("process").disabled = false;
    return setTask({ original_name: state.file.name, status: "failed", stage: "failed", error: task.error });
  }
  setTask(task);
  poll(task.task_id);
}

function poll(taskId) {
  clearTimeout(state.timer);
  state.timer = setTimeout(async () => {
    const response = await api(`/api/tasks/${taskId}`);
    if (!response.ok) return;
    const task = await response.json();
    setTask(task);
    if (task.status === "queued" || task.status === "processing") return poll(taskId);
    $("process").disabled = false;
    await refreshHistory();
  }, 700);
}

function fileUrl(taskId, kind) { return `/api/tasks/${taskId}/files/${kind}`; }

async function loadImage(path, element) {
  const response = await api(path);
  if (!response.ok) throw new Error("结果文件读取失败");
  const url = URL.createObjectURL(await response.blob());
  state.objectUrls.push(url);
  element.src = url;
}

async function showResult(task) {
  $("viewer").hidden = false;
  await loadImage(fileUrl(task.task_id, "original"), $("original"));
  await loadImage(fileUrl(task.task_id, "transparent"), $("preview"));
}

async function selectPreview(kind) {
  if (!state.task || state.task.status !== "succeeded") return;
  document.querySelectorAll("[data-kind]").forEach((button) => button.classList.toggle("active", button.dataset.kind === kind));
  await loadImage(fileUrl(state.task.task_id, kind), $("preview"));
  $("preview").parentElement.classList.toggle("checker", kind === "transparent");
}

async function refreshHistory() {
  const response = await api("/api/tasks");
  if (!response.ok) return;
  const data = await response.json();
  const history = $("history");
  history.innerHTML = data.tasks.length ? "" : '<p class="muted">暂无任务</p>';
  data.tasks.forEach((task) => {
    const row = document.createElement("button");
    row.className = "history-row";
    row.innerHTML = `<span>${task.original_name}</span><span class="${task.status}">${task.status}</span>`;
    row.onclick = () => { setTask(task); if (task.status === "succeeded") showResult(task); };
    history.appendChild(row);
  });
}

$("connect").onclick = connect;
$("choose").onclick = () => $("file").click();
$("file").onchange = (event) => setFile(event.target.files[0]);
$("process").onclick = processFile;
$("refresh").onclick = refreshHistory;
$("dropzone").onclick = (event) => { if (event.target.id !== "choose") $("file").click(); };
$("dropzone").ondragover = (event) => { event.preventDefault(); $("dropzone").classList.add("dragging"); };
$("dropzone").ondragleave = () => $("dropzone").classList.remove("dragging");
$("dropzone").ondrop = (event) => { event.preventDefault(); $("dropzone").classList.remove("dragging"); setFile(event.dataTransfer.files[0]); };
$("zoom").oninput = (event) => { const scale = `${event.target.value}%`; document.querySelectorAll(".image-frame img").forEach((image) => { image.style.width = scale; }); };
document.querySelectorAll("[data-kind]").forEach((button) => { button.onclick = () => selectPreview(button.dataset.kind); });
$("download").onclick = async () => {
  if (!state.task) return;
  const response = await api(fileUrl(state.task.task_id, "transparent"));
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a"); link.href = url; link.download = `${state.task.original_name.replace(/\.[^.]+$/, "")}_transparent.png`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
