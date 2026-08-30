const $ = (sel) => document.querySelector(sel);

const loaderPanel = $("#loader-panel");
const completionPanel = $("#completion-panel");
const peopleView = $("#people-view");
const detailView = $("#detail-view");

const foldersContainer = $("#folders-container");
const addFolderBtn = $("#add-folder-btn");
const processBtn = $("#process-btn");
const statusLine = $("#status-line");
const statusFill = $("#status-fill");
const statusText = $("#status-text");

// Navigation buttons
const completionBackBtn = $("#completion-back-btn");
const completionViewBtn = $("#completion-view-btn");
const backToHomeBtn = $("#back-to-home");

// Stats elements
const statDiscovered = $("#stat-discovered");
const statSkipped = $("#stat-skipped");
const statProcessed = $("#stat-processed");
const statFaces = $("#stat-faces");

const peopleGrid = $("#people-grid");
const peopleCount = $("#people-count");
const photoGrid = $("#photo-grid");
const personLabelInput = $("#person-label-input");

// Selection elements
const selectedCount = $("#selected-count");
const selectAllBtn = $("#select-all-btn");
const deselectAllBtn = $("#deselect-all-btn");
const downloadSelectedBtn = $("#download-selected-btn");
const downloadAllBtn = $("#download-all-btn");

const lightbox = $("#lightbox");
const lightboxImg = $("#lightbox-img");

let currentJobId = null; // Opaque public token string
let currentPersonId = null;
let pollHandle = null;
let selectedPhotoIds = new Set();

const STATUS_PROGRESS = {
  pending: 5,
  connecting: 10,
  listing: 20,
  downloading: 35,
  detecting: 70,
  clustering: 90,
  done: 100,
  error: 100,
};

// --- Multiple folders input management ---
function createFolderInputRow(value = "") {
  const row = document.createElement("div");
  row.className = "folder-input-row";
  row.innerHTML = `
    <input type="text" class="folder-link-input" placeholder="Paste Google Drive folder URL or ID (shared as Anyone with the link \u2192 Viewer)" value="${value}" autocomplete="off" />
    <button class="remove-folder-btn" type="button" aria-label="Remove input">&times;</button>
  `;
  
  // Set delete listener
  row.querySelector(".remove-folder-btn").addEventListener("click", () => {
    // Only remove if there are more than 1 input rows
    if (document.querySelectorAll(".folder-input-row").length > 1) {
      row.remove();
    } else {
      alert("At least one folder link is required.");
    }
  });

  return row;
}

// Initialize single initial row
foldersContainer.innerHTML = "";
foldersContainer.appendChild(createFolderInputRow());


addFolderBtn.addEventListener("click", () => {
  foldersContainer.appendChild(createFolderInputRow());
  const inputs = document.querySelectorAll(".folder-link-input");
  inputs[inputs.length - 1].focus();
});

processBtn.addEventListener("click", startProcessing);

// Reset frontend state to Home screen
function resetToHome() {
  clearTimeout(pollHandle);
  pollHandle = null;
  currentJobId = null;
  currentPersonId = null;
  selectedPhotoIds.clear();
  
  // Re-enable form controls
  processBtn.disabled = false;
  addFolderBtn.disabled = false;
  document.querySelectorAll(".remove-folder-btn").forEach(btn => btn.disabled = false);
  document.querySelectorAll(".folder-link-input").forEach(inp => inp.disabled = false);

  // Reset statuses & counters
  statusFill.style.width = "5%";
  statusText.textContent = "";
  statusText.classList.remove("error");
  statusLine.hidden = true;

  statDiscovered.textContent = "0";
  statSkipped.textContent = "0";
  statProcessed.textContent = "0";
  statFaces.textContent = "0";

  showOnly(loaderPanel);
}

// Navigation wire-ups
completionBackBtn.addEventListener("click", resetToHome);
completionViewBtn.addEventListener("click", loadPeople);
backToHomeBtn.addEventListener("click", resetToHome);

$("#back-to-people").addEventListener("click", () => {
  showOnly(peopleView);
});

$("#lightbox-close").addEventListener("click", () => (lightbox.hidden = true));
lightbox.addEventListener("click", (e) => {
  if (e.target === lightbox) lightbox.hidden = true;
});

personLabelInput.addEventListener("change", async () => {
  if (!currentPersonId || !currentJobId) return;
  await fetch(`/api/people/${currentPersonId}?token=${currentJobId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label: personLabelInput.value }),
  });
});

function showOnly(section) {
  [loaderPanel, completionPanel, peopleView, detailView].forEach((s) => (s.hidden = s !== section));
}

// --- Start processing with multi-link and duplicate link validation ---
async function startProcessing() {
  const inputs = Array.from(document.querySelectorAll(".folder-link-input"));
  const folderLinks = inputs.map(inp => inp.value.trim()).filter(Boolean);

  if (folderLinks.length === 0) {
    if (inputs[0]) inputs[0].focus();
    alert("Please paste a Google Drive folder link or folder ID.");
    return;
  }

  // Check for blank fields in dynamic list
  const hasBlank = inputs.some(inp => !inp.value.trim());
  if (hasBlank) {
    alert("Please fill in all folder fields or remove empty ones.");
    return;
  }

  // Detect exact duplicate links
  const duplicates = folderLinks.filter((lnk, idx) => folderLinks.indexOf(lnk) !== idx);
  if (duplicates.length > 0) {
    alert(`Duplicate links detected:\n${duplicates[0]}\nPlease remove duplicate inputs before starting.`);
    return;
  }

  processBtn.disabled = true;
  addFolderBtn.disabled = true;
  document.querySelectorAll(".remove-folder-btn").forEach(btn => btn.disabled = true);
  document.querySelectorAll(".folder-link-input").forEach(inp => inp.disabled = true);

  statusLine.hidden = false;
  statusText.classList.remove("error");
  statusText.textContent = "Connecting to Google Drive...";
  statusFill.style.width = "10%";
  
  // Clear statistics counters
  statDiscovered.textContent = "0";
  statSkipped.textContent = "0";
  statProcessed.textContent = "0";
  statFaces.textContent = "0";

  try {
    const res = await fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folder_links: folderLinks }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Failed to initiate processing." }));
      statusText.textContent = err.detail || "Couldn't initiate processing. Check the server logs.";
      statusText.classList.add("error");
      processBtn.disabled = false;
      addFolderBtn.disabled = false;
      document.querySelectorAll(".remove-folder-btn").forEach(btn => btn.disabled = false);
      document.querySelectorAll(".folder-link-input").forEach(inp => inp.disabled = false);
      return;
    }

    const data = await res.json();
    currentJobId = data.job_id;
    pollJob();
  } catch (err) {
    console.error(err);
    statusText.textContent = "Error connecting to the server.";
    statusText.classList.add("error");
    processBtn.disabled = false;
    addFolderBtn.disabled = false;
    document.querySelectorAll(".remove-folder-btn").forEach(btn => btn.disabled = false);
    document.querySelectorAll(".folder-link-input").forEach(inp => inp.disabled = false);
  }
}


async function pollJob() {
  clearTimeout(pollHandle);

  try {
    const res = await fetch(`/api/jobs/${currentJobId}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Job not found" }));
      statusText.textContent = err.detail || `Job not found (${res.status}). Please try starting the process again.`;
      statusText.classList.add("error");
      processBtn.disabled = false;
      addFolderBtn.disabled = false;
      document.querySelectorAll(".remove-folder-btn").forEach(btn => btn.disabled = false);
      document.querySelectorAll(".folder-link-input").forEach(inp => inp.disabled = false);
      return; // Stop polling immediately on 404/error
    }

    const job = await res.json();
    const pct = STATUS_PROGRESS[job.status] ?? 50;
    statusFill.style.width = `${pct}%`;
    statusText.textContent = job.message || job.status;

    // Update stats cards dynamically
    statDiscovered.textContent = job.total_files || 0;
    statSkipped.textContent = job.duplicate_files_skipped || 0;
    statProcessed.textContent = job.processed_files || 0;
    statFaces.textContent = job.faces_count || 0;

    if (job.status === "error") {
      statusText.classList.add("error");
      processBtn.disabled = false;
      addFolderBtn.disabled = false;
      document.querySelectorAll(".remove-folder-btn").forEach(btn => btn.disabled = false);
      document.querySelectorAll(".folder-link-input").forEach(inp => inp.disabled = false);
      return;
    }

    if (job.status === "done") {
      processBtn.disabled = false;
      addFolderBtn.disabled = false;
      document.querySelectorAll(".remove-folder-btn").forEach(btn => btn.disabled = false);
      document.querySelectorAll(".folder-link-input").forEach(inp => inp.disabled = false);
      showCompletionPanel(job);
      return;
    }

    pollHandle = setTimeout(pollJob, 1200);
  } catch (err) {
    console.error("Polling network error:", err);
    pollHandle = setTimeout(pollJob, 2000);
  }
}



async function showCompletionPanel(job) {
  const people = await fetch(`/api/jobs/${currentJobId}/people`).then((r) => r.json());
  
  // Set summary numbers
  $("#final-discovered").textContent = job.total_files || 0;
  $("#final-skipped").textContent = job.duplicate_files_skipped || 0;
  $("#final-processed").textContent = job.processed_files || 0;
  $("#final-faces").textContent = job.faces_count || 0;
  $("#final-people").textContent = people.length;

  showOnly(completionPanel);
}

async function loadPeople() {
  const people = await fetch(`/api/jobs/${currentJobId}/people`).then((r) => r.json());

  peopleCount.textContent = `(${people.length})`;
  peopleGrid.innerHTML = "";

  people.forEach((person, idx) => {
    const card = document.createElement("div");
    card.className = "person-card";
    card.innerHTML = `
      <div class="person-frame">
        <span class="frame-number">No. ${String(idx + 1).padStart(3, "0")}</span>
        <img src="/api/faces/${person.representative_face_id}/thumbnail?token=${currentJobId}" alt="${person.label}" loading="lazy" />
      </div>
      <div class="person-name">${person.label}</div>
      <div class="person-meta">${person.photo_count} photo${person.photo_count === 1 ? "" : "s"}</div>
    `;
    card.addEventListener("click", () => openPerson(person));
    peopleGrid.appendChild(card);
  });

  showOnly(peopleView);
}

// --- Photo Selection & ZIP downloads ---

async function openPerson(person) {
  currentPersonId = person.id;
  personLabelInput.value = person.label;

  // Clear selections
  selectedPhotoIds.clear();
  updateSelectionUI();

  const photos = await fetch(`/api/people/${person.id}/photos?token=${currentJobId}`).then((r) => r.json());

  photoGrid.innerHTML = "";
  photos.forEach((photo) => {
    const card = document.createElement("div");
    card.className = "photo-card";
    card.innerHTML = `
      <input type="checkbox" class="photo-checkbox" data-id="${photo.id}" />
      <img src="${photo.url}" alt="${photo.filename}" loading="lazy" />
    `;

    const img = card.querySelector("img");
    const checkbox = card.querySelector(".photo-checkbox");

    // Click checkbox toggles selection
    checkbox.addEventListener("change", (e) => {
      e.stopPropagation();
      const id = parseInt(checkbox.dataset.id);
      if (checkbox.checked) {
        selectedPhotoIds.add(id);
        card.classList.add("selected");
      } else {
        selectedPhotoIds.delete(id);
        card.classList.remove("selected");
      }
      updateSelectionUI();
    });

    // Clicking image opens lightbox (existing behaviour)
    img.addEventListener("click", (e) => {
      e.stopPropagation();
      lightboxImg.src = photo.url;
      lightbox.hidden = false;
    });

    // Clicking anywhere else on card toggles checkbox
    card.addEventListener("click", (e) => {
      if (e.target !== checkbox && e.target !== img) {
        checkbox.checked = !checkbox.checked;
        checkbox.dispatchEvent(new Event("change"));
      }
    });

    photoGrid.appendChild(card);
  });

  showOnly(detailView);
}

function updateSelectionUI() {
  selectedCount.textContent = `${selectedPhotoIds.size} selected`;
  downloadSelectedBtn.disabled = selectedPhotoIds.size === 0;
}

// Select All
selectAllBtn.addEventListener("click", () => {
  document.querySelectorAll(".photo-card").forEach((card) => {
    const checkbox = card.querySelector(".photo-checkbox");
    if (checkbox) {
      const id = parseInt(checkbox.dataset.id);
      selectedPhotoIds.add(id);
      checkbox.checked = true;
      card.classList.add("selected");
    }
  });
  updateSelectionUI();
});

// Deselect All
deselectAllBtn.addEventListener("click", () => {
  document.querySelectorAll(".photo-card").forEach((card) => {
    const checkbox = card.querySelector(".photo-checkbox");
    if (checkbox) {
      const id = parseInt(checkbox.dataset.id);
      selectedPhotoIds.delete(id);
      checkbox.checked = false;
      card.classList.remove("selected");
    }
  });
  updateSelectionUI();
});

// Download Selected ZIP
downloadSelectedBtn.addEventListener("click", async () => {
  if (selectedPhotoIds.size === 0) return;
  
  downloadSelectedBtn.disabled = true;
  const originalText = downloadSelectedBtn.textContent;
  downloadSelectedBtn.textContent = "Downloading...";

  try {
    const res = await fetch("/api/photos/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        photo_ids: Array.from(selectedPhotoIds),
        public_job_token: currentJobId
      })
    });

    if (res.ok) {
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `selected_photos_${new Date().toISOString().slice(0,10)}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } else {
      alert("Failed to download selected photos. Check server logs.");
    }
  } catch (err) {
    console.error(err);
    alert("Error downloading selected photos.");
  } finally {
    downloadSelectedBtn.disabled = false;
    downloadSelectedBtn.textContent = originalText;
  }
});

// Download All for Person ZIP
downloadAllBtn.addEventListener("click", async () => {
  if (!currentPersonId || !currentJobId) return;

  downloadAllBtn.disabled = true;
  const originalText = downloadAllBtn.textContent;
  downloadAllBtn.textContent = "Downloading...";

  try {
    const res = await fetch(`/api/people/${currentPersonId}/download`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        public_job_token: currentJobId
      })
    });

    if (res.ok) {
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const name = personLabelInput.value.trim().replace(/\s+/g, "_") || `person_${currentPersonId}`;
      a.download = `${name}_photos.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } else {
      alert("Failed to download person photos. Check server logs.");
    }
  } catch (err) {
    console.error(err);
    alert("Error downloading photos.");
  } finally {
    downloadAllBtn.disabled = false;
    downloadAllBtn.textContent = originalText;
  }
});
