const $ = (sel) => document.querySelector(sel);

const loaderPanel = $("#loader-panel");
const peopleView = $("#people-view");
const detailView = $("#detail-view");

const folderInput = $("#folder-input");
const processBtn = $("#process-btn");
const statusLine = $("#status-line");
const statusFill = $("#status-fill");
const statusText = $("#status-text");

const peopleGrid = $("#people-grid");
const peopleCount = $("#people-count");
const photoGrid = $("#photo-grid");
const personLabelInput = $("#person-label-input");

const lightbox = $("#lightbox");
const lightboxImg = $("#lightbox-img");

let currentJobId = null;
let currentPersonId = null;
let pollHandle = null;

const STATUS_PROGRESS = {
  pending: 5,
  connecting: 10,
  listing: 15,
  downloading: 25,
  detecting: 70,
  clustering: 90,
  done: 100,
  error: 100,
};

processBtn.addEventListener("click", startProcessing);
folderInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") startProcessing();
});

$("#back-to-loader").addEventListener("click", () => {
  showOnly(loaderPanel);
  folderInput.value = "";
  statusLine.hidden = true;
});

$("#back-to-people").addEventListener("click", () => {
  showOnly(peopleView);
});

$("#lightbox-close").addEventListener("click", () => (lightbox.hidden = true));
lightbox.addEventListener("click", (e) => {
  if (e.target === lightbox) lightbox.hidden = true;
});

personLabelInput.addEventListener("change", async () => {
  if (!currentPersonId) return;
  await fetch(`/api/people/${currentPersonId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label: personLabelInput.value }),
  });
});

function showOnly(section) {
  [loaderPanel, peopleView, detailView].forEach((s) => (s.hidden = s !== section));
}

async function startProcessing() {
  const folderId = folderInput.value.trim();
  if (!folderId) {
    folderInput.focus();
    return;
  }

  processBtn.disabled = true;
  statusLine.hidden = false;
  statusText.classList.remove("error");
  statusText.textContent = "Starting...";
  statusFill.style.width = "5%";

  const res = await fetch("/api/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder_id: folderId }),
  });

  if (!res.ok) {
    statusText.textContent = "Couldn't start the job. Check the server logs.";
    statusText.classList.add("error");
    processBtn.disabled = false;
    return;
  }

  const { job_id } = await res.json();
  currentJobId = job_id;
  pollJob();
}

function pollJob() {
  clearTimeout(pollHandle);

  fetch(`/api/jobs/${currentJobId}`)
    .then((r) => r.json())
    .then((job) => {
      const pct = STATUS_PROGRESS[job.status] ?? 50;
      statusFill.style.width = `${pct}%`;
      statusText.textContent = job.message || job.status;

      if (job.status === "error") {
        statusText.classList.add("error");
        processBtn.disabled = false;
        return;
      }

      if (job.status === "done") {
        processBtn.disabled = false;
        loadPeople();
        return;
      }

      pollHandle = setTimeout(pollJob, 1200);
    });
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
        <img src="/api/faces/${person.representative_face_id}/thumbnail" alt="${person.label}" loading="lazy" />
      </div>
      <div class="person-name">${person.label}</div>
      <div class="person-meta">${person.photo_count} photo${person.photo_count === 1 ? "" : "s"}</div>
    `;
    card.addEventListener("click", () => openPerson(person));
    peopleGrid.appendChild(card);
  });

  showOnly(peopleView);
}

async function openPerson(person) {
  currentPersonId = person.id;
  personLabelInput.value = person.label;

  const photos = await fetch(`/api/people/${person.id}/photos`).then((r) => r.json());

  photoGrid.innerHTML = "";
  photos.forEach((photo) => {
    const img = document.createElement("img");
    img.src = photo.url;
    img.alt = photo.filename;
    img.loading = "lazy";
    img.addEventListener("click", () => {
      lightboxImg.src = photo.url;
      lightbox.hidden = false;
    });
    photoGrid.appendChild(img);
  });

  showOnly(detailView);
}
