async function loadModels() {
  const response = await fetch("/api/models");
  const models = await response.json();
  const select = document.getElementById("model");
  for (const [key, displayName] of Object.entries(models)) {
    const option = document.createElement("option");
    option.value = key;
    option.textContent = displayName;
    select.appendChild(option);
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  const statusEl = document.getElementById("status");
  statusEl.textContent = "Generating...";

  const formData = new FormData(event.target);

  const response = await fetch("/api/generate", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    const errors = Array.isArray(error.detail) ? error.detail : [String(error.detail)];
    statusEl.innerHTML =
      "<strong>Errors:</strong><ul>" + errors.map((e) => `<li>${e}</li>`).join("") + "</ul>";
    return;
  }

  const blob = await response.blob();
  const domain = document.getElementById("domain").value.trim().replace(/\s+/g, "_") || "project";
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${domain}_finetuning.zip`;
  link.click();
  URL.revokeObjectURL(url);
  statusEl.textContent = "Done! Your zip has downloaded.";
}

document.getElementById("generate-form").addEventListener("submit", handleSubmit);
loadModels();
