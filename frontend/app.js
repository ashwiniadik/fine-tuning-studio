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

  let response;
  try {
    response = await fetch("/api/generate", {
      method: "POST",
      body: formData,
    });
  } catch (err) {
    statusEl.textContent = "Something went wrong. Please try again.";
    return;
  }

  if (!response.ok) {
    let errors;
    try {
      const error = await response.json();
      errors = Array.isArray(error.detail) ? error.detail : [String(error.detail)];
    } catch (err) {
      statusEl.textContent = "Something went wrong. Please try again.";
      return;
    }
    statusEl.replaceChildren();
    const strong = document.createElement("strong");
    strong.textContent = "Errors:";
    const ul = document.createElement("ul");
    for (const err of errors) {
      const li = document.createElement("li");
      li.textContent = err;
      ul.appendChild(li);
    }
    statusEl.appendChild(strong);
    statusEl.appendChild(ul);
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
