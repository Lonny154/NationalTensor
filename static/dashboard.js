(() => {
  const form = document.getElementById("proposal-request");
  if (!form) return;

  const button = form.querySelector("button");
  const status = document.getElementById("proposal-status");
  const title = status.querySelector("b");
  const message = status.querySelector("span");
  const detail = status.querySelector("a");
  const labels = {
    ready: "Ready",
    requesting_proposal: "Requesting proposal",
    validating_response: "Validating response",
    saved_for_review: "Saved for review",
    failed: "Failed"
  };
  let poll = null;

  function show(data) {
    title.textContent = labels[data.state] || "Failed";
    message.textContent = data.message || "Request status is unavailable.";
    status.dataset.state = data.state;
    button.disabled = Boolean(data.active);
    detail.hidden = !data.detail_url;
    if (data.detail_url) {
      detail.href = data.detail_url;
      detail.textContent = data.state === "failed" ? "View original proposal →" : "View saved proposal →";
    }
    if (data.active && poll === null) {
      poll = window.setInterval(refresh, 1000);
    } else if (!data.active && poll !== null) {
      window.clearInterval(poll);
      poll = null;
    }
  }

  async function refresh() {
    try {
      const response = await fetch(form.dataset.statusUrl, {cache: "no-store"});
      if (!response.ok) throw new Error("status unavailable");
      show(await response.json());
    } catch (_) {
      if (poll !== null) {
        window.clearInterval(poll);
        poll = null;
      }
      title.textContent = "Failed";
      message.textContent = "Could not read request status. Reload the page to check again.";
      button.disabled = true;
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    button.disabled = true;
    title.textContent = "Requesting proposal";
    message.textContent = "Sending one proposal request to Muse.";
    detail.hidden = true;
    try {
      const response = await fetch(form.action, {
        method: "POST",
        body: new URLSearchParams(new FormData(form)),
        credentials: "same-origin",
        cache: "no-store"
      });
      if (!response.ok && response.status !== 409) throw new Error("request failed");
      show(await response.json());
    } catch (_) {
      title.textContent = "Failed";
      message.textContent = "Request could not be started. Reload before trying again.";
      button.disabled = true;
    }
  });

  refresh();
})();

(() => {
  const panel = document.getElementById("workflow-status");
  if (!panel) return;
  const initial = JSON.parse(panel.dataset.snapshot);
  const active = data => data.planning.status === "requesting" || data.jobs.some(job => ["queued", "running", "validating"].includes(job.status));
  const state = data => JSON.stringify([data.planning.status, data.jobs.map(job => [job.id, job.status])]);
  if (!active(initial)) return;
  const timer = window.setInterval(async () => {
    try {
      const response = await fetch(panel.dataset.statusUrl, {cache: "no-store"});
      if (!response.ok) throw new Error("status unavailable");
      const current = await response.json();
      if (state(current) !== state(initial)) window.location.reload();
      if (!active(current)) window.clearInterval(timer);
    } catch (_) { window.clearInterval(timer); }
  }, 1500);
})();
