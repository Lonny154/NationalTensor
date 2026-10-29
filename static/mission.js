// Operation Stolen Cycles: presentation only. The existing native POST forms own approval and execution.
(() => {
  const overlay = document.getElementById("mission-overlay");
  if (!overlay) return;
  const $ = id => document.getElementById(id);
  const plan = JSON.parse(overlay.dataset.plan);
  const replay = JSON.parse(overlay.dataset.replay);
  const initialJob = JSON.parse(overlay.dataset.job);
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  let mode = "live", job = initialJob, timer = null, sound = false, audioContext = null;
  let previousFocus = null;
  const text = (tag, value, className) => {
    const node = document.createElement(tag);
    node.textContent = value == null ? "Not recorded" : String(value);
    if (className) node.className = className;
    return node;
  };
  const button = (label, action) => {
    const node = text("button", label);
    node.type = "button";
    node.className = "action-button";
    node.addEventListener("click", action);
    $("mission-controls").append(node);
  };
  const line = (label, value) => {
    const row = text("div", "", "mission-fact");
    row.append(text("span", label), text("strong", value));
    $("mission-body").append(row);
  };
  const link = (label, href) => {
    const node = text("a", label, "mission-link");
    node.href = href;
    $("mission-body").append(node);
  };
  const story = (part, summary) => {
    const card = text("div", "", "mission-story");
    card.append(text("b", part), text("p", summary));
    $("mission-body").append(card);
  };
  const vaultState = (state, signal) => {
    overlay.dataset.vault = state;
    $("vault-signal").textContent = signal;
  };
  function cue() {
    if (!sound || reduced.matches) return;
    try {
      audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gain = audioContext.createGain();
      oscillator.type = "sine"; oscillator.frequency.value = 620;
      gain.gain.setValueAtTime(0.035, audioContext.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.1);
      oscillator.connect(gain).connect(audioContext.destination);
      oscillator.start(); oscillator.stop(audioContext.currentTime + 0.1);
    } catch (_) { sound = false; $("mission-sound").textContent = "Sound unavailable"; }
  }
  function paint(step, title, quote) {
    $("mission-step").textContent = ({briefing:"01 / BRIEFING", authorization:"02 / AUTHORIZATION", hack:"03 / THE HACK", vault:"04 / THE VAULT", failed:"04 / MISSION REPORT"})[step];
    $("mission-title").textContent = title;
    $("mission-quote").textContent = quote;
    $("mission-body").replaceChildren();
    $("mission-controls").replaceChildren();
    overlay.dataset.scene = step;
    cue();
  }
  function briefing() {
    paint("briefing", "OPERATION STOLEN CYCLES", "“The compiler is hiding something.”");
    vaultState("locked", "LOCKED / AWAITING EVIDENCE");
    story("01 / THE SECRET", "Can we make this matrix multiplication cheaper by moving repeated boundary checks outside its inner loop? The source changes; the compiler may already know the trick.");
    line("Evidence source", mode === "replay" ? "Historical evidence replay · Experiment 003" : "Live trusted diagnostic");
    line("Hypothesis", mode === "replay" ? replay.hypothesis : plan.technical_hypothesis);
    line("GPU target", mode === "replay" ? replay.gpu : "NVIDIA GeForce RTX 4060 Ti");
    line("Workload", mode === "replay" ? replay.shape.join(" × ") + " FP16" : (plan.target_shape_mnk || []).join(" × ") + " FP16");
    button("Inspect the clearance", authorization);
  }
  function authorization() {
    paint("authorization", "THE HUMAN KEY", "“Permission is not ignition.”");
    vaultState("locked", "LOCKED / HUMAN CLEARANCE");
    story("02 / THE PLAN", "Compile the original and reviewed GPU kernels, compare their final machine code, and check that both give correct answers. Only run a speed benchmark if the approved diagnostic calls for one.");
    if (mode === "replay") {
      line("Clearance", "Historical evidence only · no approval or GPU execution");
      button("Enter the archive", hack);
      return;
    }
    line("Approval", overlay.dataset.review || "Awaiting human review");
    line("Execution", job ? job.status : "Not started");
    line("Security", "Approval and execution are separate protected actions below this overlay.");
    if (job) button("Inspect live dispatch", hack);
    else button("Review protected controls", () => {
      close();
      const target = document.getElementById("authorization");
      target.scrollIntoView({behavior: reduced.matches ? "instant" : "smooth"});
      const control = target.querySelector("button:not(:disabled)");
      if (control) control.focus({preventScroll:true});
    });
  }
  function hack() {
    paint("hack", "THE HACK", "“Enhance the compiler. Enhance it again.”");
    vaultState(mode === "replay" ? "archive" : job && ["queued", "running", "validating"].includes(job.status) ? "scanning" : "locked",
      mode === "replay" ? "ARCHIVE SCAN / NO GPU JOB" : job ? "BACKEND STATE / " + job.status.toUpperCase() : "LOCKED / NO JOB");
    story("02 / THE PLAN", mode === "replay" ?
      "We are reading already verified compiler and correctness artifacts. This archive scan is theatrical; it runs no GPU code." :
      "The trusted diagnostic compares compiled kernels and checks correctness. The state below comes from the saved backend job record; the moving beam is theatrical.");
    const terminal = text("div", "", "mission-terminal");
    terminal.append(text("div", "> ACCESSING VERIFIED COMPUTE RECORDS", "mission-terminal-line"));
    terminal.append(text("div", "> THEATRICAL DISPLAY // NOT EXECUTION LOG", "mission-terminal-line"));
    if (mode === "replay") {
      terminal.append(text("div", "> HISTORICAL ARTIFACTS ALREADY VALIDATED", "mission-terminal-line"));
      $("mission-body").append(terminal);
      line("Execution", "Replay only · no job created");
      button("Open the vault", () => vault(replay));
    } else {
      terminal.append(text("div", "> JOB STATE: " + (job ? job.status.toUpperCase() : "NOT STARTED"), "mission-terminal-line"));
      $("mission-body").append(terminal);
      if (job) {
        line("Job ID", job.id);
        line("Backend state", job.status);
        link("View actual job record →", "/research/jobs/" + encodeURIComponent(job.id) + "/job.json");
      }
      if (!job) button("Return to authorization", authorization);
      else if (job.status === "failed" || job.status === "interrupted") failure(job.error || "Execution did not complete.");
      else if (job.status === "completed") fetchEvidence();
      else poll();
    }
  }
  async function fetchEvidence() {
    try {
      if (!job || !job.memory_id || job.memory_id !== "workflow:" + job.id) throw Error("No validated evidence is linked to this job.");
      const response = await fetch(overlay.dataset.evidencePrefix + encodeURIComponent(job.memory_id), {cache:"no-store"});
      if (!response.ok) throw Error("Validated evidence is unavailable.");
      const evidence = await response.json();
      if (evidence.id !== job.memory_id || evidence.warnings.length) throw Error("Evidence validation is incomplete.");
      vault(evidence);
    } catch (error) {
      vaultState("locked", "SEALED / EVIDENCE UNAVAILABLE");
      line("Evidence", error.message + " The vault remains sealed.");
      button("Retry evidence check", () => { hack(); });
    }
  }
  function failure(message) {
    paint("failed", "MISSION FAILED", "“The GPU has filed an incident report.”");
    vaultState("failed", "MISSION STOPPED / " + (job ? job.status.toUpperCase() : "UNKNOWN"));
    story("03 / THE DISCOVERY", "This execution did not produce a validated result. The vault stays locked; the backend record and error log explain what happened.");
    line("Recorded state", job ? job.status : "Unavailable");
    line("Backend error", message);
    if (job) link("Read actual error log →", "/research/jobs/" + encodeURIComponent(job.id) + "/stderr.log");
    button("Close report", close);
  }
  function vault(evidence) {
    const ending = ["equivalent", "improvement", "slowdown", "inconclusive"].includes(evidence.ending) ? evidence.ending : "inconclusive";
    const equivalent = ending === "equivalent" && evidence.executable_identical === true && evidence.runtime_measured === false;
    const headlines = {
      equivalent: ["THE COMPILER GOT THERE FIRST.", "“Two kernels. One executable. The vault was already empty.”"],
      improvement: ["THE TREASURE IS REAL.", "“Confirmed improvement. Someone call the accounting department.”"],
      slowdown: ["THE VAULT BIT BACK.", "“Measured screening slowdown. The escape van is slower than the bicycle.”"],
      inconclusive: ["THE CLUE IS INCOMPLETE.", "“The vault opened, but the map refuses to agree.”"]
    };
    paint("vault", ...headlines[ending]);
    overlay.dataset.ending = ending;
    vaultState("open", "VALIDATED EVIDENCE / VAULT OPEN");
    story("03 / THE DISCOVERY", equivalent ?
      "The compiler made both GPU programs identical. Moving those checks in the source did not change the final executable for this tested setup. No speed benchmark was run." :
      ending === "improvement" ? "Validated measurements confirm an improvement for the recorded test. Read the exact method and limits below." :
      ending === "slowdown" ? "The recorded screening test was slower for this candidate. This is a measurement for this setup, not a verdict on every possible kernel." :
      "The validated evidence does not establish a clear performance win or loss. Read the recorded outcome and measurement limits below.");
    line("Evidence", mode === "replay" ? "Verified historical Experiment 003 · replay" : "Validated live workflow evidence · workflow validation replay");
    line("Outcome", evidence.outcome);
    line("Correctness", evidence.correctness + (evidence.correctness_cases ? " · " + evidence.correctness_cases.join("/") + " checks" : ""));
    line("Compiler", evidence.executable_identical === true ? "Executable bytes identical" : evidence.executable_identical === false ? "Executable bytes differ" : "Comparison unavailable");
    if (evidence.executable_bytes != null) line("Executable section", Number(evidence.executable_bytes).toLocaleString() + " bytes per kernel");
    line("Metadata", evidence.metadata_identical === true ? "Constant data and launch metadata match" : evidence.metadata_identical === false ? "Metadata differs" : "Not recorded");
    line("Compiler finding", evidence.compiler_note);
    line("Runtime", evidence.runtime_measured === false ? "Benchmarking skipped · no runtime performance measurement" : evidence.runtime_measured === true ? "Measured; inspect saved evidence" : "Measurement status unavailable");
    if (evidence.runtime_measured === false) line("Reason", evidence.skip_reason);
    line("Scientific conclusion", equivalent ? "Compiler-equivalent for the tested configuration; no runtime performance measurement." : evidence.scientific_conclusion || evidence.performance);
    if (evidence.screening) line("Recorded performance", evidence.performance);
    link("Inspect full validated evidence →", evidence.evidence_url);
    button("Close dossier", close);
  }
  async function poll() {
    if (timer || mode !== "live" || overlay.hidden) return;
    timer = setInterval(async () => {
      try {
        const response = await fetch(overlay.dataset.statusUrl, {cache:"no-store"});
        if (!response.ok) throw Error("status unavailable");
        const state = await response.json();
        const updated = state.jobs.find(row => row.id === job.id);
        if (!updated) throw Error("job unavailable");
        if (updated.status !== job.status || updated.memory_id !== job.memory_id) {
          job = updated;
          clearInterval(timer); timer = null;
          hack();
        }
      } catch (_) {
        clearInterval(timer); timer = null;
        vaultState("locked", "STATUS UNAVAILABLE / SCAN PAUSED");
        line("Connection", "Status unavailable. Close and reopen the mission to retry.");
      }
    }, 1000);
  }
  function close() {
    overlay.hidden = true;
    document.documentElement.removeAttribute("data-mission-active");
    if (timer) { clearInterval(timer); timer = null; }
    if (previousFocus && previousFocus.isConnected) previousFocus.focus();
    if (mode === "live" && job && initialJob &&
        (job.status !== initialJob.status || job.memory_id !== initialJob.memory_id)) window.location.reload();
  }
  function open(kind, startHack=false) {
    if (kind === "replay" && !replay) return;
    mode = kind;
    previousFocus = document.activeElement;
    overlay.hidden = false;
    document.documentElement.setAttribute("data-mission-active", "");
    $("mission-kind").textContent = kind === "replay" ? "HISTORICAL EVIDENCE REPLAY / NO GPU EXECUTION" : "LIVE MISSION / REAL JOB STATES";
    (startHack && job ? hack : briefing)();
    $("mission-close").focus();
  }
  document.querySelectorAll("[data-mission-open]").forEach(node => node.addEventListener("click", () => open(node.dataset.missionOpen)));
  document.querySelector('[data-presentation-choice="research"]').addEventListener("click", () => {
    if (!overlay.hidden) close();
  });
  $("mission-close").addEventListener("click", close);
  $("mission-effects").addEventListener("click", () => {
    const off = overlay.dataset.effects !== "off";
    overlay.dataset.effects = off ? "off" : "on";
    $("mission-effects").textContent = off ? "Effects off · restore" : "Skip effects";
    $("mission-effects").setAttribute("aria-pressed", String(off));
  });
  $("mission-sound").addEventListener("click", () => {
    sound = !sound;
    $("mission-sound").textContent = sound ? "Sound on · mute" : "Sound off";
    $("mission-sound").setAttribute("aria-pressed", String(sound));
    cue();
  });
  document.addEventListener("keydown", event => {
    if (overlay.hidden) return;
    if (event.key === "Escape") close();
    if (event.key === "Tab") {
      const controls = [...overlay.querySelectorAll("button,a[href]")].filter(node => node.getClientRects().length);
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {event.preventDefault();last.focus();}
      else if (!event.shiftKey && document.activeElement === last) {event.preventDefault();first.focus();}
    }
  });
  const execute = document.querySelector(`form[action$="/${overlay.dataset.planId}/execute"]`);
  if (execute) execute.addEventListener("submit", () => {
    if (document.documentElement.dataset.presentation === "hollywood") {
      try { sessionStorage.setItem("nt-mission-launch", overlay.dataset.planId); } catch (_) {}
    }
  });
  try {
    if (sessionStorage.getItem("nt-mission-launch") === overlay.dataset.planId && job) {
      sessionStorage.removeItem("nt-mission-launch");
      open("live", true);
    }
  } catch (_) {}
})();
