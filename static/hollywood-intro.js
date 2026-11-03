// Fictional, presentation-only opening. No requests, forms, GPU work, or research state.
(() => {
  const overlay = document.getElementById("hollywood-intro");
  if (!overlay) return;
  const $ = id => document.getElementById(id);
  const stage = $("hollywood-intro-stage");
  const scenes = [
    {id: "opening", beats: 4,
      lines: ["MILLISECONDS MATTER", "BILLIONS OF OPERATIONS", "ONE DEVELOPER", "CAN WE GO FASTER?"]},
    {id: "enhance", beats: 4,
      lines: ["COMMANDER: ENHANCE THE KERNEL!", "ENGINEER: SIR, THAT'S NOT HOW OPTIMIZATION WORKS!", "COMMANDER: THEN ENHANCE THE BENCHMARK!", "ENGINEER: SIR, WE NEED STATISTICAL EVIDENCE!"]},
    {id: "title", beats: 4}
  ];
  let sceneIndex = 0, beatIndex = 0, previousFocus = null;
  const node = (tag, className, value) => {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (value !== undefined) item.textContent = value;
    return item;
  };
  const matrixBlock = values => {
    const block = node("div", "intro-matrix-block");
    for (const value of values) block.append(node("span", "", value));
    return block;
  };
  const isFirst = () => sceneIndex === 0 && beatIndex === 0;
  const isLast = () => sceneIndex === scenes.length - 1 && beatIndex === scenes[sceneIndex].beats - 1;
  function render() {
    const scene = scenes[sceneIndex];
    stage.dataset.scene = scene.id;
    stage.dataset.beat = String(beatIndex);
    stage.replaceChildren();
    stage.append(node("p", "intro-scene-label", `SCENE ${String(sceneIndex + 1).padStart(2, "0")} / ${String(scenes.length).padStart(2, "0")} · DRAMATIZED`));
    if (scene.id === "opening") {
      const matrices = node("div", "intro-matrix");
      matrices.append(matrixBlock(["1", "0", "0", "1"]), node("span", "intro-matrix-op", "×"),
        matrixBlock(["1", "1", "0", "1"]), node("span", "intro-matrix-op", "="),
        matrixBlock(["1", "1", "0", "1"]));
      stage.append(matrices);
      stage.append(node("h2", "intro-big-line", scene.lines[beatIndex]));
    } else if (scene.id === "enhance") {
      stage.append(node("h2", "intro-console-title", "ENHANCE THE KERNEL"));
      const exchange = node("div", "intro-exchange");
      exchange.append(node("p", beatIndex % 2 ? "intro-engineer" : "intro-commander", scene.lines[beatIndex]));
      stage.append(exchange);
    } else if (beatIndex <= 1) {
      stage.append(node("h2", "intro-console-title", "AI ANALYSIS INTERFACE"));
      stage.append(node("p", "intro-analysis", "ANALYZING GPU PERFORMANCE..."));
      stage.append(node("p", "intro-fiction-note", "FICTIONAL SYSTEM / NO LIVE GPU"));
    } else if (beatIndex === 2) {
      stage.append(node("h2", "intro-error", "CUDA OUT OF MEMORY"));
      stage.append(node("p", "intro-fiction-note", "FICTIONAL PUNCHLINE / NOT AN EXPERIMENT RESULT"));
    } else {
      stage.append(node("h2", "intro-title", "NATIONAL TENSOR"));
      stage.append(node("p", "intro-tagline", "THIS TIME, THE FLOPs ARE PERSONAL"));
    }
    $("hollywood-intro-progress").textContent = `SCENE ${String(sceneIndex + 1).padStart(2, "0")} / ${String(scenes.length).padStart(2, "0")} · BEAT ${String(beatIndex + 1).padStart(2, "0")} / ${String(scene.beats).padStart(2, "0")}`;
    $("hollywood-intro-previous").disabled = isFirst();
    $("hollywood-intro-next").disabled = isLast();
    $("hollywood-intro-research").hidden = !isLast();
  }
  function move(direction) {
    if (direction > 0 && !isLast()) {
      if (++beatIndex === scenes[sceneIndex].beats) { sceneIndex++; beatIndex = 0; }
    } else if (direction < 0 && !isFirst()) {
      if (--beatIndex < 0) { sceneIndex--; beatIndex = scenes[sceneIndex].beats - 1; }
    } else return;
    render();
    stage.focus();
  }
  function restart() {
    sceneIndex = 0;
    beatIndex = 0;
    overlay.dataset.motion = "running";
    $("hollywood-intro-play").textContent = "Pause motion";
    $("hollywood-intro-play").setAttribute("aria-pressed", "false");
    render();
    stage.focus();
  }
  function toggleMotion() {
    const paused = overlay.dataset.motion !== "paused";
    overlay.dataset.motion = paused ? "paused" : "running";
    $("hollywood-intro-play").textContent = paused ? "Resume motion" : "Pause motion";
    $("hollywood-intro-play").setAttribute("aria-pressed", String(paused));
    stage.focus();
  }
  function close() {
    overlay.hidden = true;
    if (previousFocus && previousFocus.isConnected) previousFocus.focus();
  }
  $("hollywood-intro-open").addEventListener("click", () => {
    previousFocus = document.activeElement;
    overlay.hidden = false;
    overlay.dataset.controls = "visible";
    overlay.querySelector(".hollywood-intro-header").inert = false;
    overlay.querySelector(".hollywood-intro-footer").inert = false;
    restart();
  });
  $("hollywood-intro-close").addEventListener("click", close);
  $("hollywood-intro-previous").addEventListener("click", () => move(-1));
  $("hollywood-intro-next").addEventListener("click", () => move(1));
  $("hollywood-intro-restart").addEventListener("click", restart);
  $("hollywood-intro-play").addEventListener("click", toggleMotion);
  $("hollywood-intro-research").addEventListener("click", () => {
    close();
    document.querySelector('[data-presentation-choice="research"]').click();
    window.location.assign(overlay.dataset.overviewUrl);
  });
  document.addEventListener("keydown", event => {
    if (overlay.hidden) return;
    if (event.key === "Escape") { event.preventDefault(); close(); return; }
    if (event.key === "Tab") {
      if (overlay.dataset.controls === "hidden") { event.preventDefault(); return; }
      const controls = [...overlay.querySelectorAll("button:not([hidden]):not(:disabled)")];
      const first = controls[0], last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      return;
    }
    const interactive = "input, textarea, select, button, a, [contenteditable], [role='button']";
    if (document.activeElement?.closest(interactive) ||
        (event.target instanceof Element && event.target.closest(interactive))) return;
    if ([" ", "ArrowRight", "ArrowLeft", "r", "R", "p", "P", "h", "H"].includes(event.key)) event.preventDefault();
    if (event.key === " " || event.key === "ArrowRight") move(1);
    else if (event.key === "ArrowLeft") move(-1);
    else if (event.key.toLowerCase() === "r") restart();
    else if (event.key.toLowerCase() === "p") toggleMotion();
    else if (event.key.toLowerCase() === "h") {
      overlay.dataset.controls = overlay.dataset.controls === "hidden" ? "visible" : "hidden";
      const hidden = overlay.dataset.controls === "hidden";
      overlay.querySelector(".hollywood-intro-header").inert = hidden;
      overlay.querySelector(".hollywood-intro-footer").inert = hidden;
      stage.focus();
    }
  });
})();
