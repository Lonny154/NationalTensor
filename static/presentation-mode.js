// Presentation only: no network requests, form changes, or workflow state writes.
(() => {
  const key = "national-tensor-presentation";
  let selected = "research";
  try { selected = localStorage.getItem(key) === "hollywood" ? "hollywood" : "research"; } catch (_) {}
  document.documentElement.dataset.presentation = selected;
  document.addEventListener("DOMContentLoaded", () => {
    const buttons = document.querySelectorAll("[data-presentation-choice]");
    // Move the original controls, never clone them: CSRF fields, selected plan,
    // listeners, focusable controls, and the polling status node retain identity.
    const transfers = [...document.querySelectorAll("[data-hollywood-source]")].map(node => {
      const slot = document.querySelector(`[data-hollywood-slot="${node.dataset.hollywoodSource}"]`);
      const marker = document.createComment("Research Mode position");
      node.before(marker);
      return {node, slot, marker};
    });
    function show(mode) {
      document.documentElement.dataset.presentation = mode;
      transfers.forEach(({node, slot, marker}) => {
        if (mode === "hollywood" && slot) slot.append(node);
        else marker.after(node);
      });
      buttons.forEach(button => button.setAttribute("aria-pressed", String(button.dataset.presentationChoice === mode)));
    }
    buttons.forEach(button => button.addEventListener("click", () => {
      const mode = button.dataset.presentationChoice;
      show(mode);
      try { localStorage.setItem(key, mode); } catch (_) {}
    }));
    show(selected);
  });
})();
