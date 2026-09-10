// Konami-code easter egg: ↑ ↑ ↓ ↓ ← → ← → then any two keys is the classic
// version, but we trigger on the arrow sequence alone to keep it simple.
(function () {
  const sequence = [
    "ArrowUp",
    "ArrowUp",
    "ArrowDown",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "ArrowLeft",
    "ArrowRight",
  ];
  let position = 0;

  window.addEventListener("keydown", (event) => {
    const expected = sequence[position];
    if (event.key === expected) {
      position += 1;
      if (position === sequence.length) {
        triggerEasterEgg();
        position = 0;
      }
    } else {
      position = event.key === sequence[0] ? 1 : 0;
    }
  });

  function triggerEasterEgg() {
    const overlay = document.createElement("div");
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-modal", "true");
    overlay.className =
      "fixed inset-0 z-50 flex items-center justify-center bg-slate-950/90 p-4 text-center";
    overlay.innerHTML = `
      <div class="max-w-sm space-y-4">
        <p class="text-5xl">🛰️✨🔭</p>
        <h2 class="text-xl font-bold text-amber-300">You found it.</h2>
        <p class="text-slate-300">
          Somewhere, an agent is quietly proud of you for reading the footer.
        </p>
        <button
          id="close-easter-egg"
          class="px-4 py-2 rounded-md bg-amber-400 text-slate-950 font-medium hover:bg-amber-300"
        >
          Back to work
        </button>
      </div>
    `;
    document.body.appendChild(overlay);
    document.getElementById("close-easter-egg")?.addEventListener("click", () => {
      overlay.remove();
    });
  }
})();
