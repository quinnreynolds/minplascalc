const progressBar = document.querySelector(".reading-progress span");

const benchmarks = {
  equilibrium: {
    context: "SiCO · 20 temperatures · 3 ratios",
    title: "Equilibrium sweep",
    speedup: "2.62× faster",
    oldLabel: "Production",
    oldValue: "0.297 s",
    newLabel: "Log Newton",
    newValue: "0.114 s",
    newWidth: "38.2%",
    note: "2574 production iterations versus 610 log iterations. Maximum absolute mole-fraction difference: 1.24×10⁻⁸.",
    lesson: "The dense linear solve is only about 4% of prototype runtime. Thermodynamic evaluation—not linear algebra—is the remaining optimization target.",
  },
  capacity: {
    context: "SiCO · 20 temperatures · 3 ratios",
    title: "Heat-capacity sweep",
    speedup: "2.68× faster",
    oldLabel: "3-temperature",
    oldValue: "1.006 s",
    newLabel: "Analytical",
    newValue: "0.376 s",
    newWidth: "37.4%",
    note: "The old 10⁻³ difference is replaced by the exact enthalpy chain rule. A fine scan found 43.5 J/(kg·K) step sensitivity near cutoffs.",
    lesson: "This gain is expected: the old method performs complete solves at T−ΔT, T, and T+ΔT. The analytical tangent needs one converged state.",
  },
  collision: {
    context: "16 species · 16 collision moments · 12,000 K",
    title: "Compiled collision kernel floor",
    speedup: "4.39× kernel",
    oldLabel: "Numba",
    oldValue: "277.3 µs",
    newLabel: "Fortran -O3",
    newValue: "63.2 µs",
    newWidth: "22.8%",
    note: "Agreement is 1.67×10⁻¹⁵ relative. End-to-end, however, the 60-state workload changes from 1.0104 s to 1.0084 s.",
    lesson: "The kernel is only 1.9% of the transport sweep. A 4.4× local win becomes 1.002× overall—useful evidence to stop here.",
  },
};

function updateProgress() {
  const scrollable = document.documentElement.scrollHeight - innerHeight;
  const ratio = scrollable > 0 ? scrollY / scrollable : 0;
  progressBar.style.width = `${Math.min(100, Math.max(0, ratio * 100))}%`;
}

addEventListener("scroll", updateProgress, { passive: true });
addEventListener("resize", updateProgress);
updateProgress();

document.querySelectorAll("[data-jump]").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelector(button.dataset.jump)?.scrollIntoView({ behavior: "smooth" });
  });
});

const equationTabs = [...document.querySelectorAll("[data-equation]")];
const equationPanels = [...document.querySelectorAll("[data-panel]")];

function selectEquation(name) {
  equationTabs.forEach((tab) => {
    const selected = tab.dataset.equation === name;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  });
  equationPanels.forEach((panel) => {
    const selected = panel.dataset.panel === name;
    panel.hidden = !selected;
    panel.classList.toggle("active", selected);
  });
}

equationTabs.forEach((tab, index) => {
  tab.addEventListener("click", () => selectEquation(tab.dataset.equation));
  tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const direction = event.key === "ArrowRight" ? 1 : -1;
    const next = equationTabs[(index + direction + equationTabs.length) % equationTabs.length];
    next.focus();
    selectEquation(next.dataset.equation);
  });
});
selectEquation("residual");

const benchmarkFields = {
  context: document.querySelector("#bench-context"),
  title: document.querySelector("#bench-title"),
  speedup: document.querySelector("#bench-speedup"),
  oldLabel: document.querySelector("#bench-old-label"),
  oldValue: document.querySelector("#bench-old-value"),
  newLabel: document.querySelector("#bench-new-label"),
  newValue: document.querySelector("#bench-new-value"),
  note: document.querySelector("#bench-note"),
  lesson: document.querySelector("#bench-lesson"),
};
const benchmarkButtons = [...document.querySelectorAll("[data-benchmark]")];

function selectBenchmark(name) {
  const data = benchmarks[name];
  Object.entries(benchmarkFields).forEach(([key, element]) => {
    element.textContent = data[key];
  });
  document.querySelector("#bench-new-bar").style.setProperty("--value", data.newWidth);
  benchmarkButtons.forEach((button) => {
    const selected = button.dataset.benchmark === name;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-selected", String(selected));
  });
}

benchmarkButtons.forEach((button) => {
  button.addEventListener("click", () => selectBenchmark(button.dataset.benchmark));
});
selectBenchmark("equilibrium");

const temperatureSlider = document.querySelector("#temperature-slider");
const temperatureOutput = document.querySelector("#temperature-output");
const activeCount = document.querySelector("#active-count");
const branchStatus = document.querySelector("#branch-status");
const branchDetail = document.querySelector("#branch-detail");

function updateCutoffLab() {
  const temperature = Number(temperatureSlider.value);
  temperatureOutput.textContent = `${temperature.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} K`;
  if (temperature >= 20861.95 && temperature <= 20862.05) {
    activeCount.textContent = "27 ↔ 28";
    branchStatus.textContent = "Two roots";
    branchDetail.textContent = "The 28-level root has lower Gibbs objective.";
  } else if (temperature >= 20862.4 && temperature <= 20862.45) {
    activeCount.textContent = "28 ↔ 29";
    branchStatus.textContent = "Two roots";
    branchDetail.textContent = "The 29-level root has lower Gibbs objective.";
  } else if (temperature < 20861.95) {
    activeCount.textContent = "27";
    branchStatus.textContent = "Single branch";
    branchDetail.textContent = "Below the observed 27/28 coexistence window.";
  } else if (temperature < 20862.4) {
    activeCount.textContent = "28";
    branchStatus.textContent = "Single branch";
    branchDetail.textContent = "Between the two observed crossing windows.";
  } else {
    activeCount.textContent = "29";
    branchStatus.textContent = "Single branch";
    branchDetail.textContent = "Above the observed 28/29 coexistence window.";
  }
}

temperatureSlider.addEventListener("input", updateCutoffLab);
updateCutoffLab();

const timelineItems = [...document.querySelectorAll(".timeline li")];

document.querySelectorAll(".timeline-controls button").forEach((button) => {
  button.addEventListener("click", () => {
    const phase = button.dataset.phase;
    document.querySelectorAll(".timeline-controls button").forEach((item) => item.classList.toggle("active", item === button));
    timelineItems.forEach((item) => item.classList.toggle("hidden", phase !== "all" && item.dataset.phase !== phase));
  });
});

document.querySelectorAll(".validation-grid details").forEach((detail) => {
  detail.addEventListener("toggle", () => {
    if (!detail.open) return;
    document.querySelectorAll(".validation-grid details").forEach((other) => {
      if (other !== detail) other.open = false;
    });
  });
});
