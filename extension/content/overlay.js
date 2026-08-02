
(() => {
    const LOW_THRESHOLD = 0.65;
    const MEDIUM_THRESHOLD = 0.8;
    const HIGH_THRESHOLD = 0.9;

    const LOW_CLASS = "model-level-low";
    const MEDIUM_CLASS = "model-level-medium";
    const HIGH_CLASS = "model-level-high";

    const ALL_CLASSES = [LOW_CLASS, MEDIUM_CLASS, HIGH_CLASS];

    function probabilityToClass(fakeProb) {
        if (fakeProb >= HIGH_THRESHOLD) return HIGH_CLASS;
        if (fakeProb >= MEDIUM_THRESHOLD) return MEDIUM_CLASS;
        if (fakeProb >= LOW_THRESHOLD) return LOW_CLASS;
        return null;
    }

    function applyOverlay(img, fakeProb) {
        const tier = probabilityToClass(fakeProb);

        img.classList.remove(...ALL_CLASSES);

        if (!tier) return;

        img.classList.add(tier);

        const confidence = Math.round(fakeProb * 100);
        img.title = `Possibly AI-generated (${confidence}% confidence)`;
    }

    document.addEventListener("AI-detection-prediction", (event) => {
        const {img, fakePrediction} = event.detail;
        if (!img) return;

        applyOverlay(img, fakePrediction);
    });
})()

