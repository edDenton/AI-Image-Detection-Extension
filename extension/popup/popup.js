const STORAGE_KEY = "flaggedCount";

const countValueElement = document.getElementById("count-value");

function renderCount(count) {
    countValueElement.textContent = count ?? 0;
}

chrome.storage.session.get(STORAGE_KEY, (result) => {
    renderCount(result[STORAGE_KEY]);
});

chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName === "session" && changes[STORAGE_KEY]) {
        renderCount(changes[STORAGE_KEY].newValue);
    }
});
