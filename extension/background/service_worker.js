

async function handleImageFetch(message, sendResponse) {
    const {id, url} = message;

    try {
        const response = await fetch(url);

        if (!response.ok) {
            sendResponse({
                type: "FETCH_IMAGE_ERROR",
                id,
                error: `Fetch failed with status ${response.status}`,
            });
            return;
        }

        const blob = await response.blob();
        const arrayBuff = await blob.arrayBuffer();

        sendResponse({
            type: "FETCH_IMAGE_SUCCESS",
            id,
            data: arrayBuff,
            mimeType: blob.type,
        });

    } catch (err) {
        sendResponse({
            type: "FETCH_IMAGE_ERROR",
            id,
            error: err.message,
        });
    }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type !== "FETCH_IMAGE") {
        return;
    }

    handleImageFetch(message, sendResponse);

    return true;
});

chrome.runtime.onMessage.addListener((message) => {
    if (message.type === "IMAGE_FLAGGED") {
        chrome.storage.session.get("flaggedCount", (result) => {
            const current = result.flaggedCount ?? 0;
            chrome.storage.session.set({ flaggedCount: current + 1 });
        });
    }
});