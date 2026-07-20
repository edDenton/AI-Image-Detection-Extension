
(() => {
    const TIME_ON_SCREEN_MS = 500
    const processedImages = new WeakSet();
    const processingTimers = new WeakMap();

    let ID = 0;
    function getID(){
        return `id_${Date.now()}_${ID++}`;
    }

    const iObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach(timeImage)
        },
        {
            threshold: 1.0,
            rootMargin: "0px",
        }
    );

    function timeImage(entry) {
        const img = entry.target;

        // Image should be fully on screen and stay on screen for a set duration before
        // sending it to our model, hopefully will limit unnecessary classifications
        // and improve performance
        if (entry.isIntersecting && entry.intersectionRatio >= 1.0) {
            if (processingTimers.has(img)) return;

            const timerId = setTimeout(() => {
                processingTimers.delete(img);
                handleImage(img);
            }, TIME_ON_SCREEN_MS)

            processingTimers.set(img, timerId);
        } else {
            if (processingTimers.has(img)) {
                clearTimeout(processingTimers.get(img));
                processedImages.delete(img);
            }
        }

    }
    function handleImage(img) {
        if (processedImages.has(img)) return;

        processedImages.add(img);

        const src = img.currentSrc || img.src;
        if (!src) return;

        const id = getID();

        // TODO: Handle response by resizing/normalizing image and passing image to model
        const response = chrome.runtime.sendMessage({
            type: "FETCH_IMAGE",
            id,
            url: src
        });

        iObserver.unobserve(img);
    }

    function watchImage(img) {
        if (processedImages.has(img)) return;

        if (img.complete && img.naturalWidth > 0) {
            iObserver.observe(img);
        } else {
            img.addEventListener(
                "load",
                () => iObserver.observe(img),
                {once: true}
            );
        }
    }

    function scanForImages(node) {
        if (!(node instanceof Element)) return;

        if (node.tagName === "IMG"){
            watchImage(node)
        }

        const imgs = node.querySelectorAll("img");
        imgs.forEach(watchImage);
    }

    function scanMutations(mutation) {
        mutation.addedNodes.forEach((node) => scanForImages(node))
    }

    const mObserver = new MutationObserver((mutations) => {
        mutations.forEach(scanMutations);
    });

    scanForImages(document.body)
})