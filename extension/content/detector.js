
(() => {
    const TIME_ON_SCREEN_MS = 500;
    const MODEL_IMAGE_SIZE = 224;

    const IMAGE_NET_MEAN = [0.485, 0.456, 0.406];
    const IMAGE_NET_STD = [0.229, 0.224, 0.225];

    const processedImages = new WeakSet();
    const processingTimers = new WeakMap();

    ort.env.wasm.wasmPaths = chrome.runtime.getURL("lib/ort/");

    let modelSession = null;
    function getSession(){
        if (!modelSession) {
            const modelURL = chrome.runtime.getURL("model/AI_image_classifier_model.onnx")
            modelSession = ort.InferenceSession.create(modelURL);
        }
        return modelSession;
    }

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

        // Sends message to service_worker.js to fetch the image
        chrome.runtime.sendMessage({
            type: "FETCH_IMAGE",
            id,
            url: src
        },
            (response) => {
                if (chrome.runtime.lastError) {
                    console.warn("sendMessage failed for ", id, chrome.runtime.lastError.message);
                    return;
                }
                if (!response) {
                    console.warn("No response for ", id);
                    return;
                }
                if (response.type === "FETCH_IMAGE_ERROR") {
                    console.warn("Failed to fetch image: ", response.error);
                    return;
                }
                if (response.type === "FETCH_IMAGE_SUCCESS") {
                    handleImageBytes(response, img);
                }
            }
        );

        iObserver.unobserve(img);
    }

    async function handleImageBytes(resp, img) {

        let bitmap;
        try {
            const blob = new Blob([resp.data], {type: resp.mimeType});
            bitmap = await createImageBitmap(blob);
        } catch (err) {
            console.warn("Failed to turn image bytes into a bitmap: ", err);
            return;
        }

        try {
            const tensor = processToTensor(bitmap);
            const fakePrediction = await modelPredict(tensor);

            img.dispatchEvent(new CustomEvent("AI-detection-prediction", {
                bubbles: true,
                detail: {
                    img,
                    fakePrediction: fakePrediction,
                }
            }));

        } catch (err) {
            console.warn("Failed to classify image: ", err);
        } finally {
            bitmap.close();
        }
    }

    function processToTensor(bitmap) {
        const canvas = new OffscreenCanvas(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE);
        const context = canvas.getContext("2d");

        context.drawImage(bitmap, 0, 0, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE);

        const { data } = context.getImageData(0, 0, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE);
        const pixelCount = MODEL_IMAGE_SIZE * MODEL_IMAGE_SIZE;

        const tensor = new Float32Array(3 * pixelCount);

        for (let i = 0; i < pixelCount; i++) {
            const red = data[i * 4] / 255;
            const green = data[i * 4 + 1] / 255;
            const blue = data[i * 4 + 2] / 255;

            tensor[i] = (red - IMAGE_NET_MEAN[0]) / IMAGE_NET_STD[0];
            tensor[pixelCount + i] = (green - IMAGE_NET_MEAN[1]) / IMAGE_NET_STD[1];
            tensor[2 * pixelCount + i] = (blue - IMAGE_NET_MEAN[2]) / IMAGE_NET_STD[2];
        }

        return new ort.Tensor("float32", tensor, [1, 3, MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE]);
    }

    async function modelPredict(tensor) {
        const session = await getSession();

        const inputName = session.inputNames[0];
        const outputName = session.outputNames[0];

        const results = await session.run({[inputName]: tensor});
        return results[outputName].data[0]; // results gives [fake, real] but we only care about fake probability
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

    mObserver.observe(document.body, {
        childList: true,
        subtree: true
    });

    scanForImages(document.body);
})()