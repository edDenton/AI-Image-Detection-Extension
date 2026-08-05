// Code primarily comes from https://jsmastery.com/blogs/infinite-scroll-with-javascript-the-step-by-step-guide
const TOTAL_IMAGES = 18
const IMAGES_PER_BATCH = 6
const EXTENSION = ".jpg"

let imgIndex = 1
function loadImages() {
    for (let i = 0; i < IMAGES_PER_BATCH; i++) {
        const img = document.createElement("img");
        img.src = `images/${imgIndex}${EXTENSION}`;
        container.appendChild(img);

        imgIndex++;
        if (imgIndex > TOTAL_IMAGES) {
            imgIndex = 1;
        }
    }
}

const container = document.querySelector(".container");

loadImages();

window.addEventListener("scroll", () => {
    if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight) {
        loadImages();
    }
});