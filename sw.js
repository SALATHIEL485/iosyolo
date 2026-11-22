// Bump this version whenever you change any file, so devices pick up the update.
const CACHE = "obb-detector-v2";
const FILES = [
  "./",
  "./index.html",
  "./app.js",
  "./ort.min.js",
  "./ort-wasm-simd.wasm",
  "./ort-wasm.wasm",
  "./my_model.onnx",
  "./manifest.json",
  "./icon-192.png",
  "./icon-512.png"
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) =>
      Promise.all(FILES.map((f) => c.add(f).catch(() => console.warn("Not cached:", f))))
    )
  );
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request))
  );
});
