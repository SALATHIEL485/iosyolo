# OBB Detector (offline web app)

1. Put `my_model.onnx` (exported from Colab) in this folder.
2. In `app.js`, set `CLASSES` to your class names, in the same order as `classes.txt`.
3. Upload every file in this folder to a GitHub repo.
4. Settings > Pages > deploy from the `main` branch, root folder.
5. Open `https://USERNAME.github.io/REPO/` in Safari while online, wait for "Model ready", then Share > Add to Home Screen.
6. Turn on Airplane Mode and open it from the Home Screen icon to confirm it works offline.

After changing any file, bump `CACHE` in `sw.js` (for example `obb-detector-v2`).
