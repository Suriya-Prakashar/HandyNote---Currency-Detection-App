# CurrencyDetection — HandyNote

**HandyNote** is an Android application that identifies **Indian rupee banknotes** in real time using on-device **TensorFlow Lite** inference. The app is aimed at accessibility and quick verification: it combines **CameraX** capture, **YOLO-style** object detection, **text-to-speech** feedback, and a **session history** with running totals.

Repository: [HandyNote — Currency Detection App](https://github.com/Suriya-Prakashar/HandyNote---Currency-Detection-App)

---

## What it does

- Points the phone camera (or an **ESP32-CAM** stream) at a note and **captures** a frame.
- Runs a **YOLOv5-exported** FP16 TFLite model to detect which denomination is present.
- Shows **denomination** and **confidence** on screen and can **speak** the result via **TTS**.
- Records detections in **History** (current session) and shows a **total** of scanned amounts.

---

## Supported denominations

The bundled model is trained for **6 classes** (Indian rupee note values):

| Class | Label |
|--------|--------|
| 1 | ₹10 |
| 2 | ₹20 |
| 3 | ₹50 |
| 4 | ₹100 |
| 5 | ₹200 |
| 6 | ₹500 |

Labels are defined in `app/src/main/assets/labels.txt` (loaded at runtime) and duplicated under `IndianCurrency_YOLO/labels.txt` for the training bundle; class names also appear in `data.yaml`.

---

## Dataset / training context

Training metadata is stored in `app/src/main/assets/IndianCurrency_YOLO/data.yaml`. The dataset is referenced from **Roboflow Universe** (Indian currency notes project), with paths suitable for a YOLOv5-style training pipeline. **Report figures** (curves, confusion matrix, sample detections, etc.) live under:

`app/src/main/assets/IndianCurrency_YOLO/report/`

---

## Key features

| Feature | Description |
|--------|-------------|
| **On-device inference** | TensorFlow Lite **2.16** with **select TF ops** for YOLO-style outputs. |
| **Mobile camera** | **CameraX** preview, capture, **torch** toggle, **front/back** switch. |
| **ESP32-CAM mode** | Optional stream from a device on the LAN; default base URL `http://192.168.4.1` (ESP32 hotspot style). Uses `/stream` for preview and `/capture` for a still frame. |
| **Post-processing** | YOLOv5-style output parsing (confidence **0.25**, IoU **0.45**, up to **5** detections) via `CameraUtils.yoloV5Postprocess`. Falls back to classification-style output if tensor shape differs. |
| **Accessibility** | **Text-to-speech** for the last result summary. |
| **History** | `HistoryActivity` lists session detections and supports **delete**; **total** is computed from stored amounts. On a fresh app launch, history is reset (see `MainActivity` startup). |
| **UI** | Material components, **View Binding**, portrait layout, scan overlay animation. |

---

## Tech stack

- **Language:** Kotlin (**2.2**), Java 17 bytecode target  
- **Build:** Android Gradle Plugin **9.0.1**, Gradle wrapper (see `gradle/wrapper/`)  
- **Min SDK:** 21 · **Target / compile SDK:** 34  
- **Package:** `com.currencyidentifier`  
- **App name (launcher):** `CurrencyDetection - HandyNote`  
- **Libraries:** AndroidX (Core, AppCompat, Material), **CameraX 1.3.1**, **TensorFlow Lite** + **support** + **select-tf-ops**, **Kotlin coroutines**

---

## Project layout (high level)

```
├── app/
│   ├── build.gradle          # App module, dependencies, TFLite no-compress, YOLO asset copy task
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/
│       │   ├── IndianCurrency_YOLO/   # Model, labels copy, data.yaml, report images, test images
│       │   └── labels.txt             # Primary labels file opened by the app (`LABELS_FILE`)
│       ├── java/com/currencyidentifier/
│       │   ├── MainActivity.kt        # Camera, ESP32, TFLite, TTS, history writes
│       │   ├── HistoryActivity.kt     # Session history + total
│       │   └── CameraUtils.kt         # Bitmap prep, YOLO post-process, helpers
│       └── res/                       # Layouts, themes, drawables, strings
├── build.gradle              # Root plugins (AGP, Kotlin)
├── settings.gradle           # Root project name: CurrencyDetection-handynote-v1
├── gradle.properties         # JVM/Android flags (see note below)
└── gradlew / gradlew.bat     # Gradle wrapper
```

The **TFLite model** file `best-fp16.tflite` is kept under `app/src/main/assets/IndianCurrency_YOLO/` and copied into generated assets at build time so the runtime can open it as **`best-fp16.tflite`** from assets (see `copyYoloModelToAssets` in `app/build.gradle`).

---

## Permissions

- **CAMERA** — required for capture and preview.  
- **INTERNET** — used for ESP32 HTTP stream/capture and optional remote/API-style flows.  

---

## Build & run

### Prerequisites

- **Android Studio** (recommended) with **Android SDK** for API **34**  
- **JDK 17** (AGP 9 / Kotlin toolchain expect a modern JDK)

**Note:** `gradle.properties` may contain a machine-specific `org.gradle.java.home` entry. If the project fails to sync on another PC, remove or update that line to point to your local JDK 17 installation.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Suriya-Prakashar/HandyNote---Currency-Detection-App.git
   cd HandyNote---Currency-Detection-App
   ```
2. Open the folder in Android Studio and let Gradle sync.  
3. Connect a device or start an emulator, then **Run** the `app` configuration.

Release builds use **minify disabled** by default (`minifyEnabled false` in `app/build.gradle`).

---

## Using the app

1. Grant **Camera** (and **Internet** if using ESP32).  
2. Choose **Mobile Cam** or **ESP32 Cam** in the mode selector.  
3. For ESP32, enter the device base URL (e.g. `http://192.168.4.1` on the default hotspot).  
4. Tap **capture** to take a frame and run inference.  
5. Use **speaker** to hear the result; open **History** from the UI to review amounts and total.

---

## ESP32-CAM integration (summary)

- Default base URL: `http://192.168.4.1`  
- **Stream:** `GET {base}/stream` (MJPEG-style preview in app)  
- **Still capture:** `GET {base}/capture`  

Adjust the IP/URL in the app to match your network. The ESP32 firmware must expose these endpoints consistently with what `MainActivity` expects.

---

## Version

- **versionName:** 1.4  
- **versionCode:** 5  

(See `app/build.gradle` `defaultConfig`.)

---

## Limitations & tips

- Detection quality depends on **lighting**, **focus**, and **angle**; poor conditions may yield low confidence or “No currency detected”.  
- History is **session-oriented** (implementation clears or uses shared preferences per app design—check `MainActivity` / `HistoryActivity` for current behavior).  
- The model is specific to the **training distribution**; new note designs or heavy occlusion may reduce accuracy.

---

## License / third-party data

- Dataset licensing information appears in `data.yaml` (e.g. **CC BY 4.0** where applicable).  
- Respect third-party dataset and model conversion terms when retraining or redistributing.

---

## Author

**Suriya Prakashar** — [GitHub @Suriya-Prakashar](https://github.com/Suriya-Prakashar)

---

*Last updated to match the repository layout and dependencies in this project.*
