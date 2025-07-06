# Lab 3: **Running & Quantizing Llama Models on Android**

This lab provides a hands-on walkthrough of how to run and optimize compact large language models (LLMs) directly on Android devices using the `llama.cpp` framework. You'll explore the full workflow, including downloading, converting, quantizing, deploying, and benchmarking LLaMA-style models. By the end, you'll have a working offline mobile LLM capable of running locally without server dependencies in an on-device android application.

###  Learning Goals

- Understand the toolchain for working with `llama.cpp`
- Learn how model quantization reduces memory and compute requirements
- Modify an Android app to load and run local quantized models on device
- Benchmark and compare the runtime performance of different quantized formats on an android device

> ⚙️ Why it matters: Running LLMs natively on mobile offers benefits like reduced latency, improved privacy, and offline capability—critical for edge AI applications.

---

## Prerequisites 

| Component           | Min version            | Why you need it                                       | Install hint                                                                 |
| ------------------- | ---------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Python**          | 3.9                    | for conversion and quantization scripts               | [https://python.org](https://python.org)                                     |
| **Git**             | any                    | to clone repositories                                 | `sudo apt install git`                                                       |
| **CMake**           | 3.16                   | to build `llama.cpp` tools                            | `brew install cmake` / `apt`                                                 |
| **huggingface-cli** | latest                 | to download models (auth required)                    | `pip install --upgrade huggingface_hub`                                      |
| **Android Studio**  | Hedgehog or newer      | includes **NDK** & **ADB** for building and deploying | [https://developer.android.com/studio](https://developer.android.com/studio) |
| **Android device**  | Android 10+, ≥ 6GB RAM | target runtime                                        | any modern phone                                                             |

>  **Windows users:** use **WSL 2** with Ubuntu 22.04 for compatibility with build tools.

---

## Step 1: Install Android Studio & ADB

Android Studio includes the tools needed to build and debug Android apps:

1. Download and install Android Studio **on your computer**.
2. During setup, select components: **SDK**, **SDK Platform Tools**, and **NDK**.
3. In settings, verify:
   - SDK path: `~/Android/Sdk/`
   - NDK ≥ version 25
   - Platform-tools ≥ version 34
4. Add `~/Android/Sdk/platform-tools` to your shell `PATH`.

Check ADB installation **on your computer**:

```bash
adb version
```

---

## Step 2: Authenticate with Hugging Face

We use Hugging Face to download a pretrained LLaMA-style model.

**On your computer**, run:

```bash
pip install --upgrade huggingface_hub
huggingface-cli login
```

1. Log in at [https://huggingface.co](https://huggingface.co).
2. Go to **Settings → Access Tokens**.
3. Create a token with **Read** access.
4. Paste the token in your terminal.

---

## Step 3: Clone & Build `llama.cpp`

This builds the tools used to convert and quantize models:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TOOLS=ON
make
```

> 🧪 **Why:** This creates binaries like `llama-quantize` and `llama-run`, which are used to transform and test models.

---

## Step 4: Set Up a Workspace for Models

**On your computer**, run:

```bash
cd ..
mkdir llama-models && cd llama-models
```

>  Organizing models in one place keeps your workflow tidy.

---

## Step 5: Download a Pretrained 3B Llama model

**On your computer**, run:

```bash
huggingface-cli download unsloth/Llama-3.2-3B-Instruct --repo-type model --local-dir ./Llama-3.2-3B-Instruct
```

>  **Tip:** Use a 1B model variant for devices with less RAM and storage.

---

## Step 6: Convert Huggingface model to GGUF Format

**On your computer**, run:

```bash
cd ../
python convert_hf_to_gguf.py llama-models/Llama-3.2-3B-Instruct --outfile llama-models/Llama-3.2-3B-Instruct-gguf
```

>  **Why GGUF?** It's the native `llama.cpp` format that enables fast, memory-mapped loading and is portable across platforms.

---

## Step 7: Quantize the Model

Quantization is a technique that reduces the precision of the model's weights from 32-bit floating point numbers to lower bit representations (like 8-bit, 4-bit or even 2-bit integers). This significantly reduces model size and speeds up inference, with minimal impact on accuracy when done carefully.

For example, a 3B parameter model normally requires ~12GB in FP32. After quantization:
- Q8_0 (8-bit) reduces it to ~3GB
- Q4_K_M (4-bit) reduces it to ~1.5GB  
- TQ2_0 (2-bit) reduces it to ~750MB

To Quantize, we can use the llama.cpp tools. **On your computer**, run:

```bash
./build/bin/llama-quantize llama-models/Llama-3.2-3B-Instruct-gguf llama-models/Llama-3.2-3B-Instruct-gguf-Q8_0 Q8_0
./build/bin/llama-quantize llama-models/Llama-3.2-3B-Instruct-gguf llama-models/Llama-3.2-3B-Instruct-gguf-Q4_K_M Q4_K_M
./build/bin/llama-quantize llama-models/Llama-3.2-3B-Instruct-gguf llama-models/Llama-3.2-3B-Instruct-gguf-TQ2_0 TQ2_0
```

| Format   | Bitwidth | Trade-off                 |
| -------- | -------- | ------------------------- |
| Q8_0    | 8-bit    | High accuracy, large size |
| Q4_K_M | 4-bit    | Balance of sizespeed     |
| TQ2_0   | ~2-bit  | Tiny, fast, less accurate |

>  **Why Quantize?** Reduces memory and compute cost, enabling real-time use on mobile.

---

## Step 8: Test Inference Locally

**On your computer**, run:

```bash
./build/bin/llama-run llama-models/Llama-3.2-3B-Instruct-gguf-Q8_0 "What is quantization in the context of machine learning?"
```

>  Sanity check the model before deploying to Android. The command should output a response from the model to your terminal.

---

## Step 9: Open the Android App Project

**In Android Studio on your computer**:

- Open `llama.cpp/examples/llama.android`
- Wait for Gradle sync
- Switch to "Project" view for easier navigation. You can do this by selecting 'Android' in the top left of your screen, then selecting 'project' from the dropdown. 

---

## Step 10: Enable Debugging mode and modify the app to use local models

First to enable debugging mode on your phone, do the following on your **mobile**:

1. Enable USB debugging in Developer Options of you mobile:
   - Navigate to **Settings** > **About phone**
   - Tap **Build number** 7 times to enable Developer options 
   - Return to **Settings** > **Developer options**
   - Toggle on **USB debugging**

**Then In Android Studio on your computer**, edit the following files:

**Edit** `llama.androic/app/src/java/com.example.llama/MainActivity.kt`
Replace the `models = listOf(...)` section with:

```kotlin
val models = listOf(
    Downloadable("Llama-3.2-3B-q8", Uri.EMPTY, File(extFilesDir, "Llama-3.2-3B-Instruct-gguf-Q8_0")),
    Downloadable("Llama-3.2-3B-q4", Uri.EMPTY, File(extFilesDir, "Llama-3.2-3B-Instruct-gguf-Q4_K_M")),
    Downloadable("Llama-3.2-3B-q2", Uri.EMPTY, File(extFilesDir, "Llama-3.2-3B-Instruct-gguf-TQ2_0")),
)
```

**Edit** `Downloadable.kt`
Replace the entire `@Composable fun Button(...)` implementation with:

```kotlin
@JvmStatic
@Composable
fun Button(viewModel: MainViewModel, dm: DownloadManager, item: Downloadable) {
    val fileExists = item.destination.exists()

    fun onClick() {
        if (fileExists) {
            viewModel.load(item.destination.path)
        }
    }

    Button(onClick = { onClick() }, enabled = fileExists) {
        Text(if (fileExists) "Load ${item.name}" else "${item.name} - Not found")
    }
}
```

Press the Play button <img src="assets/lab3/play.png" alt="LLaMA Android App UI" width="30"/> in android studio to push the application to the device, and initialize the activation space. We haven't pushed the models yet so you won't be able to load them. We can fix that in the next step. 

---

## Step 11: Push Quantized Models to Android Device

1. Connect your device via USB cable and authorize your computer when prompted

2. Push the model files to device:

```bash
adb push llama-models/*gguf* /sdcard/Android/data/com.example.llama/files/
```

Verify the files have been transferred with:

```bash
adb shell ls /sdcard/Android/data/com.example.llama/files/
```

It should output something like this:

```bash
Llama-3.2-3B-Instruct-gguf-Q4_K_M
Llama-3.2-3B-Instruct-gguf-Q8_0
Llama-3.2-3B-Instruct-gguf-TQ2_0
```

---

## Step 12: Build and Run the App

**In Android Studio on your computer**, press the **Run** button <img src="assets/lab3/play.png" alt="LLaMA Android App UI" width="30"/>. The app will install on your device and open automatically.

---

## Step 13: Use the App 🌟

With the app installed and your models loaded onto the device, it's time to interact with them **on your Android phone**.

###  User Interface Overview

When the app opens **on your phone**, you'll see a simple interface with buttons to select your model, a text input field, and action buttons:

<img src="assets/lab3/llamaapp_im1.jpeg" alt="LLaMA Android App UI" width="300"/>

- **Load**: Select one of the quantized models you pushed earlier.
- **Text Field**: Type your prompt or question.
- **Send**: Run inference on-device and receive a generated response.

This setup lets you evaluate both the usability and responsiveness of different quantized versions of the same model.

>  **Note:** Only one model can be loaded per app session. To switch models, fully close the app and rerun it via Android Studio.

---

###  Benchmarking Model Performance

To evaluate how each quantized model performs, tap the **Bench** button after loading a model. This will execute a benchmark routine that reports:

- **Prompt processing speed (pp)** – tokens per second during initial input
- **Token generation speed (tg)** – tokens per second during autoregressive generation

Repeat this for each model version (Q8, Q4, Q2) to compare their efficiency.

Your output should resemble:

<img src="assets/lab3/llamaapp_q8.jpeg" alt="Benchmark Results for Q8 Model" width="300"/>

And you can summarize your findings in a table:

| Quant | Prompt Processing (pp) | Token Generation (tg) |
|--------|------------------------|------------------------|
| Q8     | 11.0 tokens/s          | 6.8 tokens/s           |
| Q4     | 12.0 tokens/s          | 7.1 tokens/s           |
| Q2     | 9.7 tokens/s           | 10.0 tokens/s          |

> 💡 **Insight:** Lower-bit models often trade slight accuracy degradation for faster runtime and smaller memory footprint. Particularly in the token generation phases as this is a memory bound process. 

This step gives you both a subjective impression of quality and a quantitative measure of model efficiency on-device.


---

##  Final Recap

- Built and tested `llama.cpp` locally
- Converted and quantized a LLaMA model to GGUF
- Integrated quantized models into an Android app
- Deployed and benchmarked on-device

> 🌟 You're now equipped to deploy and iterate on LLMs for edge/mobile inference!

