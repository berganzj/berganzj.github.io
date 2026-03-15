#Data gather plan

### Path of Least Resistance: OBS → Training-Ready Data

---

### Step 1: OBS Setup (one-time, 2 minutes)

**Settings → Output → Recording:**
- **Recording Format:** MP4
- **Encoder:** NVIDIA NVENC H.264 (uses your RTX 4060, fast)
- **Recording Quality:** High Quality
- Leave everything else default

**Settings → Video:**
- **Output Resolution:** 1280×720 (smaller = faster processing, plenty for training)
- **FPS:** 30

That's it for OBS settings.

---

### Step 2: Recording (the filming session)

**The lazy efficient approach — record one long video per action:**

1. Hit Record
2. Perform **punch** 30–50 times with short pauses between each
3. Stop recording → `punch_session.mp4`
4. Repeat for kick, throw, jump

**Total: 4 MP4 files, ~2–5 minutes each.**

You don't need to trim individual clips — the script below handles it.

---

### Step 3: One Script Does Everything

Put your MP4s in a folder and run this. It extracts frames, organizes them into the exact folder structure fastai expects, and splits train/val automatically.

```python
import cv2
import os
import random
import shutil

# === CONFIGURE THESE ===
VIDEOS = {
    "punch": "recordings/punch_session.mp4",
    "kick":  "recordings/kick_session.mp4",
    "throw": "recordings/throw_session.mp4",
    "jump":  "recordings/jump_session.mp4",
}
OUTPUT_DIR = "dataset"
FRAME_INTERVAL = 10      # Extract every 10th frame (skip redundant ones)
TARGET_SIZE = (224, 224)  # Resize for model input
VAL_SPLIT = 0.2          # 20% goes to validation
# =======================

def extract_and_organize():
    # Clean start
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for action, video_path in VIDEOS.items():
        print(f"Processing: {action} from {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"  ERROR: Cannot open {video_path}")
            continue

        frames = []
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % FRAME_INTERVAL == 0:
                # Resize
                frame = cv2.resize(frame, TARGET_SIZE)
                # Skip dark frames (loading screens, pauses)
                if cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() > 20:
                    frames.append(frame)
            count += 1

        cap.release()
        print(f"  Extracted {len(frames)} usable frames")

        # Shuffle and split into train/val
        random.shuffle(frames)
        split_idx = int(len(frames) * (1 - VAL_SPLIT))
        splits = {
            "train": frames[:split_idx],
            "val":   frames[split_idx:]
        }

        for split_name, split_frames in splits.items():
            out_dir = os.path.join(OUTPUT_DIR, split_name, action)
            os.makedirs(out_dir, exist_ok=True)
            for i, frame in enumerate(split_frames):
                path = os.path.join(out_dir, f"{action}_{i:04d}.png")
                cv2.imwrite(path, frame)
            print(f"  {split_name}: {len(split_frames)} frames → {out_dir}")

    print("\nDone! Dataset ready at:", OUTPUT_DIR)

if __name__ == "__main__":
    extract_and_organize()
```

---

### What this produces

```
dataset/
├── train/
│   ├── punch/    (~ 80% of frames)
│   │   ├── punch_0000.png
│   │   ├── punch_0001.png
│   │   └── ...
│   ├── kick/
│   ├── throw/
│   └── jump/
└── val/          (~ 20% of frames)
    ├── punch/
    ├── kick/
    ├── throw/
    └── jump/
```

This is exactly what `fastai.vision` expects. Plug straight in:

```python
from fastai.vision.all import *

path = Path("dataset")
dls = ImageDataLoaders.from_folder(path, train="train", valid="val",
                                    item_tfms=Resize(224), bs=32)
dls.show_batch()

learn = vision_learner(dls, resnet18, metrics=accuracy)
learn.fine_tune(5)
```

---

### Summary: Total effort

| Step | Time | What you do |
|---|---|---|
| OBS settings | 2 min | Set MP4, 720p, 30fps, NVENC |
| Record 4 videos | 10–20 min | One session per action |
| Run the script | 1 min | Automatic extraction + organization |
| Train in fastai | 3 lines of code | `vision_learner` + `fine_tune` |

**4 files in, trained model out.** No manual frame cutting, no manual labeling, no folder sorting by hand.

