# Llama LOL

An experiment in making a funny LLM

## Setup

Download and build whisper

```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
bash ./models/download-ggml-model.sh large
make
```

Scrape some data

```bash
python3 yt.py
```

Copy the txt files into `/data`

Clean up the data

- Seperate jokes (comedy bits) to individual lines
- Remove extra whisper annotations

## Train

For this you will need a large GPU > 20 Gbs of VRAM

```bash
python3 train.py
```

## Sample

To get new jokes run

```bash
python3 sample.py
```
