# Llama LOL

An experiment in making a funny LLM

**Blog:** <https://philliphaeusler.com/posts/llama_lol/>

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

## More Data

The can scrape more data from youtube with `python yt.py`

Just add additional videos to the python script.

Download and build whisper for (higher quality) transcription

```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
bash ./models/download-ggml-model.sh large
make
```

Copy the txt files into `/data`

Clean up the data

- Separate jokes (comedy bits) to individual lines
- Remove extra whisper annotations
