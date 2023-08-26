import subprocess

videos = {
    "mashup": "https://www.youtube.com/watch?v=HT7bIWDJ5w0",
    "netflix_is_a_joke": "https://www.youtube.com/watch?v=IwuarzMMHAg",
    "all_awards_are_stupid": "https://www.youtube.com/watch?v=ityRn2IA24A",
    "sucks": "https://www.youtube.com/watch?v=s4Df4L6lAgs",
}

for name, url in videos.items():
    print(name, url)
    subprocess.check_call(
        f"python3 -m youtube_dl -o './yt/{name}.%(ext)s' -x --audio-format=aac --audio-quality=0 --verbose --all-subs {url}",
        shell=True,
    )
    subprocess.check_call(
        f"ffmpeg -i yt/{name}.aac -ar 16000 -ac 1 -c:a pcm_s16le yt/{name}.wav",
        shell=True,
    )
    subprocess.check_call(
        f"whisper.cpp/main -m models/ggml-large.bin --output-text -of data/{name}.txt yt/{name}.wav",
        shell=True,
    )
