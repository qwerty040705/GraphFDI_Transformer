# /Users/dnbn/code/transformer_fault_diagnosis/compress_to_10s.py
# -*- coding: utf-8 -*-
"""
기존 mp4 영상을 정확히 10초로 압축/늘리기
 - ffmpeg 필요 (brew install ffmpeg)
"""

import subprocess
import argparse
import os
import json

def get_duration(input_file):
    """ffprobe로 영상 길이(초)를 얻음"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", input_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="입력 mp4 파일 경로")
    parser.add_argument("--output", required=True, help="저장할 mp4 파일 경로")
    parser.add_argument("--target_sec", type=float, default=10.0, help="목표 영상 길이 (초)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {args.input}")

    duration = get_duration(args.input)
    speed_factor = duration / args.target_sec   # 원본길이 / 목표길이

    # setpts는 시간 스케일을 조정: PTS=PTS/속도배율
    cmd = [
        "ffmpeg", "-y",
        "-i", args.input,
        "-vf", f"setpts=PTS/{speed_factor}",
        "-an",  # 오디오 제거
        args.output
    ]

    print(f"원본 길이: {duration:.2f}s → 목표: {args.target_sec:.2f}s | 배속: {speed_factor:.2f}x")
    print("🚀 실행:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"✅ 변환 완료: {args.output}")

if __name__ == "__main__":
    main()


"""
python3 compress_to_10s.py \
  --input lasdra_link2.mp4 \
  --output lasdra_link2_compressed.mp4 \
  --target_sec 10
"""