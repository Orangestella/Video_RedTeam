"""
Video generation via commercial T2V model APIs.

Supports multiple commercial video generation services tested in the paper:
  - Sora (OpenAI-compatible API)
  - Veo (Google)
  - Generic API (Hailuo, Luma, etc.)

Usage:
    from video_generation.api_generator import create_video

    video_path = create_video(
        prompt="A cat playing piano",
        model_name="sora-1.0",
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
        output_dir="./output"
    )
"""

import json
import os
import time

import requests
from openai import OpenAI
from pathlib import Path
from typing import Dict, Optional


class VideoGenerator:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            print(f"Warning: OpenAI client init failed: {e}")

    def generate_video(self, config: Dict) -> str:
        if "model_name" not in config:
            raise ValueError("config must contain 'model_name'")
        if "prompt" not in config:
            raise ValueError("config must contain 'prompt'")

        model_name = config["model_name"]
        if model_name.startswith("sora"):
            return self._generate_with_openai(config)
        elif model_name.startswith("veo"):
            return self._generate_with_veo(config)
        else:
            return self._generate_with_generic(config)

    def _generate_with_openai(self, config: Dict) -> str:
        if self.client is None:
            raise RuntimeError("OpenAI client not initialized")

        response = self.client.videos.create(
            prompt=config["prompt"],
            model=config["model_name"]
        )
        video_id = response.id
        video_url = self._poll_video_status_sdk(video_id)
        return self._save_video(video_url, video_id, config)

    def _generate_with_veo(self, config: Dict) -> str:
        url = f"{self.base_url}/videos"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": config["model_name"],
            "prompt": {"text": config["prompt"]},
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if "code" in result and result["code"] != 200:
            raise RuntimeError(f"API error: {result.get('msg', 'Unknown')} (code: {result['code']})")

        video_id = (result.get("id") or result.get("video_id") or
                    result.get("task_id") or result.get("job_id"))
        if not video_id:
            raise RuntimeError(f"No video ID in response: {result}")

        video_url = self._poll_video_status_requests(video_id)
        return self._save_video(video_url, video_id, config)

    def _generate_with_generic(self, config: Dict) -> str:
        url = f"{self.base_url}/videos"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {"model": config["model_name"], "prompt": config["prompt"]}

        if "negative_prompt" in config and config["negative_prompt"]:
            payload["negative_prompt"] = config["negative_prompt"]
        if "durationSeconds" in config:
            payload["duration_seconds"] = config["durationSeconds"]
        if "aspectRatio" in config:
            payload["aspect_ratio"] = config["aspectRatio"]

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if "code" in result and result["code"] != 200:
            raise RuntimeError(f"API error: {result.get('msg', 'Unknown')}")

        video_id = (result.get("id") or result.get("video_id") or
                    result.get("task_id") or result.get("job_id"))
        if not video_id:
            raise RuntimeError(f"No video ID in response: {result}")

        video_url = self._poll_video_status_requests(video_id)
        return self._save_video(video_url, video_id, config)

    def _poll_video_status_sdk(self, video_id: str, max_wait: int = 300) -> str:
        start = time.time()
        while True:
            video = self.client.videos.retrieve(video_id)
            status = getattr(video, 'status', 'unknown')
            if status == "completed":
                video_url = getattr(video, 'url', '')
                if not video_url:
                    return self.client.videos.download_content(video_id).read()
                return video_url
            elif status == "failed":
                raise RuntimeError(f"Video generation failed: {getattr(video, 'error', 'Unknown')}")
            if time.time() - start > max_wait:
                raise TimeoutError(f"Timeout ({max_wait}s)")
            time.sleep(5)

    def _poll_video_status_requests(self, video_id: str, max_wait: int = 300) -> str:
        start = time.time()
        url = f"{self.base_url}/videos/{video_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        while True:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            result = response.json()
            status = result.get("status", "unknown")

            if status == "completed":
                video_url = result.get("url", "")
                if not video_url:
                    raise RuntimeError("No download URL")
                return video_url
            elif status in ("failed", "error"):
                raise RuntimeError(f"Video generation failed: {result.get('error', 'Unknown')}")
            if time.time() - start > max_wait:
                raise TimeoutError(f"Timeout ({max_wait}s)")
            time.sleep(5)

    def _save_video(self, video_url, video_id: str, config: Dict) -> str:
        output_dir = config.get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{video_id}_{int(time.time())}.mp4"
        save_path = os.path.join(output_dir, filename)

        if isinstance(video_url, bytes):
            with open(save_path, "wb") as f:
                f.write(video_url)
        elif video_url.startswith("http"):
            response = requests.get(video_url)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(response.content)
        else:
            with open(save_path, "wb") as f:
                with open(video_url, "rb") as src:
                    f.write(src.read())

        print(f"Video saved to: {save_path}")
        return save_path


def create_video(prompt: str, model_name: str = "sora-1.0",
                 api_key: str = None, base_url: str = "https://api.openai.com/v1",
                 output_dir: str = "./output", **kwargs) -> str:
    if api_key is None:
        api_key = os.environ.get("VIDEO_API_KEY", "")
    if not api_key:
        raise ValueError("API key required. Set VIDEO_API_KEY env var or pass api_key argument.")

    generator = VideoGenerator(api_key=api_key, base_url=base_url)
    config = {"model_name": model_name, "prompt": prompt, "output_dir": output_dir, **kwargs}
    return generator.generate_video(config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate video via commercial API")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="sora-1.0")
    parser.add_argument("--api_key", default=None, help="API key (or set VIDEO_API_KEY env)")
    parser.add_argument("--base_url", default="https://api.openai.com/v1")
    parser.add_argument("--output_dir", default="./output")
    args = parser.parse_args()

    path = create_video(
        prompt=args.prompt, model_name=args.model,
        api_key=args.api_key, base_url=args.base_url,
        output_dir=args.output_dir,
    )
    print(f"Generated: {path}")
