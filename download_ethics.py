#!/usr/bin/env python
"""Download ETHICS dataset from HuggingFace."""

import os
from huggingface_hub import snapshot_download

print("Downloading ETHICS dataset from HuggingFace...")
print("This may take a few minutes...")

try:
    local_dir = "data/ethics"
    os.makedirs(local_dir, exist_ok=True)

    snapshot_download(
        repo_id="hendrycks/ethics",
        repo_type="dataset",
        local_dir=local_dir,
        resume_download=True,
        max_workers=1,  # Use single thread to avoid connection issues
    )

    print(f"\n✅ Download complete! Dataset saved to: {local_dir}")

    # List downloaded files
    print("\nDownloaded files:")
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, local_dir)
            print(f"  - {rel_path}")

except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("\nYou can try manually downloading from:")
    print("https://github.com/hendrycks/ethics")
