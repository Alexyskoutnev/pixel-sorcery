#!/usr/bin/env python3
"""Download real estate test images from Unsplash."""

import os
import urllib.request
import time

OUTPUT_DIR = "real_estate_test/input"

# Unsplash source URLs - direct download without API key
# Using interior/real estate related search terms
SEARCH_TERMS = [
    "living-room",
    "kitchen-interior",
    "bedroom-interior",
    "bathroom-interior",
    "modern-house",
    "apartment-interior",
    "dining-room",
    "home-office",
    "real-estate",
    "house-interior",
]

def download_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    count = 0
    target = 40

    for term in SEARCH_TERMS:
        if count >= target:
            break

        # Download 4 images per search term
        for i in range(4):
            if count >= target:
                break

            # Unsplash source API - random image from search
            # Using 1600x1200 resolution (similar to real estate photos)
            url = f"https://source.unsplash.com/1600x1200/?{term}&sig={count}"
            filename = f"{OUTPUT_DIR}/{count:04d}_{term.replace('-', '_')}.jpg"

            print(f"Downloading {count+1}/{target}: {term}...")

            try:
                urllib.request.urlretrieve(url, filename)
                # Check if file was downloaded (not a redirect page)
                if os.path.getsize(filename) > 10000:  # At least 10KB
                    count += 1
                    time.sleep(0.5)  # Be nice to the API
                else:
                    os.remove(filename)
                    print(f"  Skipped (too small)")
            except Exception as e:
                print(f"  Error: {e}")
                continue

    print(f"\nDownloaded {count} images to {OUTPUT_DIR}/")

if __name__ == "__main__":
    download_images()
