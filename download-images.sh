#!/bin/bash
# Download images from S3 bucket

S3_BUCKET="s3://hackathon-dec12/autohdr-real-estate-577/"
LOCAL_DIR="$(dirname "$0")/autohdr-real-estate-577/"

echo "Syncing images from $S3_BUCKET to $LOCAL_DIR"
aws s3 sync "$S3_BUCKET" "$LOCAL_DIR"
echo "Done!"
