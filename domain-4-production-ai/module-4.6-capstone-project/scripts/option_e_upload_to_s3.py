#!/usr/bin/env python3
"""
Option E: S3 Upload with CORS Configuration

Uploads model files to S3 with proper CORS for browser access.

Usage:
    python option_e_upload_to_s3.py \
        --model-path ./models/matcha-onnx-int4 \
        --bucket matcha-expert-model \
        --region us-east-1
"""

import argparse
from pathlib import Path
import json
import mimetypes


def upload_to_s3(
    model_path: Path,
    bucket_name: str,
    region: str = "us-east-1",
    allowed_origins: list = None,
) -> str:
    """
    Upload model files to S3 with CORS configuration.

    Sets up proper CORS headers for browser access from
    your deployed application.

    Args:
        model_path: Path to model files
        bucket_name: S3 bucket name
        region: AWS region
        allowed_origins: List of allowed origins for CORS

    Returns:
        Base URL for the model files
    """
    import boto3
    from botocore.exceptions import ClientError

    print("=" * 70)
    print("S3 UPLOAD WITH CORS")
    print("=" * 70)

    s3 = boto3.client("s3", region_name=region)

    # Default allowed origins
    if allowed_origins is None:
        allowed_origins = [
            "http://localhost:5173",  # Vite dev
            "http://localhost:3000",  # Create React App
            "*",  # Allow all (restrict in production!)
        ]

    # Create bucket if needed
    print(f"\nBucket: {bucket_name}")
    try:
        s3.head_bucket(Bucket=bucket_name)
        print("  Bucket exists")
    except ClientError:
        print("  Creating bucket...")
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region}
            )
        print("  Created!")

    # Configure CORS
    print("\nConfiguring CORS...")
    cors_config = {
        "CORSRules": [{
            "AllowedHeaders": ["*"],
            "AllowedMethods": ["GET", "HEAD"],
            "AllowedOrigins": allowed_origins,
            "ExposeHeaders": ["Content-Length", "Content-Type", "ETag"],
            "MaxAgeSeconds": 3600,
        }]
    }

    s3.put_bucket_cors(Bucket=bucket_name, CORSConfiguration=cors_config)
    print(f"  Allowed origins: {allowed_origins}")

    # Upload files
    print(f"\nUploading files from {model_path}...")

    if not model_path.exists():
        print(f"  Error: Path not found!")
        return None

    uploaded = 0
    for file_path in model_path.iterdir():
        if file_path.is_file():
            key = file_path.name

            # Determine content type
            content_type, _ = mimetypes.guess_type(str(file_path))
            if content_type is None:
                if file_path.suffix == ".onnx":
                    content_type = "application/octet-stream"
                elif file_path.suffix == ".json":
                    content_type = "application/json"
                else:
                    content_type = "application/octet-stream"

            size = file_path.stat().st_size / 1e6
            print(f"  Uploading {key} ({size:.1f} MB)...")

            s3.upload_file(
                str(file_path),
                bucket_name,
                key,
                ExtraArgs={
                    "ACL": "public-read",
                    "ContentType": content_type,
                }
            )
            uploaded += 1

    # Get URL
    base_url = f"https://{bucket_name}.s3.{region}.amazonaws.com/"

    print(f"\nUpload complete!")
    print(f"  Files uploaded: {uploaded}")
    print(f"  Base URL: {base_url}")

    return base_url


def generate_cors_json(output_path: Path, allowed_origins: list = None) -> None:
    """Generate CORS configuration JSON for manual upload."""

    if allowed_origins is None:
        allowed_origins = [
            "https://your-domain.vercel.app",
            "http://localhost:5173",
            "http://localhost:3000",
        ]

    cors_config = [{
        "AllowedHeaders": ["*"],
        "AllowedMethods": ["GET", "HEAD"],
        "AllowedOrigins": allowed_origins,
        "ExposeHeaders": ["Content-Length", "Content-Type", "ETag"],
        "MaxAgeSeconds": 3600,
    }]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cors_config, f, indent=2)

    print(f"CORS config saved to {output_path}")
    print("\nTo apply manually:")
    print(f"  aws s3api put-bucket-cors --bucket YOUR_BUCKET --cors-configuration file://{output_path}")


def print_cli_commands(bucket_name: str, model_path: Path, region: str) -> None:
    """Print AWS CLI commands for manual upload."""

    print("\n" + "=" * 70)
    print("MANUAL UPLOAD COMMANDS")
    print("=" * 70)

    commands = f"""
# Create bucket
aws s3 mb s3://{bucket_name} --region {region}

# Apply CORS (save the cors.json first)
aws s3api put-bucket-cors --bucket {bucket_name} --cors-configuration file://cors.json

# Upload model files
aws s3 sync {model_path} s3://{bucket_name}/ --acl public-read

# Verify
aws s3 ls s3://{bucket_name}/

# Your model URL
# https://{bucket_name}.s3.{region}.amazonaws.com/
"""
    print(commands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload to S3 with CORS")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--cors-only", type=Path, help="Just generate CORS JSON")
    parser.add_argument("--cli-commands", action="store_true", help="Print CLI commands only")
    args = parser.parse_args()

    if args.cors_only:
        generate_cors_json(args.cors_only)
    elif args.cli_commands:
        print_cli_commands(args.bucket, args.model_path, args.region)
    else:
        upload_to_s3(args.model_path, args.bucket, args.region)
