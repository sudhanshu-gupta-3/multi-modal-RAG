#!/usr/bin/env python3
import argparse
from core.ingest.extract_images import extract_images_and_update

def main(pdf, chunks):
    print("Running ingestion for:", pdf)
    res = extract_images_and_update(pdf, chunks)
    print("Result:", res)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True)
    p.add_argument("--chunks", required=True)
    args = p.parse_args()
    main(args.pdf, args.chunks)
