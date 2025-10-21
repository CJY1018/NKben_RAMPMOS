"""
将 prompt_info.txt按行拆分为单个 txt 文件，供后续文本 embedding 提取使用

示例：
python prepare_embed.py \
    --input /path/to/demo_prompt_info.txt \
    --output /path/to/text_files
"""

import os
import argparse
import logging
from typing import List


def detect_delimiter(line: str) -> str:
    """简单检测分隔符：优先制表符，其次逗号，否则返回 None。"""
    if "\t" in line:
        return "\t"
    if "," in line:
        return ","
    return None


def prepare_text_files(input_file: str, output_dir: str, skip_header: bool = True) -> int:
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    if not lines:
        logging.warning("Input file is empty: %s", input_file)
        return 0

    header = None
    if skip_header:
        header = lines[0]
        data_lines = lines[1:]
    else:
        data_lines = lines

    # 尝试从第一数据行检测分隔符
    delimiter = None
    for ln in data_lines:
        if ln.strip():
            delimiter = detect_delimiter(ln)
            break

    if delimiter is None:
        logging.warning("Could not detect delimiter; defaulting to tab. First non-empty line: %s", data_lines[0] if data_lines else "<empty>")
        delimiter = "\t"

    processed = 0
    for line in data_lines:
        if not line.strip():
            continue
        parts: List[str] = line.split(delimiter)
        if len(parts) < 2:
            logging.debug("Skipping invalid line (not enough columns): %s", line)
            continue
        text_id = parts[0].strip()
        text_content = parts[1].strip()
        if not text_id:
            logging.debug("Skipping line with empty id: %s", line)
            continue

        output_file = os.path.join(output_dir, f"{text_id}.txt")
        with open(output_file, "w", encoding="utf-8") as out_f:
            out_f.write(text_content)
        processed += 1

    logging.info("Processed %d lines. Output files saved to %s", processed, output_dir)
    return processed


def main():
    parser = argparse.ArgumentParser(description="Prepare per-prompt text files for CLAMP embedding extraction.")
    parser.add_argument("--input", "-i", required=True, help="Input prompt info file (e.g. demo_prompt_info.txt)")
    parser.add_argument("--output", "-o", required=True, help="Output directory to write per-prompt .txt files")
    parser.add_argument("--no-header", action="store_true", help="Set if the input file has no header line to skip")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    processed = prepare_text_files(args.input, args.output, skip_header=(not args.no_header))
    print(f"Processed {processed} lines. Output files saved to {args.output}")


if __name__ == "__main__":
    main()