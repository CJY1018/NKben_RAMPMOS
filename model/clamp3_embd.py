import os
import sys
from utils import *
import numpy as np
import shutil

def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('Usage: python clamp3_embd.py <input_dir_path> <output_dir_path> [--get_global]')
        sys.exit(1)

    input_dir_path = os.path.abspath(sys.argv[1])
    output_dir_path = os.path.abspath(sys.argv[2])  # output_dir不需要预新建

    input_dir = os.path.basename(input_dir_path)
    output_dir = os.path.basename(output_dir_path)

    # 除非运行时指定'--get_global'，否则默认不global
    # global:(1,768)
    # 不global:默认(1,T,768)
    global_flag = True if len(sys.argv) == 4 and sys.argv[3] == '--get_global' else False

    # Step 1: Create a temporary directory
    os.makedirs('temp', exist_ok=True)

    # Step 2: Determine modalities automatically
    input_modality = get_modality_from_dir(input_dir_path)

    if input_modality is None:
        print(f'Error: Could not determine input modality for "{input_dir}"')
        sys.exit(1)

    print(f'Detected input modality: {input_modality}')

    # Step 3: Extract features based on detected modality
    modality_functions = {
        'txt': extract_txt_features,
        'img': extract_img_features,
        'xml': extract_xml_features,
        'mid': extract_mid_features,
        'audio': extract_audio_features,
    }

    if os.path.exists(output_dir_path):
        print(f'Warning: {output_dir} already exists, will delete and re-extract.')
        # shutil.rmtree(output_dir)
    else:
        modality_functions[input_modality](input_dir_path, output_dir_path, global_flag)
    # 可复用的优美代码，实际上就是extract_audio_features(input_dir_path, output_dir_path, global_flag)

    # Step 4: Clean up
    remove_folder('temp')

if __name__ == '__main__':
    main()