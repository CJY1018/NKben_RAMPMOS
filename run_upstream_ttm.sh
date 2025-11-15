WAV_DIR="InputData/ttm_regenerate/wavs"
# step1:生成,默认32khz
python Upstream/run_musicgen.py --out_dir "$WAV_DIR"
# step2:重采样为16khz(musiceval训练采样率)
python Upstream/ttm_resample.py --input_dir "$WAV_DIR" --output_dir "$WAV_DIR" --sr 16000  