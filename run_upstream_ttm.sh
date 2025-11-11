# step1:生成,默认32khz
python Upstream/run_musicgen.py
# step2:重采样为16khz(musiceval训练采样率)
python Upstream/ttm_resample.py