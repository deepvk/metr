# you will need gdown and svn

TMPDIR=/data/varlamov_a_data/tmpdir pip3 --cache-dir /data/varlamov_a_data/cache_dir install -r /data/varlamov_a_data/dima/requirements.txt

export PATH_TO_OPEN_CLIP=/data/varlamov_a_data/miniconda3/envs/wm_env/lib/python3.10/site-packages/tree_ring_watermark/open_clip

svn export --force https://github.com/Alphonsce/tree-ring-watermark/trunk/src/tree_ring_watermark/open_clip $PATH_TO_OPEN_CLIP