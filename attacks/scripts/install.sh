# you will need gdown and svn

# git clone https://github.com/Alphonsce/WatermarkAttacker.git /data/varlamov_a_data/dima/attacks/WatermarkAttacker

export PATH_TO_OPEN_CLIP=/data/varlamov_a_data/miniconda3/envs/new_attack/lib/python3.10/site-packages/tree_ring_watermark/open_clip

svn export --force https://github.com/Alphonsce/tree-ring-watermark/trunk/src/tree_ring_watermark/open_clip $PATH_TO_OPEN_CLIP