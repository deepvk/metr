# conda install -y svn

# TMPDIR=/data/varlamov_a_data/tmpdir pip3 --cache-dir /data/varlamov_a_data/cache_dir install -r /data/varlamov_a_data/dima/requirements.txt

# export PATH_TO_OPEN_CLIP=/data/varlamov_a_data/miniconda3/envs/stable_tree/lib/python3.10/site-packages/tree_ring_watermark

# svn export --force https://github.com/Alphonsce/tree-ring-watermark/src/tree_ring_watermark/open_clip $PATH_TO_OPEN_CLIP

# export WORKING_OPEN_CLIP_FOLDER=/data/varlamov_a_data/miniconda3/envs/new_attack/lib/python3.10/site-packages/tree_ring_watermark/open_clip


SOURCE_FOLDER="/data/varlamov_a_data/miniconda3/envs/new_attack/lib/python3.10/site-packages/tree_ring_watermark/open_clip"
DESTINATION_FOLDER="/data/varlamov_a_data/miniconda3/envs/stable_tree/lib/python3.10/site-packages/tree_ring_watermark/open_clip"

cp -r "$SOURCE_FOLDER"/* "$DESTINATION_FOLDER"

echo "Files copied successfully."
