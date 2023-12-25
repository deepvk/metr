accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_fid \
  --run_name fid_with_logs \
  --w_channel 3 \
  --w_pattern ring \
  --start 0 --end 5000 \
  --with_tracking \
  --run_no_w \
  --prompt_file /data/varlamov_a_data/dima/fid_outputs/coco/meta_data.json \
  --gt_folder /data/varlamov_a_data/dima/fid_outputs/coco/ground_truth