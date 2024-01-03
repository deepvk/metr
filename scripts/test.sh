# accelerate launch -m tree_ring_watermark.run_tree_ring_watermark_fid \
#   --run_name test_fid_run \
#   --w_channel 3 \
#   --w_pattern ring \
#   --start 0 --end 10 \
#   --with_tracking \
#   --run_no_w \
#   --prompt_file /data/varlamov_a_data/dima/fid_outputs/coco/meta_data.json \
#   --gt_folder /data/varlamov_a_data/dima/fid_outputs/coco/ground_truth \
#   --num_inference_steps 50 \
#   --freq_log 2

# accelerate launch -m tree_ring_watermark.run_tree_ring_watermark \
#   --run_name test_jpeg \
#   --w_channel 3 \
#   --w_pattern ring \
#   --jpeg_ratio 25 \
#   --start 0 --end 10 \
#   --with_tracking \
#   --reference_model ViT-g-14 \
#   --reference_model_pretrain laion2b_s12b_b42k \
#   --freq_log 2 \
#   --num_inference_steps 3

accelerate launch -m tree_ring_watermark.run_tree_ring_watermark --run_name no_attack --w_channel 3 --w_pattern ring --start 0 --end 1000 --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k