# accelerate launch -m tree_ring_watermark.run_tree_ring_watermark \
#     --run_name no_attack \
#     --w_channel 3 \
#     --w_pattern ring \
#     --start 0 \
#     --end 10 \
#     --num_inference_steps 5 \
#     --reference_model ViT-g-14 \
#     --reference_model_pretrain laion2b_s12b_b42k \
#     --with_tracking \
#     --freq_log 2 \
#     --image_length 128 \

# Already changed it in .bashrc, because of space issues:

# export HOME=/data/varlamov_a_data
# export TMPDIR=/data/varlamov_a_data/tmpdir

accelerate launch -m tree_ring_watermark.run_tree_ring_watermark --run_name no_attack --w_channel 3 --w_pattern ring --start 0 --end 10 --num_inference_steps 3 --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k --with_tracking --freq_log 2 --image_length 64