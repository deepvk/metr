while getopts "r:s:" opt; do
  case $opt in
    r) r="$OPTARG"
    ;;
    s) s="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

accelerate launch -m metr.run_metr \
  --project_name metr-detect-no-att \
  --run_name s=$s-r=$r --w_channel 3 --w_pattern ring \
  --start 0 --end 1000 \
  --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k \
  --with_tracking \
  --w_radius $r \
  --msg_type binary \
  --use_random_msgs \
  --msg_scaler $s \
  --no_stable_sig