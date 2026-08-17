CUDA_VISIBLE_DEVICES=0 uv run python vmax/scripts/evaluate/evaluate.py \
    --waymo_dataset=true \
    --path_dataset=/data/Motion_Planning_and_Prediction/trainset_91f/trainset_91f.tfrecord@85126 \
    --sdc_actor=ai \
    --path_model=bc_sac_tutorial \
    --batch_size=64