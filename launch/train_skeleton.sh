python train_skeleton.py \
    --train_data_list data/train_list.txt \
    --val_data_list data/val_list.txt \
    --data_root data \
    --model_name pct \
    --output_dir output/skeleton \
    --batch_size 32 \
    --epochs 500 \
    --learning_rate 0.0001
