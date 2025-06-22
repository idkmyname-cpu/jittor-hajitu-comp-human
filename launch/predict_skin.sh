python predict_skin.py \
    --predict_data_list data/test_list.txt \
    --data_root data \
    --model_name pct \
    --pretrained_model output/skin/best_model.pkl \
    --predict_output_dir predict \
    --batch_size 16
