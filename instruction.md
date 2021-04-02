    # Generate annotations for training
    cd tool
    python3 coco_annotation.py \
    --json_file_path ../data/instances_train.json \
    --images_dir_path /path/to/detection_dataset/train \
    --output_path ../data/train_simple.txt
    
    python3 coco_annotation.py \
    --json_file_path ../data/instances_test.json \
    --images_dir_path /path/to/detection_dataset/test \
    --output_path ../data/test_simple.txt
    
    # Run training
    cd ../
    python3 train.py \
    --pretrained /path/to/yolov4.weights \
    --batch_size 32 \
    --subdivisions 32 \
    --learning_rate 0.000000000261 \
    --num_epochs 300 \
    --data_dir /home/luch/Programming/Python/TestTasks/detection_dataset \
    --train_label data/train_simple.txt \
    --val_label test_simple.txt
    
    