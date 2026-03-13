#!/bin/bash
cd /home/smorgan
python train_design_classifier.py --mode transfer --model roberta-base --pretrained_model ./pretrained_model/roberta_conservative_0309_0738 --tawos_path ./manually_labelled_data/manually_labelled_data --manually_labelled_dir ./manually_labelled_data/manually_labelled_data --max_n_labels 99999 --confidence_threshold 0.85 --output_dir ./output/roberta_transfer_improved_0310 --batch_size 16 --max_length 512 --no_preprocess 2>&1 | tee training_transfer.log
echo "Training complete. Stopping VM..."
sudo shutdown -h now
