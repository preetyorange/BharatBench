#!/bin/bash
source venv/bin/activate
echo "Training HGT_prl..."
python3 train_transformer.py --target HGT_prl
echo "Training TMP_2m..."
python3 train_transformer.py --target TMP_2m
echo "Training APCP_sfc..."
python3 train_transformer.py --target APCP_sfc
echo "All training completed!"
