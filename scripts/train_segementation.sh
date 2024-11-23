export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=ac6fadd5c937cb76a00106a28a5986a73e0cad60

# python src/train.py experiment=test_segmentation_unet trainer.devices=1
# python src/train.py experiment=segmentation_caranet trainer.devices=1
# python src/train.py experiment=segmentation_unet data.batch_size=128 trainer.devices=2
python src/train.py experiment=segmentation_caranet data.batch_size=10 trainer.devices=1
python src/train.py experiment=segmentation_unet data.batch_size=2 trainer.devices=1