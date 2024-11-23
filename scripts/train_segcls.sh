export CUDA_VISIBLE_DEVICES=3
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=ac6fadd5c937cb76a00106a28a5986a73e0cad60

python src/train.py experiment=segcls data.batch_size=4 trainer.devices=1

# cd kc_cancer
# conda activate ./env
# cd new_models
