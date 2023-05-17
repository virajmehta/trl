while getopts g:p:e:h:s:d: flag
do
    case "${flag}" in
        g) gpu=${OPTARG};;
        p) port=${OPTARG};;
        e) ensemble=${OPTARG};;
        h) heads=${OPTARG};;
        s) head_size=${OPTARG};;
        d) ensemble_dropout=${OPTARG};;
    esac
done

LOCAL_RANK=0,1 CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port $port reward_summarization.py --bf16 --ensemble $ensemble --n_ensembles $heads --num_labels $head_size --ensemble_dropout $ensemble_dropout
LOCAL_RANK=0,1 CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --use-env --nproc_per_node=1 --master_port $port uncertainty_tests.py --bf16 --ensemble $ensemble --n_ensembles $heads --num_labels $head_size --ensemble_dropout $ensemble_dropout