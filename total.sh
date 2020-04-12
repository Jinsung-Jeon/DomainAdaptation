#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 256 --method self-supervision --width 16 --source cifar10 --target stl10 --num_batches_per_test 10000 --nepoch 200 --lr 0.01 --milestone_1 50 --milestone_2 75 --rotation --outf 'output/cifar_stl_AD'
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 256 --method self-supervision --width 16 --source cifar10 --target stl10 --num_batches_per_test 10000 --nepoch 200 --lr 0.01 --milestone_1 50 --milestone_2 75 --flip --outf 'output/mnist_svhn_AD'

# soft label knowledge distillation
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 256 --method self-supervision --width 16 --source cifar10 --target stl10 --num_batches_per_test 10000 --nepoch 100 --lr 0.01 --milestone_1 50 --milestone_2 75 --outf 'output/cs_kd'
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 256 --method self-supervision --width 16 --source mnist --target svhn_extra --num_batches_per_test 10000 --nepoch 100 --lr 0.01 --milestone_1 50 --milestone_2 75 --outf 'output/ms_kd'

# soft label knowledge distillation and Self-supervision(source & target)
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 256 --method self-supervision --width 16 --source cifar10 --target stl10 --num_batches_per_test 10000 --nepoch 100 --lr 0.01 --milestone_1 50 --milestone_2 75 --flip --outf 'output/cs_kd_ss'
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 256 --method self-supervision --width 16 --source mnist --target svhn_extra --num_batches_per_test 10000 --nepoch 100 --lr 0.01 --milestone_1 50 --milestone_2 75 --flip --outf 'output/ms_kd_ss'

# soft label knowledge distillation and Self-supervision(target)
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 256 --method self-supervision --width 16 --source cifar10 --target stl10 --num_batches_per_test 10000 --nepoch 100 --lr 0.01 --milestone_1 50 --milestone_2 75 --flip --outf 'output/cs_kd_ss_T'
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 256 --method self-supervision --width 16 --source mnist --target svhn_extra --num_batches_per_test 10000 --nepoch 100 --lr 0.01 --milestone_1 50 --milestone_2 75 --flip --outf 'output/ms_kd_ss_T'

# soft label knowledge distillation and pseudo_label
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 256 --method self-supervision --width 16 --source cifar10 --target stl10 --num_batches_per_test 10000 --nepoch 100 --lr 0.01 --milestone_1 50 --milestone_2 75 --outf 'output/cs_kd_label'
#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 256 --method self-supervision --width 16 --source mnist --target svhn_extra --num_batches_per_test 10000 --nepoch 100 --lr 0.01 --milestone_1 50 --milestone_2 75 --outf 'output/ms_kd_label'

# self-supervised and pseudo_label

#CUDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 256 --method self-supervision --width 16 --source cifar10 --target stl10 --num_batches_per_test 10000 --nepoch 100 --lr 0.1 --milestone_1 50 --milestone_2 75 --rotation --outf 'output/cs_SS_label'
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 256 --method self-supervision_Adapt --width 16 --source mnist --target svhn --num_batches_per_test 10000 --nepoch 100 --lr 0.1 --milestone_1 50 --milestone_2 75 --rotation --outf 'output/ms_SS_label'