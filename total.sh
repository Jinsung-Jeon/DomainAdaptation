CUDA_VISIBLE_DEVICES=1,2 python main.py --method self-supervision --width 16 --source cifar10 --target stl10 --num_batches_per_test 10000 --nepoch 100 --milestone_1 50 --milestone_2 75 --rotation --outf 'output/cifar_stl_r_2'