CUiDA_VISIBLE_DEVICES=0,1,2 python main.py --batch_size 256 --method self-supervision --width 16 --source cifar10 --target stl10 --num_batches_per_test 10000 --nepoch 200 --lr 0.01 --milestone_1 50 --milestone_2 75 --rotation --outf 'output/cifar_stl_lr0.01_256'
