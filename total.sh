cd DANN_py3-master
python main.py

cd ..
CUDA_VISIBLE_DEVICES=0,1 python main.py --method self-supervision --source cifar10 --nepoch 35 --milestone_1 20 --milestone_2 30  --batch_size 100 --width 8 --target stl10 --flip --outf output/cifar10

cd PixDA
python main.py

cd ..
cd python-adda-master
python main.py
