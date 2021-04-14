python main.py jester RGB \
     --arch mobilenetv2 --num_segments 8 \
     --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 \
     --batch-size 8 -j 8 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres \
     --tune_from=online_demo/mobilenetv2_jester_online.pth.tar