RUN_PATH="/home/centos/autoimage2/denseNet_car"
cd $RUN_PATH

scp -r ./* 192.168.13.212:~/autoimage2/denseNet_car/

scp -r ./* 192.168.12.239:~/autoimage2/denseNet_car/

scp -r ./* 192.168.12.235:~/autoimage2/denseNet_car/