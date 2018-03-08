#!/bin/sh
sudo apt-get update;
sudo apt-get upgrade -y;
sudo apt-get install -y cmake ffmpeg pkg-config qtbase5-dev libqt5opengl5-dev libassimp-dev libpython3.5-dev libboost-python-dev libtinyxml-dev;
git clone https://github.com/openai/roboschool.git ~/roboschool;
cd ~/roboschool;
ROBOSCHOOL_PATH=$(pwd);
cd ..;
git clone https://github.com/olegklimov/bullet3 -b roboschool_self_collision;
mkdir bullet3/build;
cd bullet3/build;
cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 -DCMAKE_INSTALL_PREFIX:PATH=$ROBOSCHOOL_PATH/roboschool/cpp-household/bullet_local_install -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF -DBUILD_OPENGL3_DEMOS=OFF ..;
make -j4;
make install;
cd ../..;
sudo apt-get install -y python3-pip;
pip3 install --upgrade pip;
sudo -H pip3 install pyopengl;
pip3 install -e $ROBOSCHOOL_PATH;

