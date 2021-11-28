source scl_source enable devtoolset-10
CC=gcc CXX=g++ cmake ..
make -j12

rm /pvfsmnt/119010114/ass4/testCUDA

cp testCUDA -d /pvfsmnt/119010114/ass4