cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j12

rm /pvfsmnt/119010114/ass4/testhb

cp testhb -d /pvfsmnt/119010114/ass4