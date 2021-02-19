cmake_minimum_required(VERSION 3.17)
project(EurusDB)

set(CMAKE_CXX_STANDARD 20)

add_executable(eurus main.cpp src/indices/trees/b_tree.cpp src/indices/trees/b_tree/btree.h include/item.cpp include/item.h include/store.cpp include/store.h src/server/server.cpp src/server/server.h include/headers.h)