//
// Created by Xiaozhe Yao on 12.12.20.
//

#ifndef EURUSDB_THREADPOOL_H
#define EURUSDB_THREADPOOL_H

#include "../include/headers.h"

class Worker;
class Server;

class ThreadPool {
private:
    friend class Worker;
    vector<thread*> workers;
    deque<int> taskQueue;
    bool stop;
    condition_variable condition;
    mutex lockQueue;
public:
    ThreadPool(const int &size, Server* server, Store<string, string>* kvStore);
    ~ThreadPool();
    void addTask(int clientfd);
};

#endif //EURUSDB_THREADPOOL_H