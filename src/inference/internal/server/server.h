//
// Created by Xiaozhe Yao on 12.12.20.
//

#ifndef EURUSDB_SERVER_H
#define EURUSDB_SERVER_H

#include "../include/headers.h"

class ThreadPool;
class Printer;

class Server {
 private:
  const int max_worker = 20;
  int server_fd;
  int opt = 1;
  struct sockaddr_in address;
  int port = 7243;
  int addr_len = sizeof(address);
  ThreadPool *thread_pool;
  Printer *printer;
  thread *threadPrinter;

  struct timeval timeout;
  fd_set master_set, working_set;
  atomic<int> max_sd;
  set<int> set_clients;
  mutex lock_set, lock_deque_close_socket, lock_deque_remove_socket;
  deque<int> deque_close_socket;
  deque<int> deque_remove_socket_to_set;

 public:
  Server(const int &port);
  ~Server();
  void addTaskCloseSocketToDeque(int clientfd);
  void addTaskRemoveSocketToDeque(int clientfd);
  atomic<int> sumTimeHandleGet{0};
  atomic<int> sumTimeHandleSet{0};
  atomic<int> sumTimeHandleRemove{0};
  atomic<int> sumTimeHandleExist{0};

 private:
  void eventLoop();
  int acceptClient() const;
  void listenToClient();
  int getSocket(int i);
  int getMaxIndex() const;
  void removeClientToSet(int clientfd);
  void handleCloseSocket();
  void handleRemoveClientToSet();
};
#endif  // EURUSDB_SERVER_H
