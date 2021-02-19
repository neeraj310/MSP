//
// Created by Xiaozhe Yao on 12.12.20.
//

#include "server.h"

Server::Server(const int &port) {
  int flag, fd;
  if (port > 1024) {
    this->port = port;
  }
  // create server instance
  if ((this->server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    perror("socket failed");
    exit(FATAL_FAILURE);
  }
  if (setsockopt(this->server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                 sizeof(opt) < 0)) {
    perror("setsocketopt failed");
    exit(FATAL_FAILURE);
  }
  this->address.sin_family = AF_INET;
  this->address.sin_addr.s_addr = INADDR_ANY;
  this->address.sin_port = htons(this->port);
  if (bind(this->server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
    perror("bind port failed");
    exit(FATAL_FAILURE);
  }
  flag = fcntl(server_fd, F_GETFL, 0);
}