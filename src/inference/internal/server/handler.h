//
// Created by Xiaozhe Yao on 12.12.20.
//

#ifndef EURUSDB_HANDLER_H
#define EURUSDB_HANDLER_H

class Handler {
    virtual int handleGet(const int &clientfd) = 0;
    virtual int handleSet(const int &clientfd) = 0;
    virtual int handleExit(const int &clientfd) = 0;
    virtual int handleRemove(const int &clientfd) = 0;
};

#endif //EURUSDB_HANDLER_H