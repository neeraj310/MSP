//
// Created by Xiaozhe Yao on 24.12.20.
//

#ifndef EURUSDB_BTREE_H
#define EURUSDB_BTREE_H

#include "../include/headers.h"
#include "node.h"

template<class Key, class Value>
class BTree : public Store<Key, Value> {

private:
    int degree;
    BTreeNode<Key, Value> * root;
    

};

#endif //EURUSDB_BTREE_H