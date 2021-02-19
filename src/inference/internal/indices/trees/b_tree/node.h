//
// Created by Xiaozhe Yao on 07.12.20.
//

#ifndef EURUSDB_BTREE_NODE_H
#define EURUSDB_BTREE_NODE_H

template<class Key, class Value>
class BTreeNode {
    int degree;
    int num_of_keys;
    bool isLeaf;
    int *keys;
    BTreeNode **child;
public:
    BTreeNode(int degree, bool isLeaf);
    ~BTreeNode();
private:
    void insertNotFull(int key);
};

#endif //EURUSDB_BTREE_NODE_H
