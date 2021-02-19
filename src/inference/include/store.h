//
// Created by Xiaozhe Yao on 08.12.20.
//

#ifndef EURUSDB_STORE_H
#define EURUSDB_STORE_H

template<class Key, class Value>
class Store {
public:
    virtual const Value* get(const Key &key) = 0;
    virtual void set(const Key &key, const Value &val) = 0;
    virtual ~Store() {

    }
};


#endif //EURUSDB_STORE_H
