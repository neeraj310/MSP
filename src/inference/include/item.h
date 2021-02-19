//
// Created by Xiaozhe Yao on 08.12.20.
//

#ifndef EURUSDB_ITEM_H
#define EURUSDB_ITEM_H

template <class Key, class Value>
class Item {
 private:
  Key* key;
  Value* value;

 public:
  Item() {
    this->key = new Key();
    this->value = new Value();
  }
  Item(const Key& key, const Value& value) {
    this->key = new Key();
    this->value = new Value();
    *(this->key) = key;
    *(this->value) = value;
  }
  Item(const Item<Key, Value>* item) {
    this->key = new Key();
    this->value = new Value();
    *(this->key) = item->getKey();
    *(this->value) = item->getValue();
  }
  ~Item() {
    if (this->key != NULL) {
      delete this->key;
      this->key = NULL;
    }
    if (this->value != NULL) {
      delete this->value;
      this->value = NULL;
    }
  }
  Key getKey() const {
      return *(this->key);
  }
  Value getValue() const {
      return *(this->value);
  }
};

#endif  // EURUSDB_ITEM_H
