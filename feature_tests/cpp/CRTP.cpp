#include <iostream>

template <typename Derived>
class Base {
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();
    }
};

class Derived1 : public Base<Derived1> {
public:
    void implementation() {
        std::cout << "Derived1 implementation" << std::endl;
    }
};

class Derived2 : public Base<Derived2> {
public:
    void implementation() {
        std::cout << "Derived2 implementation" << std::endl;
    }
};


/*
CRTP(Curiously Recurring Template Pattern)
模板的方式实现多态，在编译时确定调用函数，避免使用虚函数表从而产生额外的开销。

*/

int main() {
    Derived1 d1;
    Derived2 d2;

    d1.interface(); // 输出: Derived1 implementation
    d2.interface(); // 输出: Derived2 implementation
}
