#include <iostream>
#include <memory>

struct WeakNode {
    std::shared_ptr<WeakNode> next;
    std::weak_ptr<WeakNode> prev; // 防止循环引用
    ~WeakNode()
    {
        std::cout << "WeakNode down" << std::endl;
    }
};

void weakPtrExample()
{
    auto node1 = std::make_shared<WeakNode>();
    auto node2 = std::make_shared<WeakNode>();
    /*
    once: node1
    twice: node2, node1 -> node2

    */
    node1->next = node2;
    node2->prev = node1; // node1 和 node2 之间不会形成循环引用
    std::cout << "node1 use count: " << node1.use_count() << std::endl;
    std::cout << "node2 use count: " << node2.use_count() << std::endl;
}


struct ShareNode {
    std::shared_ptr<ShareNode> next;
    std::shared_ptr<ShareNode> prev; // 防止循环引用
    ~ShareNode()
    {
        std::cout << "ShareNode down" << std::endl;
    }
};


void sharePtrExample()
{
    auto node1 = std::make_shared<ShareNode>();
    auto node2 = std::make_shared<ShareNode>();
    node1->next = node2;
    node2->prev = node1; // node1 和 node2 之间不会形成循环引用
    std::cout << "node1 use count: " << node1.use_count() << std::endl;
    std::cout << "node2 use count: " << node2.use_count() << std::endl;
}


/*
g++ smart_pointer.cpp -o bin/smart_pointer


Output:

running testcase weakPtrExample
node1 use count: 1
node2 use count: 2
WeakNode down
WeakNode down
running testcase sharePtrExample
node1 use count: 2
node2 use count: 2
running all testcase over

ShareNode 没有被释放，所以没有调用析构函数，产生了内存泄漏
*/
int main()
{
    std::cout << "running testcase weakPtrExample" << std::endl;
    weakPtrExample();
    std::cout << "running testcase sharePtrExample" << std::endl;
    sharePtrExample();
    std::cout << "running all testcase over" << std::endl;
    return 0;
}
