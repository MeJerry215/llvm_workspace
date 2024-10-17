# Chapter 1: Toy Language and AST

Chapter 1主要完成的目标是将toy language转换为语法树 ast。

将language 转换为ast 为编译器前端工作，无需特别关注。

目标产物为AST， AST 则是后续优化的起始点，将AST 转换为其他的Dialect，然后进行优化。

其中主要的三个部件为Lexer、AST、Parser

- Parser: 将整个toy language 读入，使用Lexer解析 token，构建AST
- Lexer： token 词法解析器，对应于单个token 解析出类型，以及内容
- AST： 语法树，抽象化的编程语言。

## AST(Abstract Syntax Tree)

核心，也是目标生成物，自定而上的多层次结构树。


```
                                ModuleAST
                                    |
                                    | functions
                                    ↓
                                FunctionAST
                               /           \
                              /             \
                             /               \
                            |                 |
                      proto ↓                 ↓ body
                     PrototypeAST        ExprASTList
                    /        |                     |
                   /         |                     |
              args |         | name                |
        VariableExprAST    string           vector<ExprAST>
                                                   |
                                                   |
NumberExprAST, LiteralExprAST, VariableExprAST, VarDeclExprAST, ReturnExprAST, BinaryExprAST, CallExprAST, PrintExprAST

```

其中核心部分 FunctionAST 分为 proto 和 body 部分，其中 body 为 ExprAST的vector。

而在解析过程中，表达式也是顺序解析的，即vector中存储的expr顺序即为表达式的顺序。

PrototypeAST：则是函数原型，一个函数应该有函数名和参数

ExprAST 分为8种：

- NumberExprAST： 数值类Expr, 只存在浮点double类型
- LiteralExprAST: 暂时不知道这个是什么AST
- VariableExprAST: 变量的语法, 只是存储一个变量名
- VarDeclExprAST：变量声明Expr, 其中包含了变量的初始值
- ReturnExprAST: return 返回值， 返回表达式
- BinaryExprAST: 二进制运运算符表达式
- CallExprAST: 调用定义的其他函数，而非内置op操作
- PrintExprAST: 单独的打印expr，接受参数arg
