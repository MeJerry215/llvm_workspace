# Chapter 1: Toy Language and AST

Chapter 1主要完成的目标是将toy language转换为语法树 ast。

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
                                     /    /   /     |     \      \
                                    /                             \
                                                                  LiteralExprAST
```