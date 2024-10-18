# [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)

类似于[LLVM IR](https://llvm.org/docs/LangRef.html) TODO， 但是引入了 [polyhedral loop optimization](https://polly.llvm.org/) 多面体循环优化 理念。

MLIR 设计目标为：可读性强、适合Debugging、适合变换和分析、结构紧凑便于存储和传输。

[MLIR Rationale](https://mlir.llvm.org/docs/Rationale/Rationale/)，看看就好，阐述MLIR设计理念的。

## High-Level Structure

理解MLIR 中 Operations, Values(Value Type), Block, Region的概念。

MLIR 本质上是基于一种类似于图的数据结构，其中节点称为 "Operation"，边称为 "Value"。

每个 "Value" 则是 "Operation" 或者是 "Block" 的结果。

"Value" 有自己的 [Type System](#Type-System) 定义。

"Operation" 包含于 "Block", "Block" 包含于 "Region" , 且 各自都在自己上级中都是有序的。

"Operation" 也可以包含 "Region", 呈现出多层次的互相嵌套关系。

MLIR 提供了一套可供扩展的对 "Operation" 转换的框架，实现类似于MLIR的[Pass](https://mlir.llvm.org/docs/Passes/) TODO 功能。

MLIR 使用 "[Traits](https://mlir.llvm.org/docs/Traits/)" TODO 和 "[Interfaces](https://mlir.llvm.org/docs/Interfaces/)" TODO 对 "Operation" 进行抽象描述，从而能够进行复杂语义上的转换。

MLIR 是 SSA-Based IR， 静态单赋值，即每个变量仅被赋值一次。


## 语法约定

文档中的语法 使用 [EBNF](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form)的方式描述。

语法类似如下

```shell
alternation ::= expr0 | expr1 | expr2  // Either expr0 or expr1 or expr2.
sequence    ::= expr0 expr1 expr2      // Sequence of expr0 expr1 expr2.
repetition0 ::= expr*  // 0 or more occurrences.
repetition1 ::= expr+  // 1 or more occurrences.
optionality ::= expr?  // 0 or 1 occurrence.
grouping    ::= (expr) // Everything inside parens is grouped together.
literal     ::= `abcd` // Matches the literal `abcd`.
```

## Dialects

"[Dialect](https://mlir.llvm.org/docs/Dialects/)" TODO 是扩展 MLIR 的重要机制，使得能够定义新的 "[Operation](#Operations)".

"Dialect" 具有独立的 namespace, 而namespace 则会作为定义于"Dialect" 内的"Operation"、"Attribute"、 "Type"的前缀

Module 内允许同时存在多种 "Dialect", 而"Dialect" 只被某些Pass 所使用。

MLIR 提供扩展转换框架 "[DialectConversion](https://mlir.llvm.org/docs/DialectConversion/)" TODO 在多种Dialect间转换。

## Operations


MLIR 引入了统一抽象概念 "Operation", 来描述多层次的计算。"Operation" 在MLIR 中是可完全扩展的。并且支持多种内置Builtin Operation Dialect.

语法:

```
operation             ::= op-result-list? (generic-operation | custom-operation)
                          trailing-location?
generic-operation     ::= string-literal `(` value-use-list? `)`  successor-list?
                          dictionary-properties? region-list? dictionary-attribute?
                          `:` function-type
custom-operation      ::= bare-id custom-operation-format
op-result-list        ::= op-result (`,` op-result)* `=`
op-result             ::= value-id (`:` integer-literal)?
successor-list        ::= `[` successor (`,` successor)* `]`
successor             ::= caret-id (`:` block-arg-list)?
dictionary-properties ::= `<` dictionary-attribute `>`
region-list           ::= `(` region (`,` region)* `)`
dictionary-attribute  ::= `{` (attribute-entry (`,` attribute-entry)*)? `}`
trailing-location     ::= `loc` `(` location `)`

```

上面我也看不懂

下面就是典型的Operation 语法举例:

```
// An operation that produces two results.
// The results of %result can be accessed via the <name> `#` <opNo> syntax.
%result:2 = "foo_div"() : () -> (f32, i32)

// Pretty form that defines a unique name for each result.
%foo, %bar = "foo_div"() : () -> (f32, i32)

// Invoke a TensorFlow function called tf.scramble with two inputs
// and an attribute "fruit" stored in properties.
%2 = "tf.scramble"(%result#0, %bar) <{fruit = "banana"}> : (f32, i32) -> f32

// Invoke an operation with some discardable attributes
%foo, %bar = "foo_div"() {some_attr = "value", other_attr = 42 : i64} : () -> (f32, i32)

```

## Blocks

"Block" 即是 "Operation" 的list，每个"Operation" 在list 都是顺序的，"terminator" 操作实实现了"Block" 之间的流程控制。

"Block" 中最后一个 "Operation" 必须是 "terminator operation"， 只有一个"Block" 的 "Region" 可以通过加上 "NoTerminator" 来不遵守上述要求。  典型的就是顶层"ModuleOp"。

"Block" 在MLIR 中输入一串入参称为 "block arguments", 类似于函数。

下面的例子中

```
func.func @simple(i64, i1) -> i64 {
^bb0(%a: i64, %cond: i1): // Code dominated by ^bb0 may refer to %a
  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  cf.br ^bb3(%a: i64)    // Branch passes %a as the argument

^bb2:
  %b = arith.addi %a, %a : i64
  cf.br ^bb3(%b: i64)    // Branch passes %b as the argument

// ^bb3 receives an argument, named %c, from predecessors
// and passes it on to bb4 along with %a. %a is referenced
// directly from its defining operation and is not passed through
// an argument of ^bb3.
^bb3(%c: i64):
  cf.br ^bb4(%c, %a : i64, i64)

^bb4(%d : i64, %e : i64):
  %0 = arith.addi %d, %e : i64
  return %0 : i64   // Return is also a terminator.
}
```

以 "bb" 开头的就是 "Block"(Basic Block), 函数的入口块默认就是"bb0"， "Region" 中存在多个 "Block" 则每个"Block" 结束都为 "terminator operation"。

上述的代码逻辑类似于如下：

```python
if cond:
    c = a
else:
    c = a + a

return a + c

```


"entry Block" 则为首个 "Block", "entry Block"的入参就是"Region"的入参。

"bb0" 作为入口，接受的两个参数a 和 cond, 如果cond为True，则跳转分支 "bb1" 传递 c = a, 否则 跳转分支 "bb2" 传递 c = a + a

到 "bb3" 进行收敛 将 参数 c 和 a 传递给 "bb4", 所以最终结果实际为 "2a" 或者是 "3a"。

## Regions

"Region" 以顺序 "Block" 的方式进行组织，MLIR 当前定义了两种 "Region": "[SSACFG Region](https://mlir.llvm.org/docs/LangRef/#control-flow-and-ssacfg-regions)" 和 "[Graph Region](https://mlir.llvm.org/docs/LangRef/#graph-regions)"。

简单的区别方式就是是否支持控制流。 "SSACFG Region" 可以重点看看。

"Region" 的语法如下:

```
region      ::= `{` entry-block? block* `}`
entry-block ::= operation+
```

简单的说就是 "{}" 包含的内容，"Region" 没有 name 和 address，"Region" 中也只会包含 "Block"。

"Region" 包含于 "Operation"，并且没有 "type" 和 "attribute"。

补充说明的是，函数体的 "body" 就是 "Region"。所以函数的入参就必须和"Region"参数的类型和数目匹配。

定义在"Region"的"Value" 作用域不会超出"Region", 默认情况下"Region" 中可以访问 "Region" 外的值，这个也是可以配置的。

例子如下:

```
  "any_op"(%a) ({ // if %a is in-scope in the containing region...
     // then %a is in-scope here too.
    %new_value = "another_op"(%a) : (i64) -> (i64)
  }) : (i64) -> (i64)
```

## Type System

MLIR 中每个 "Value" 的类型都由"Type System" 所定义。"Type System" 也是可以扩展的。

语法如下：

```
type ::= type-alias | dialect-type | builtin-type

type-list-no-parens ::=  type (`,` type)*
type-list-parens ::= `(` `)`
                   | `(` type-list-no-parens `)`

// This is a common way to refer to a value with a specified type.
ssa-use-and-type ::= ssa-use `:` type
ssa-use ::= value-use

// Non-empty list of names and types.
ssa-use-and-type-list ::= ssa-use-and-type (`,` ssa-use-and-type)*

function-type ::= (type | type-list-parens) `->` (type | type-list-parens)
```

同时也支持"Type Aliases"，就是类型别名，语法如下：

```
type-alias-def ::= `!` alias-name `=` type
type-alias ::= `!` alias-name

# 具体例子如下，foo的两种方式是等价的
!avx_m128 = vector<4 x f32>

// Using the original type.
"foo"(%x) : vector<4 x f32> -> ()

// Using the type alias.
"foo"(%x) : !avx_m128 -> ()
```

如果只想定义Dialect Type

```
dialect-namespace ::= bare-id

dialect-type ::= `!` (opaque-dialect-type | pretty-dialect-type)
opaque-dialect-type ::= dialect-namespace dialect-type-body
pretty-dialect-type ::= dialect-namespace `.` pretty-dialect-type-lead-ident
                                              dialect-type-body?
pretty-dialect-type-lead-ident ::= `[A-Za-z][A-Za-z0-9._]*`

dialect-type-body ::= `<` dialect-type-contents+ `>`
dialect-type-contents ::= dialect-type-body
                            | `(` dialect-type-contents+ `)`
                            | `[` dialect-type-contents+ `]`
                            | `{` dialect-type-contents+ `}`
                            | [^\[<({\]>)}\0]+
```

具体如何定义类型可以参考 [AttributesAndTypes](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/) 
 TODO 来实现定义。

## Properties

"Properties" 是直接存储在 "Operation" 中的额外数据成员。属性通过 "Interface" 访问，并且可以被序列化为 "Attributes"。

## Attributes

语法如下：

```
attribute-entry ::= (bare-id | string-literal) `=` attribute-value
attribute-value ::= attribute-alias | dialect-attribute | builtin-attribute
```

"Properties" 中存储的必然是 常量数据，不可以是变量数据。同时也是可以在 Dialect 中实现扩展。

"Properties" 分为两种：

- inherent attributes: 固有属性，操作必须要使用到的，不带Dialect前缀的属性
- discardable attributes: 可丢弃属性，带Dialect前缀的属性。就是Operation可以选择不用。




