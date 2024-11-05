# [Chapter 7: Adding a Composite Type to Toy](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/)

在上一章节中，toy language lowering 成 llvm ir，并且能够通过JIT 的方式运行已经达成端到端。

而在这一章节中，则继续扩展 toy language以支持 自定义数据类型 `struct`。

这一章节，目录结构没有任何改变，而是增加对结构体的解析，以及对现有的一些定义的修改。改动相对来说比较琐碎。

在一个语言中，存在比较复杂的数据类型结构类型，本章添加了`struct` 作为自定义新增的复合数据类型。

目录结构:

```
|-- CMakeLists.txt
|-- README.md
|-- include
|   |-- AST.h
|   |-- Dialect.h
|   |-- Lexer.h
|   |-- MLIRGen.h
|   |-- Parser.h
|   |-- Passes.h
|   `-- ShapeInferenceInterface.h
|-- mlir
|   |-- Dialect.cpp
|   `-- MLIRGen.cpp
|-- parser
|   `-- AST.cpp
|-- pass
|   |-- CMakeLists.txt
|   |-- LowerToAffineLoops.cpp
|   |-- LowerToLLVM.cpp
|   |-- ShapeInferenceInterface.td
|   |-- ShapeInferencePass.cpp
|   |-- ToyCombine.cpp
|   `-- ToyCombine.td
|-- td
|   |-- CMakeLists.txt
|   `-- Ops.td
`-- toy.cpp
```

## Define `struct` in Toy Language

对于新增的数据类型`struct` 的语法和c 语言中的定义很像，通过关键字`struct` 表示，并且存在结构体的类型名，还有结构体内的变量。结构体在定义后可以在后续的代码中使用。

```
struct Struct {
  var a;
  var b;
}

# User defined generic function may operate on struct types as well.
def multiply_transpose(Struct value) {
  # We can access the elements of a struct via the '.' operator.
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # We initialize struct values using a composite initializer.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};

  # We pass these arguments to functions like we do with variables.
  var c = multiply_transpose(value);
  print(c);
}
```

MLIR 只提供基础数据类型，而复杂数据类型需要用户自己定义。本文中新增的`struct`只会在 AST 层面上使用，到了toy dialect转换为特定的Op，而到了更底层则会消失。所以不需要考虑更多MLIR 上的表示。

### Define the Type Class

MLIR 中的类型对象是值类型的，依赖于保存该类型的实际数据的内部存储对象。

`Type` 类本身充当内部 `TypeStorage` 对象的简单包装器，该对象在 `MLIRContext` 实例中是唯一的。在构造 `Type` 时，我们在内部只是构造和唯一化存储类的实例。

#### Define the Storage Class

类型内部存储对象包含了所有构建该种`Type`的所有数据信息，存储对象类必须继承基类`mlir::TypeStorage`并且提供一组对应的实现`hook`。

`StructTypeStorage`只会被自定义的类型所使用的，所以不必公开定义在头文件中，可以只实现在cpp文件中，被`StructType` 访问即可。

以下就是存储对象类定义的例子：

```c++
struct StructTypeStorage : public mlir::TypeStorage {

  /// KeyTy 定义为结构体内元素类型的Array，该类型用来作为结构体实例比较的key使用，具有唯一性。
  /// 相同结构体类型的KeyTy是相等的，即其内包含的元素类型是一致的。
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  /// 比较两个类型存储结构体是否一样，如果一样则内部存储的元素的类型是一样的，即key一样
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// 这个函数非必要，完全可以忽略。因为需要llvm::ArrayRef和mlir::Type都有hash函数才有效
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// 非必要方法，可以通过构造函数构造。
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// 动态构造该类，使用allocator
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  /// 类型存储结构存储的类型
  llvm::ArrayRef<mlir::Type> elementTypes;
};
```


#### Define the Type Class

当类型存储类被定义，就可以增加user-visible 的`StructType` 类。

新增数据类型需要继承自`mlir::Type::TypeBase`, 持有数据存储类`detail::StructTypeStorage`

具有如下的声明:


```c++
/// This class defines the Toy struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
    detail::StructTypeStorage>
{
public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// 创建 StructType的方法，类似于工厂类方法了。
    static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

    // 返回持有数据类型类的元素类型
    llvm::ArrayRef<mlir::Type> getElementTypes();

    // 大概类似于结构体定义多少个元素
    size_t getNumElementTypes() { return getElementTypes().size(); }

    /// 结构体的名称
    static constexpr StringLiteral name = "toy.struct";
};
```

而在Toy Dialect初始化的时候，这次将类型添加进去。

```c++
void ToyDialect::initialize()
{
    addOperations <
#define GET_OP_LIST
#include "Ops.cpp.inc"
    > ();
    addInterfaces<ToyInlinerInterface>();
    addTypes<StructType>();
}
```

###  Expose to ODS

这个时候还需要将类型暴露给ODS框架，感知到数据类型的新增。这样就可以在Operation定义的时候自动生成对应的代码, 这里对于判断一个类型是否为 `StructType`, 使用CPred的方式 c++代码直接判断。

而后定义的 `Toy_Type` 在 ODS 中使用，`Toy_Type` 可以是内置的F64Tensor，也可以是自定义的数据类型 `Toy_StructType`


```
def Toy_StructType :
    DialectType<Toy_Dialect, CPred<"::llvm::isa<StructType>($_self)">,
                "Toy struct type">;

// Provide a definition of the types that are used within the Toy dialect.
def Toy_Type : AnyTypeOf<[F64Tensor, Toy_StructType]>;

```


### Parsing and Printing

此时可以在MLIR generation 和 transformation 过程中使用 `StructType`， 但是此时是不能输出和解析`.mlir`，所以需要增加如何解析和转换`StructType`。实现上述功能，则要重写 `parseType`和 `printType` 方法。这两个方法已经通过expose to ODS的时候生成了头文件在`Dialect.h.inc` 文件中

```c++
/// Parse a type registered to this dialect.
::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

/// Print a type registered to this dialect.
void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &os) const override;

```

对于Parsing 和 Printing 的MLIR 要符合 [MLIR language reference](https://mlir.llvm.org/docs/LangRef/#dialect-types) 所描述的那样。

```
struct-type ::= `struct` `<` type (`,` type)* `>`
```

#### Parsing

循环解析类型

```c++

/// Parse an instance of a type registered to the toy dialect.
mlir::Type ToyDialect::parseType(mlir::DialectAsmParser& parser) const
{
    // Parse a struct type in the following form:
    //   struct-type ::= `struct` `<` type (`,` type)* `>`
    // NOTE: All MLIR parser function return a ParseResult. This is a
    // specialization of LogicalResult that auto-converts to a `true` boolean
    // value on failure to allow for chaining, but may be used with explicit
    // `mlir::failed/mlir::succeeded` as desired.

    // Parse: `struct` `<`
    if (parser.parseKeyword("struct") || parser.parseLess())
        return Type();

    // Parse the element types of the struct.
    SmallVector<mlir::Type, 1> elementTypes;

    do {
        // Parse the current element type.
        SMLoc typeLoc = parser.getCurrentLocation();
        mlir::Type elementType;

        if (parser.parseType(elementType))
            return nullptr;

        // Check that the type is either a TensorType or another StructType.
        if (!llvm::isa<mlir::TensorType, StructType>(elementType)) {
            parser.emitError(typeLoc, "element type for a struct must either "
                "be a TensorType or a StructType, got: ")
                    << elementType;
            return Type();
        }

        elementTypes.push_back(elementType);
        // Parse the optional: `,`
    } while (succeeded(parser.parseOptionalComma()));

    // Parse: `>`
    if (parser.parseGreater())
        return Type();

    return StructType::get(elementTypes);
}
```



#### Printing

Printing 就相对来说较为简单了。

```c++

/// Print an instance of a type registered to the toy dialect.
void ToyDialect::printType(mlir::Type type,
    mlir::DialectAsmPrinter& printer) const
{
    // Currently the only toy type is a struct type.
    StructType structType = llvm::cast<StructType>(type);
    // Print the struct type according to the parser format.
    printer << "struct<";
    llvm::interleaveComma(structType.getElementTypes(), printer);
    printer << '>';
}
```

### Operating On StructType

现在需要对Operation 进行一下改造从而能够在当前Operation 上支持自定义的数据类型。

#### Updating Existing Operations

对现有的Operation 进行修改，以支持自定义的 `Toy_StructType`, 对于Call Operation 支持入参和出参都是`Toy_Type`

这里的问题在于为什么 arguments 和 results 是`Toy_Type`，而不是`Toy_StructType`:

对于函数调用来说可以是原来的 `Tensor`类型，也可以是新定义的`Toy_StructType`，如果仅支持`Toy_StructType` 会导致不兼容。

```
def GenericCallOp : Toy_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  ...

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<Toy_Type>:$inputs);

  // The generic call operation returns a single value of TensorType or
  // StructType.
  let results = (outs Toy_Type);

  ...
}


def ReturnOp : Toy_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  ...

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<Toy_Type>:$input);

  ...
}

```


#### Adding New Toy Operations

除了对现存的operations进行改造，而要增加一些其他的operation 提供更具体的struct type的操作，要能够访问 `Toy_StructType` 存储的元素类型。

提供`StructConstantOp` Operation 来定义 `Toy_StructType` 的复合常量, `StructAccessOp` 来访问对应索引的元素。

##### toy.struct_constant

```
def StructConstantOp : Toy_Op<"struct_constant", [ConstantLike, Pure]> {
  let summary = "struct constant";
  let description = [{
    Constant operation turns a literal struct value into an SSA value. The data
    is attached to the operation as an attribute. The struct constant is encoded
    as an array of other constant values. For example:

    ```mlir
      %0 = toy.struct_constant [
        dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
      ] : !toy.struct<tensor<*xf64>>
    ```
  }];

  let arguments = (ins ArrayAttr:$value);
  let results = (outs Toy_StructType:$output);

  let assemblyFormat = "$value attr-dict `:` type($output)";

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
  let hasFolder = 1;
}
```

##### toy.struct_access



```
def StructAccessOp : Toy_Op<"struct_access", [Pure]> {
  let summary = "struct access";
  let description = [{
    Access the Nth element of a value returning a struct type.
  }];

  let arguments = (ins Toy_StructType:$input, I64Attr:$index);
  let results = (outs Toy_Type:$output);

  let assemblyFormat = [{
    $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
  }];

  // Allow building a StructAccessOp with just a struct value and an index.
  let builders = [
    OpBuilder<(ins "Value":$input, "size_t":$index)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;

  // Set the folder bit so that we can fold constant accesses.
  let hasFolder = 1;
}
```

#### Optimizing Operations On StructType

这里引入一个新的优化技术： constant fold， 常量折叠，当某个Operation的入参和属性都是确定的时候，这个时候这个Operation 就可以从IRModule 中移除，可以提前计算出该Opeartion 的结果。

这里对于`toy.struct_access` 在MLIR 使用过程中，访问了结构体中的一个元素或全部元素，则这个时候通过 fold 消除 toy.struct_access的调用，从而避免`toy.struct_access` 的调用，这也就是说为什么 struct 最多到 Toy MLIR层面的原因，因为这个层面以下就会被优化掉。

同理对于 `toy.constant` 和 `toy.struct_constant` 也是同样的道理。

对于使能常量折叠，则需要再对应的Operation td文件定义中 `let hasFolder = 1` 即可。


当前使能了fold 的Opeartion存在： ConstantOp、StructConstantOp、StructAccessOp。

参考其中的一个实现的定义如下：

```c++
/// Fold constants.
OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

/// Fold struct constants.
OpFoldResult StructConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

/// Fold simple struct access operations that access into a constant.
OpFoldResult StructAccessOp::fold(FoldAdaptor adaptor)
{
    auto structAttr =
        llvm::dyn_cast_if_present<mlir::ArrayAttr>(adaptor.getInput());

    if (!structAttr)
        return nullptr;

    size_t elementIndex = getIndex();
    return structAttr[elementIndex];
}

```

而为能够在优化过程中，保证始终保证产生正确的结果类型，需要实现`materializeConstant` 方法。

`StructType`封装为`StructConstantOp`, 其他类型封装为`ConstantOp`。

```c++
mlir::Operation* ToyDialect::materializeConstant(mlir::OpBuilder& builder,
    mlir::Attribute value,
    mlir::Type type,
    mlir::Location loc)
{
    if (llvm::isa<StructType>(type))
        return builder.create<StructConstantOp>(loc, type,
                llvm::cast<mlir::ArrayAttr>(value));

    return builder.create<ConstantOp>(loc, type,
            llvm::cast<mlir::DenseElementsAttr>(value));
}

```