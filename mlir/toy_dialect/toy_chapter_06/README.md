# [Chapter 6: Lowering to LLVM and CodeGeneration](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)

无论中间有多少层MLIR，经过多少层的优化，最终Lowrering到底层的目标都是LLVM IR， 并且生成可执行二进制。

所以本章节的目标是Lowering 到最终目标，并且将我们的example 运行起来。

上一章节中，我们将除了`PrintOp`以为的其他的Operation Lowering 为`Affine`、`Arith`、`Func`、`MemRef`等Dialect，本章节最终的目标是转换为统一的Dialect LLVMIR。

转换路径为

```
toy::mul -> affine::mulf -> llvm::mul -> llvm asm
...
toy::print               -> llvm::call a set of ir -> llvm asm
```

所以并不是上层Dialect 遵循统一的Lowering 层次，允许MLIR::Module 是Mixture IRModule。

目录结构如下：
```
|-- CMakeLists.txt
|-- README.md
|-- affine-lowering.mlir
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


## Lowering To LLVM

[LLVM Dialect](https://mlir.llvm.org/docs/Dialects/LLVM/) TODO。LLVM IR 则类似于ASM层面的东西。

`toy::PrintOp` 可以通过直接调用内置的函数来达到print的目的，而调用一个内置函数首先需要得到函数的符号。即类似于函数声明一下，添加进IRModule，同时由于PrintOp 原本的入参为带shape 信息的memref。

所以lowering 调用print函数的时候，需要按照shape信息，依次进行外部循环、外部循环的构建，同时在每一次迭代的结束插入 `nl` 字符串即换行，在循环的最内部，则需要调用print函数，带上格式化参数frmt_spec和常量值(通过memref::Load获取)调用。

`PrintOp` 单独Lowering也是和上一章节一样创建Lowering Pass，同时增加对应Op的ConversionPattern。


```c++
    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
        ConversionPatternRewriter& rewriter) const override
    {
        auto* context = rewriter.getContext();
        auto memRefType = llvm::cast<MemRefType>((*op->operand_type_begin()));
        auto memRefShape = memRefType.getShape();
        auto loc = op->getLoc();
        ModuleOp parentModule = op->getParentOfType<ModuleOp>();
        // Get a symbol reference to the printf function, inserting it if necessary.
        auto printfRef = getOrInsertPrintf(rewriter, parentModule);
        Value formatSpecifierCst = getOrCreateGlobalString(
                loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
        Value newLineCst = getOrCreateGlobalString(
                loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);
        // Create a loop for each of the dimensions within the shape.
        SmallVector<Value, 4> loopIvs;

        for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
            auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
            auto upperBound =
                rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
            auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
            auto loop =
                rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);

            for (Operation& nested : *loop.getBody())
                rewriter.eraseOp(&nested);

            loopIvs.push_back(loop.getInductionVar());
            // Terminate the loop body.
            rewriter.setInsertionPointToEnd(loop.getBody());

            // Insert a newline after each of the inner dimensions of the shape.
            if (i != e - 1)
                rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef,
                    newLineCst);

            rewriter.create<scf::YieldOp>(loc);
            rewriter.setInsertionPointToStart(loop.getBody());
        }

        // Generate a call to printf for the current element of the loop.
        auto printOp = cast<toy::PrintOp>(op);
        auto elementLoad =
            rewriter.create<memref::LoadOp>(loc, printOp.getInput(), loopIvs);
        rewriter.create<LLVM::CallOp>(
            loc, getPrintfType(context), printfRef,
            ArrayRef<Value>({formatSpecifierCst, elementLoad}));
        // Notify the rewriter that this operation has been removed.
        rewriter.eraseOp(op);
        return success();
    }

```

创建`ToyToLLVMLoweringPass`, 同样定义Conversion Target, 这里是`LLVMConversionTarget`，相比普通ConversionTarget，默认添加进来了 `mlir::LLVMDialect`作为默认`legal`的Dialect。

然后就是添加上各种Pass，这里可以认为是照抄某个模板。 然后添加创建Pass接口

```c++

void ToyToLLVMLoweringPass::runOnOperation()
{
    /*
    和如下定义是等价的。
      mlir::ConversionTarget target(getContext());
      target.addLegalDialect<mlir::LLVMDialect>();
      target.addLegalOp<mlir::ModuleOp>();
    */
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();


    LLVMTypeConverter typeConverter(&getContext());


    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    patterns.add<PrintOpLowering>(&getContext());

    auto module = getOperation();

    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}


std::unique_ptr<mlir::Pass> mlir::toy::createLowerToLLVMPass()
{
    return std::make_unique<ToyToLLVMLoweringPass>();
}

```

至此所有的上层IR 都被Lowering 为LLVM IR Dialect。

## CodeGen: Getting Out of MLIR

在获得LLVM Dilect之后，就可以生成LLVM IR。模板化代码，照抄即可。

```c++
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());
    // Convert the module to LLVM IR in a new LLVM IR context.
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);

    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return -1;
    }

    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    // Configure the LLVM Module
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();

    if (!tmBuilderOrError) {
        llvm::errs() << "Could not create JITTargetMachineBuilder\n";
        return -1;
    }

    auto tmOrError = tmBuilderOrError->createTargetMachine();

    if (!tmOrError) {
        llvm::errs() << "Could not create TargetMachine\n";
        return -1;
    }

    mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
        tmOrError.get().get());
    /// Optionally run an optimization pipeline over the llvm module.
    auto optPipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);

    if (auto err = optPipeline(llvmModule.get())) {
        llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }

    llvm::errs() << *llvmModule << "\n";
```

产生了LLVM IR 之后则是希望能够运行代码查看我们自己定义的toy language 能否运行，
这里使用JIT方式运行 toy language, 也是模板代码，拷贝即可。


```c++
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    // Register the translation from MLIR to LLVM IR, which must happen before we
    // can JIT-compile.
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());
    // An optimization pipeline to use within the execution engine.
    auto optPipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);
    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
    // the module.
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto& engine = maybeEngine.get();
    // Invoke the JIT-compiled function.
    auto invocationResult = engine->invokePacked("main");
```


