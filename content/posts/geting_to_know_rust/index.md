+++
date = '2026-01-21T23:13:14+08:00'
draft = true
title = '初识Rust'
tags = ["Rust"]
image = "cover.png"
+++

## Rust是一门怎样的编程语言?

Rust是一门系统编程语言，像C一样接近系统底层，同时也是一门编译型语言，速度能和C媲美。而Rust最大的特点是其严格的内存管理方式，它不像其他高级语言一样拥有`GC`机制来管理内存，也不想`C/C++`一样容易因为手动管理内存而导致内存泄露和`double free`的问题。

> **GC（Garbage Collection，垃圾回收）**： 是一种自动内存管理机制。它存在于许多高级编程语言（如 Java, C#, Python, Go,JavaScript等）的运行时环境中，其核心任务是：自动识别并释放程序中不再使用的内存（即“垃圾”），交还给系统，防止内存泄漏。
> 
> **为什么需要 GC？**
> 
> 在像 C/C++ 这样的语言中，程序员必须手动管理内存：用 malloc/new 申请，用 free/delete释放。这带来了两个主要问题：
> 1. 内存泄漏：忘记释放内存，导致内存被持续占用，最终可能耗尽。
> 2. 悬空指针野指针：释放了内存后，又继续使用指向该内存的指针，导致程序崩溃或数据错误。
> 
> GC通过自动化这个过程，将程序员从复杂且易错的手动内存管理中解放出来，提高了开发效率和程序的健壮性。

## Rust是如何实现安全可靠的内存管理的？

### 所有权机制

所有权机制是`Rust`实现内存管理的核心, 所有权机制主要用与管理堆上的数据，而这里就要理解**堆**和**栈**的概念。

* 栈：先进后出的数据结构，存放在栈中的数据必须是已知且固定的大小，直接寻址。
* 堆：堆是缺乏组织的，在堆上分配内存时，操作系统会找到一块足够大的空闲空间，标记为已用，并返回一个指针。分配速度较慢，且需要通过指针间接访问，间接寻址。

首先，让我们看一下所有权的规则。当我们通过举例说明时，请谨记这些规则：

> 1. Rust 中的每一个值都有一个 所有者（owner）。
> 2. 值在任一时刻有且只有一个所有者。
> 3. 当所有者离开作用域，这个值将被丢弃。

#### Rust 中的每一个值都有一个 所有者（owner）。

首先，观察下面的一个例子。

```rust
fn main() {
    let s1 = String::from("hello");
    let s2 = s1;
    println!("{s1}, world");
}
```

这段代码首先创建一个不可变的字符串`s1`，然后将`s1`赋值给`s2`，最后打印一个和`s1`拼接的字符串。
在其他语言中这样的做法完全没有问题，但是在`rust`中这是不合法的行为，第一次创建`s1`时就表示`s1`对字符串`"hello`有所有权，
当其赋值给`s2`时表示它将其对`"hello`的所有权转交给`s2`。

从内存的角度来看，传统的编程语言在这种情形会将指向`"hello`的指针复制一份
给`s2`这叫**浅拷贝**，而`rust`则会在复制的同时将`s1`的指针给删除掉，在`rust`的中这种行为叫做**移动**。

这就能解释为什么上面的
代码是不合法的了，因为当`s1`赋值给`s2`后，`s1`的指针就被删除了，再次试图访问`s1`自然就会报错。

从报错信息也可以给看出，出错的原因在于借用了已经移动的值
`Compiling learn_rust v0.1.0 (/home/codersgl/study/learn_rust)
error[E0382]: borrow of moved value: `s1``。


```text
➜  learn_rust git:(master) ✗ cargo run
   Compiling learn_rust v0.1.0 (/home/codersgl/study/learn_rust)
error[E0382]: borrow of moved value: `s1`
 --> src/main.rs:4:16
  |
2 |     let s1 = String::from("hello");
  |         -- move occurs because `s1` has type `String`, which does not implement the `Copy` trait
3 |     let s2 = s1;
  |              -- value moved here
4 |     println!("{s1}, world");
  |                ^^ value borrowed here after move
  |
  = note: this error originates in the macro `$crate::format_args_nl` which comes from the expansion of the macro `println` (in Nightly
 builds, run with -Z macro-backtrace for more info)
help: consider cloning the value if the performance cost is acceptable
  |
3 |     let s2 = s1.clone();
  |                ++++++++

warning: unused variable: `s2`
 --> src/main.rs:3:9
  |
3 |     let s2 = s1;
  |         ^^ help: if this is intentional, prefix it with an underscore: `_s2`
  |
  = note: `#[warn(unused_variables)]` (part of `#[warn(unused)]`) on by default

For more information about this error, try `rustc --explain E0382`.
warning: `learn_rust` (bin "learn_rust") generated 1 warning
error: could not compile `learn_rust` (bin "learn_rust") due to 1 previous error; 1 warning emitted

```

#### 值在任一时刻有且只有一个所有者。

`rust`的这种特性，能有效避免`double free`。正如上面的例子，如果一个值同时拥有`s1`和`s2`两个所有者的话，当释放完`s1`的内存后，如果不小心
试图释放`s2`的内存就会导致`double free`，而`rust`从机制上规避了这种情况。

#### 当所有者离开作用域，这个值将被丢弃。

每当变量的所有者离开作用域后其值对应的内存地址会被自动释放，这时`rust`会自动调用一个名为`drop`的函数来释放内存。

