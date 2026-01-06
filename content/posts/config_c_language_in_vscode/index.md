+++
date = '2026-01-06T17:52:23+08:00'
draft = false
title = 'VSCode中配置C/C++运行和调试环境'
categories = 'tools'
tags = ['C/C++'] 
+++

首先配置C/C++的插件，然后执行`Ctrl+Shift+B`执行生成任务，注意此时焦点一定要将待执行的文件上，否则会报错。

然后，生成`lunch.json`文件，下面是我的配置模板，该模板可复用，但我现在使用的是WSL环境，不用的操作系统细节可能有些不同。主要的修改点在`"program": "${workspaceRoot}/${fileBasenameNoExtension}"`和`"miDebuggerPath": "/usr/bin/gdb"`，前者的意思是调试工作目录下的当前可执行文件，后者是你的GDB路径，在Linux下执行`which gdb`即可得到，如果没有出现路径，安装GDB即可。

```json
{
  // 使用 IntelliSense 了解相关属性。
  // 悬停以查看现有属性的描述。
  // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
  "version": "0.2.0",
  "configurations": [
    {
      "name": "(gdb) 启动",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceRoot}/${fileBasenameNoExtension}",
      "args": [],
      "stopAtEntry": false,
      "cwd": "${fileDirname}",
      "environment": [],
      "externalConsole": false,
      "MIMode": "gdb",
      "miDebuggerPath": "/usr/bin/gdb",
      "setupCommands": [
        {
          "description": "为 gdb 启用整齐打印",
          "text": "-enable-pretty-printing",
          "ignoreFailures": true
        },
        {
          "description": "将反汇编风格设置为 Intel",
          "text": "-gdb-set disassembly-flavor intel",
          "ignoreFailures": true
        }
      ]
    }
  ]
}
```

值得一提的是，现在的配置文件是单文件配置，对于一个大型项目肯定包含多个C/C++文件，当前配置可能不适用，我还在探索中，有机会后面在补充。

好了，这就是这篇博客的全部内容，简单讲解了一下VSCode中配置C/C++运行和调试环境的核心内容，当然还有一些前置条件没提，比如安装`C/C++`插件，配置该插件，这些请读者自行探索。
