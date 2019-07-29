Google Protocol Buffer( 简称 Protobuf) 是 Google 公司内部的混合语言数据标准，目前已经正在使用的有超过 48,162 种报文格式定义和超过 12,183 个 .proto 文件。他们用于 RPC 系统和持续数据存储系统。 
Protocol Buffers 是一种轻便高效的结构化数据存储格式，可以用于结构化数据串行化，或者说序列化。它很适合做数据存储或 RPC 数据交换格式。可用于通讯协议、数据存储等领域的语言无关、平台无关、可扩展的序列化结构数据格式。目前提供了 C++、Java、Python三种语言的 API。

##### 简单例子

使用Protobuf和C++开发一个十分简单的例子程序；该程序由两部分组成。第一部分被称为Writer，第二部分叫做Reader；Writer负责将一些结构化的数据写入一个磁盘文件，Reader则负责从该磁盘文件中读取结构化数据并打印到屏幕上；准备用于演示的结构化数据是HelloWorld，它包含两个基本数据：

- ID，为一个整数类型的数据
- Str，这是一个字符串

##### 书写.proto文件

首先需要编写一个proto文件，定义程序中需要处理的结构化数据，在 protobuf 的术语中，结构化数据被称为 Message。proto 文件非常类似 java 或者 C 语言的数据定义，下面的代码显示了例子应用中的 proto 文件内容。 

```Protobuf
package lm; 
message helloworld 
{ 
    required int32     id = 1;  // ID 
    required string    str = 2;  // str 
    optional int32     opt = 3;  //optional field 
}
```

一个比较好的习惯是将 proto 文件命名规则定于 `packageName.MessageName.proto` 。在上例中，package 名字叫做 lm，定义了一个消息 helloworld，该消息有三个成员，类型为 int32 的 id，另一个为类型为 string 的成员 str。opt 是一个可选的成员，即消息中可以不包含该成员。

##### 编译.proto文件

写好proto文件之后就可以用 Protobuf 编译器将该文件编译成目标语言。本例中使用 C++，假设proto文件存放在 $SRC_DIR下面，想把生成的文件放在同一个目录下的话，可以使用如下命令：

```Bash
protoc -I=$SRC_DIR --cpp_out=$DST_DIR $SRC_DIR/addressbook.proto
```

命令将生成两个文件：lm.helloworld.pb.h，定义C++类的头文件；lm.helloworld.pb.cc。C++ 类的实现文件在生成的头文件中，定义了一个 C++ 类 helloworld，后面的Writer和Reader将使用这个类来对消息进行操作。诸如对消息的成员进行赋值，将消息序列化等等都有相应的方法。

##### 编写writer和reader

Writer将把一个结构化数据写入磁盘，以便其他人来读取。假如我们不使用 Protobuf，一个简单的 Helloworld 也需要写许多处理消息格式的代码。如果使用 Protobuf，那么这些细节就可以不需要应用程序来考虑了。使用Protobuf，Writer 的工作很简单，需要处理的结构化数据由 .proto 文件描述，经过上一节中的编译过程后，该数据化结构对应了一个 C++ 的类，并定义在 lm.helloworld.pb.h 中。对于本例，类名为 lm::helloworld。 Writer 需要 include 该头文件，然后便可以使用这个类了。 

在 Writer 代码中，将要存入磁盘的结构化数据由一个 lm::helloworld 类的对象表示，它提供了一系列的 get/set 函数用来修改和读取结构化数据中的数据成员，或者叫 field。当我们需要将该结构化数据保存到磁盘上时，类 lm::helloworld已经提供相应的方法来把一个复杂的数据变成一个字节序列，我们可以将这个字节序列写入磁盘。对于想要读取这个数据的程序来说，也只需要使用类lm::helloworld的相应反序列化方法来将这个字节序列重新转换会结构化数据。

下面演示了writer的主要代码：

```cpp
#include "lm.helloworld.pb.h"
…

int main(void) 
{ 
	lm::helloworld msg1; 
  	msg1.set_id(101); 
	msg1.set_str(“hello”); 

  	// Write the new address book back to disk. 
  	fstream output("./log", ios::out | ios::trunc | ios::binary); 

  	if (!msg1.SerializeToOstream(&output))
    { 
    	cerr << "Failed to write msg." << endl; 
    	return -1; 
  	}
  	return 0; 
}
```

下面是reader的主要代码：

```cpp
#include "lm.helloworld.pb.h" 
…

void ListMsg(const lm::helloworld & msg)
{ 
    cout << msg.id() << endl; 
    cout << msg.str() << endl; 
} 

int main(int argc, char* argv[])
{ 
    lm::helloworld msg1; 

  	{
    	fstream input("./log", ios::in | ios::binary); 
    	if (!msg1.ParseFromIstream(&input))
        {
      		cerr << "Failed to parse address book." << endl; 
      		return -1; 
        }
    }

  	ListMsg(msg1); 
  	… 
}
```

writer中使用SerializeToOstream将对象序列化后写入一个fstream流；Reader利用 ParseFromIstream 从一个fstream流中读取信息并反序列化。此后，ListMsg 中采用 get 方法读取消息的内部信息，并进行打印输出操作。

这个例子本身并无意义，但只要稍加修改就可以将它变成更加有用的程序。比如将磁盘替换为网络socket，那么就可以实现基于网络的数据交换任务。而存储和交换正是Protobuf最有效的应用领域。