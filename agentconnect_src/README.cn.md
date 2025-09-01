<div align="center">
  
[English](README.md) | [中文](README.cn.md)

</div>

## AgentConnect

### AgentConnect是什么

AgentConnect是[Agent Network Protocol(ANP)](https://github.com/agent-network-protocol/AgentNetworkProtocol)的开源实现。

AgentNetworkProtocol(ANP)的目标是成为**智能体互联网时代的HTTP**。

我们的愿景是**定义智能体之间的连接方式，为数十亿智能体构建一个开放、安全、高效的协作网络**。

<p align="center">
  <img src="/images/agentic-web.png" width="50%" alt="Agentic Web"/>
</p>

当前互联网基础设施虽已相当完善，但针对智能体网络的特殊需求，当下仍缺乏最适合的通信和连接方案。我们致力于解决智能体网络面临的三大挑战：

- 🌐 **互联互通**：让所有的智能体相互之间都能够进行通信，打破数据孤岛，让AI能够获得完整的上下文信息。
- 🖥️ **原生接口**：AI无需模仿人类访问互联网，AI应该用它最擅长的方式（API或通信协议）与数字世界交互。
- 🤝 **高效协作**：利用AI，智能体之间可以自组织、自协商，构建比现有互联网更低成本、更高效率的协作网络。

### AgentConnect架构

AgentConnect的技术架构如下图：

<p align="center">
  <img src="/images/agent-connect-architecture.png" width="50%" alt="项目架构图"/>
</p>

对应Agent Network Protocol的三层架构，AgentConnect主要包括以下几个部分：

1. 🔒 **身份认证模块与端到端加密模块**
   主要实现基于W3C DID的身份认证和端到端加密通信，包括DID文档的生成、校验、获取，以及基于DID和ECDHE(Elliptic Curve Diffie-Hellman Ephemeral，椭圆曲线迪菲-赫尔曼临时密钥交换)端到端加密通信方案实现。现在已经支持**基于HTTP的DID身份认证**。

2. 🌍 **元协议模块**
   元协议模块需要基于LLM（大语言模型）和元协议实现，主要功能包含基于元协议的应用协议协商、协议代码实现、协议联调、协议处理等。

3. 📡 **应用层协议集成框架**
   主要的目的是管理和其他智能体通信的协议规范文档以及协议代码，包括应用协议加载、应用协议卸载、应用协议配置、应用协议处理。使用这个框架，智能体可以方便的、按需加载运行所需要的现成协议，加快智能体协议协商过程。

除了以上的功能之外，AgentConnect未来也会在性能、多平台支持等特性上发力：

- **性能**：作为一个基础的代码库，我们希望能够提供极致的性能，未来会用Rust来重写核心部分代码。
- **多平台**：现在支持mac、Linux、windows，未来将会支持移动端、浏览器。

### 文档

- 进一步了解AgnetNetworkProtocol：[Agent Network Protocol(ANP)](https://github.com/agent-network-protocol/AgentNetworkProtocol)
- 如果你想了解我们整体的设计思路和理念，可以查看我们的技术白皮书：[AgentNetworkProtocol技术白皮书](https://github.com/agent-network-protocol/AgentNetworkProtocol/blob/main/chinese/01-AgentNetworkProtocol%E6%8A%80%E6%9C%AF%E7%99%BD%E7%9A%AE%E4%B9%A6.md)

这里有一些我们的blogs:

- 这是我们对智能体网络的理解：[智能体互联网有什么不同](https://github.com/agent-network-protocol/AgentNetworkProtocol/blob/main/blogs/cn/智能体互联网有什么不同.md)

- 这是一个did:wba的简要介绍：[did:wba-基于web的去中心化身份标识符](https://github.com/agent-network-protocol/AgentNetworkProtocol/blob/main/blogs/did:wba-基于web的去中心化身份标识符.md)

- 我们对比了did:wba与OpenID Connect、API keys等技术方案的区别：[did:wba对比OpenID Connect、API keys](https://github.com/agent-network-protocol/AgentNetworkProtocol/blob/main/blogs/cn/did:wba对比OpenID%20Connect、API%20keys.md)

- 我们分析了did:wba的安全性原理：[did:wba安全性原理解析](https://github.com/agent-network-protocol/AgentNetworkProtocol/blob/main/blogs/cn/did:wba安全性原理解析.md)

- 从OpenAI的Operator，谈AI与互联网交互的三种技术路线：[从OpenAI的Operator，看AI与互联网交互的三种技术路线](https://github.com/agent-network-protocol/AgentNetworkProtocol/blob/main/blogs/cn/从OpenAI的Operator，看AI与互联网交互的三种技术路线.md)

### 里程碑

无论是协议还是开源代码实现，我们整体式是按照以下的顺序逐步的推进：

- [x] 构建身份认证与端到端加密通信协议与实现。这是我们整个项目的基础与核心，当前协议设计和代码基本完成。
- [x] 元协议设计与元协议代码实现。当前协议设计和代码开发基本完成。
- [ ] 应用层协议设计与开发。目前正在进行中。

为了推动Agent Network Protocol(ANP)成为行业的标准，我们将会在合适的时间组建ANP标准化委员会，致力于推动ANP成为W3C等国际标准化组织认可的行业标准。

### 安装

```bash
pip install agent-connect
```

### 运行

在安装完agent-connect库后，可以运行我们的demo，体验agent-connect的强大功能。

下载仓库代码：

```bash
git clone https://github.com/agent-network-protocol/AgentConnect.git
```

#### 基于did:wba和HTTP的去中心化身份认证

did:wba是一个基于Web的去中心化身份标识。更多信息：[did:wba, a Web-based Decentralized Identifier](https://github.com/agent-network-protocol/AgentNetworkProtocol/blob/main/blogs/did%3Awba-%E5%9F%BA%E4%BA%8Eweb%E7%9A%84%E5%8E%BB%E4%B8%AD%E5%BF%83%E5%8C%96%E8%BA%AB%E4%BB%BD%E6%A0%87%E8%AF%86%E7%AC%A6.md)。

我们的最新版本已经支持基于did:wba和HTTP的去中心化身份认证。我们提供了一个did:wba服务端用于你的体验与测试。服务端接口文档：[did:wba服务端接口文档](https://github.com/agent-network-protocol/AgentNetworkProtocol/blob/main/chinese/docs/did%3Awba%E6%9C%8D%E5%8A%A1%E7%AB%AF%E6%B5%8B%E8%AF%95%E6%8E%A5%E5%8F%A3.md)。

示例代码路径：`examples/did_wba_examples`。其中：

- basic.py: 这是一个使用DID WBA身份认证的基础示例。它首先创建一个DID文档和私钥，然后将DID文档上传到服务器，最后生成身份认证头并测试DID身份认证。
- full.py: 在basic.py的基础之上，增加了对token的验证，以及对上传的DID文档的验证。
- client.py: 这是一个客户端示例，用于测试你的服务器是否支持DID WBA身份认证。它使用预先创建的DID文档和私钥来访问你服务器上的测试接口。

你可以通过直接运行上面三个文件，来体验DID WBA身份认证。

```bash
python basic.py
python full.py
python client.py
```

你也可以通过我们demo页面来体验DID WBA身份认证：[DID WBA身份认证页面](https://service.agent-network-protocol.com/wba/examples/)。这个页面演示了在一个平台（pi-unlimited.com）上创建DID身份，然后在另外一个平台（service.agent-network-protocol.com）进行身份验证的过程。

#### 元协议协商示例

我们目前支持元协议协商。流程如下：alice和bob先协商出一个协议，然后根据协议生成处理代码，然后运行代码完成数据通信。后面alice和bob就可以使用协议代码直接进行数据通信。

你可以运行examples/negotiation_mode目录下demo代码。先启动bob的节点，再启动alice的节点。

1. 启动bob的节点
```bash
python negotiation_bob.py
``` 

2. 启动alice的节点
```bash
python negotiation_alice.py
```

可以通过日志看到，alice和bob成功连接，然后进行协议的协商，协商通过后，Alice和Bob会根据协议生成协议处理代码，然后运行代码完成数据通信。

> 注意:
> 运行元协议协商需要配置Azure OpenAI（暂时只支持Azure OpenAI）的API Key。请在项目根目录的".env"中配置如下环境变量：AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_MODEL_NAME

### 工具

#### DID文档生成工具

我们提供了一个DID文档生成工具，你可以通过直接运行`python generate_did_doc.py`来生成DID文档。

```bash
python generate_did_doc.py <did> [--agent-description-url URL] [--verbose]
```

详细用法参考文档：[README_did_generater_cn.md](tools/did_generater/README_did_generater_cn.md)。

#### 智能体网络探索工具

您可以通过我们基于网页的工具使用自然语言探索智能体网络：

- [ANP 网络探索工具](https://service.agent-network-protocol.com/anp-explorer/)

该工具允许您：
- 使用自然语言探索智能体网络协议（ANP）生态系统
- 通过 ANP 协议连接智能体世界
- 只需输入智能体描述文档的 URL，即可与各类智能体进行交互

探索工具提供了直观的界面，帮助理解智能体如何在 ANP 框架内通信和运作，使您能够更轻松地可视化网络中不同智能体的连接和能力。

### 联系我们

作者：常高伟  
邮箱：chgaowei@gmail.com  
- Discord: [https://discord.gg/sFjBKTY7sB](https://discord.gg/sFjBKTY7sB)  
- 官网：[https://agent-network-protocol.com/](https://agent-network-protocol.com/)  
- GitHub：[https://github.com/agent-network-protocol/AgentNetworkProtocol](https://github.com/agent-network-protocol/AgentNetworkProtocol)
- 微信：flow10240


## 贡献

欢迎对本项目进行贡献，详细请参阅[CONTRIBUTING.cn.md](CONTRIBUTING.cn.md)。

## 许可证
    
本项目基于MIT许可证开源。详细信息请参阅[LICENSE](LICENSE)文件。


## 版权声明  
Copyright (c) 2024 GaoWei Chang  
本文件依据 [MIT 许可证](./LICENSE) 发布，您可以自由使用和修改，但必须保留本版权声明。
