# DID文档和私钥生成工具

这个工具用于根据DID字符串生成对应的DID文档和私钥文档。生成的文件将保存在脚本执行目录下的用户目录中。

## 功能

- 解析DID字符串，提取主机名和路径段
- 生成DID文档和相应的私钥
- 将生成的文档保存到本地文件系统

## 使用方法

```bash
python generate_did_doc.py <did> [--agent-description-url URL] [--verbose]
```

### 参数说明

- `<did>`: 必需参数，DID字符串，例如 `did:wba:service.agent-network-protocol.com:wba:user:lkcoffe`
- `--agent-description-url URL`: 可选参数，代理描述URL
- `--verbose` 或 `-v`: 可选参数，启用详细日志记录

### 示例

```bash
# 基本用法
python generate_did_doc.py "did:wba:service.agent-network-protocol.com:wba:user:lkcoffe"

# 带代理描述URL
python generate_did_doc.py "did:wba:service.agent-network-protocol.com:wba:user:lkcoffe" --agent-description-url "https://example.com/agent.json"

# 启用详细日志
python generate_did_doc.py "did:wba:service.agent-network-protocol.com:wba:user:lkcoffe" -v
```

## 输出文件

脚本将在当前目录下创建一个名为 `user_<unique_id>` 的文件夹，其中 `<unique_id>` 是DID路径的最后一个段。在这个文件夹中，将生成以下文件：

1. `did.json` - DID文档
2. `private_keys.json` - 私钥文档，包含私钥文件的路径和类型信息
3. `<key_id>_private.pem` - 私钥文件，例如 `key-1_private.pem`

### DID文档格式示例

```json
{
  "@context": [
    "https://www.w3.org/ns/did/v1",
    "https://w3id.org/security/suites/jws-2020/v1",
    "https://w3id.org/security/suites/secp256k1-2019/v1"
  ],
  "id": "did:wba:service.agent-network-protocol.com:wba:user:lkcoffe",
  "verificationMethod": [
    {
      "id": "did:wba:service.agent-network-protocol.com:wba:user:lkcoffe#key-1",
      "type": "EcdsaSecp256k1VerificationKey2019",
      "controller": "did:wba:service.agent-network-protocol.com:wba:user:lkcoffe",
      "publicKeyJwk": {
        "kty": "EC",
        "crv": "secp256k1",
        "x": "...",
        "y": "...",
        "kid": "..."
      }
    }
  ],
  "authentication": [
    "did:wba:service.agent-network-protocol.com:wba:user:lkcoffe#key-1"
  ]
}
```

### 私钥文档格式示例

```json
{
  "did": "did:wba:service.agent-network-protocol.com:wba:user:lkcoffe",
  "keys": {
    "key-1": {
      "path": "key-1_private.pem",
      "type": "EcdsaSecp256k1"
    }
  }
}
```

## 注意事项

- 私钥文件包含敏感信息，请妥善保管
- 当前仅支持secp256k1密钥类型
- DID必须以`did:wba:`开头
