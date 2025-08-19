# DID Document and Private Key Generation Tool

This tool is used to generate corresponding DID documents and private key documents based on a DID string. The generated files will be saved in the user's directory under the script execution directory.

## Features

- Parse DID string to extract hostname and path segments
- Generate DID document and corresponding private key
- Save generated documents to the local file system

## Usage

```bash
python generate_did_doc.py <did> [--agent-description-url URL] [--verbose]
```

### Parameters

- `<did>`: Required parameter, the DID string, e.g., `did:wba:service.agent-network-protocol.com:wba:user:lkcoffe`
- `--agent-description-url URL`: Optional parameter, agent description URL
- `--verbose` or `-v`: Optional parameter, enable detailed logging

### Examples

```bash
# Basic usage
python generate_did_doc.py "did:wba:service.agent-network-protocol.com:wba:user:lkcoffe"

# With agent description URL
python generate_did_doc.py "did:wba:service.agent-network-protocol.com:wba:user:lkcoffe" --agent-description-url "https://example.com/agent.json"

# Enable detailed logging
python generate_did_doc.py "did:wba:service.agent-network-protocol.com:wba:user:lkcoffe" -v
```

## Output Files

The script will create a folder named `user_<unique_id>` in the current directory, where `<unique_id>` is the last segment of the DID path. The following files will be generated in this folder:

1. `did.json` - DID document
2. `private_keys.json` - Private key document, containing the path and type information of private key files
3. `<key_id>_private.pem` - Private key file, e.g., `key-1_private.pem`

### DID Document Format Example

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

### Private Key Document Format Example

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

## Notes

- Private key files contain sensitive information, please keep them secure
- Currently, only secp256k1 key type is supported
- DID must start with `did:wba:`
