import fastjsonschema

"""
JSON Schema definitions for validating JSON-RPC requests in PearlGateway.
"""

# Schema for the JSON-RPC request envelope
JSON_RPC_SCHEMA = {
    "type": "object",
    "required": ["jsonrpc", "method", "id"],
    "properties": {
        "jsonrpc": {"type": "string", "enum": ["2.0"]},
        "method": {"type": "string"},
        "params": {"type": "object"},
        "id": {"type": ["string", "number", "null"]},
    },
    "additionalProperties": False,
}

# Schema for getMiningInfo request.  Empty params keep the worker-0 behavior.
GET_MINING_INFO_SCHEMA = {
    "type": "object",
    "properties": {
        "worker_id": {"type": "integer", "minimum": 0, "maximum": 255},
    },
    "additionalProperties": False,
}

# Base64 pattern for encoding
BASE64_PATTERN = "^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$"
MAX_PROOF_BASE64_CHARS = ((8 * 1024 * 1024 + 2) // 3) * 4

# Schema for submitPlainProof request - simplified to just base64 string + mining_job
SUBMIT_PLAIN_PROOF_SCHEMA = {
    "type": "object",
    "required": [
        "plain_proof",
        "mining_job",
    ],
    "properties": {
        "plain_proof": {
            "type": "string",
            "pattern": BASE64_PATTERN,
            "minLength": 4,
            "maxLength": MAX_PROOF_BASE64_CHARS,
        },
        "mining_job": {
            "type": "object",
            "required": ["incomplete_header_bytes", "target", "cert_version"],
            "properties": {
                "incomplete_header_bytes": {
                    "type": "string",
                    "pattern": BASE64_PATTERN,
                    "minLength": 104,
                    "maxLength": 104,
                },
                "target": {"type": "integer", "minimum": 1},
                "target_decimal": {
                    "type": "string",
                    "pattern": "^[1-9][0-9]{0,77}$",
                },
                "cert_version": {"type": "integer", "minimum": 1},
                "expected_reward": {"type": "integer", "minimum": 0},
                "worker_id": {"type": "integer", "minimum": 0, "maximum": 255},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

# Pre-compile validators for better performance
validate_jsonrpc = fastjsonschema.compile(JSON_RPC_SCHEMA)
validate_get_mining_info = fastjsonschema.compile(GET_MINING_INFO_SCHEMA)
validate_submit_plain_proof = fastjsonschema.compile(SUBMIT_PLAIN_PROOF_SCHEMA)
