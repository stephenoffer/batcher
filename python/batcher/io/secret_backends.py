"""Key-store backends for a secret reference, resolved on the machine that needs the secret.

`credentials.resolve_secret` answers ``env:``, ``file:`` and ``cmd:``. Those three cover
every deployment, because a platform's own secret delivery (Vault Agent, the External
Secrets Operator, the secrets-store CSI driver) materializes a secret as a file, and
``cmd:`` reaches anything else through an operator-configured helper. What they do not
cover is the case where nobody has set that up yet, and every team writes the same
twenty-line shell shim around ``aws secretsmanager get-secret-value``.

This module is those shims, written once. Each scheme names a key store directly:

=====================  ==================================================================
``vault:``             HashiCorp Vault KV v2. ``vault:secret/data/db#password``.
``aws-sm:``            AWS Secrets Manager. ``aws-sm:prod/warehouse``.
``aws-ssm:``           AWS SSM Parameter Store. ``aws-ssm:/prod/warehouse/password``.
``gcp-sm:``            GCP Secret Manager. ``gcp-sm:projects/p/secrets/s/versions/latest``.
``azure-kv:``          Azure Key Vault. ``azure-kv:https://v.vault.azure.net/secrets/db``.
=====================  ==================================================================

# Every one of these runs on the worker, not the driver

That is not an implementation detail, it is the reason the reference scheme exists.
`resolve_secret` is called where a connection is opened, so on a distributed query each
worker resolves against *its own* ambient identity — an IRSA role, a Workload Identity
binding, a Vault Kubernetes auth login. Only the reference is pickled into a split. A
driver-side resolution would put the plaintext on the object that travels, which is the
one thing this whole design exists to prevent.

It also means the credential a worker uses is the credential that machine is entitled to,
so a node without a role fails closed rather than inheriting the driver's.

# Dependencies

Vault needs nothing: its KV API is a GET with a token header, so it is answered with
`urllib`. The three cloud stores use their vendor SDK, imported only when that scheme is
actually used, and a missing one raises `BackendError` naming the extra to install.

# No caching, deliberately

The same reasoning as `credentials._from_command`: this runs once per connection, not per
batch the way the data plane's own resolver does, so a cache would buy nothing and would
keep a rotated secret alive past its rotation. The data plane caches because it resolves
per batch; this does not because it does not.
"""

from __future__ import annotations

import json
import os
from typing import Any

from batcher._internal.errors import BackendError

__all__ = ["BACKEND_SCHEMES", "resolve_backend_ref"]

#: The schemes this module answers. `credentials.is_secret_ref` includes these, so a
#: reference using one is recognized as a reference rather than passed through as a literal.
BACKEND_SCHEMES = ("vault:", "aws-sm:", "aws-ssm:", "gcp-sm:", "azure-kv:")


def resolve_backend_ref(scheme: str, target: str, *, what: str, reference: str) -> str:
    """Fetch the secret `target` names from the key store `scheme` selects.

    Args:
        scheme: The reference scheme, without its colon (e.g. ``"aws-sm"``).
        target: Everything after the colon — the store-specific identifier.
        what: What is being resolved, for the error message.
        reference: The whole original reference, for the error message.

    Returns:
        The secret material.

    Raises:
        BackendError: If the store is unreachable, the secret is absent, or the SDK the
            scheme needs is not installed. The message names the *reference*, never the
            secret.
    """
    handlers = {
        "vault": _from_vault,
        "aws-sm": _from_aws_secrets_manager,
        "aws-ssm": _from_aws_parameter_store,
        "gcp-sm": _from_gcp_secret_manager,
        "azure-kv": _from_azure_key_vault,
    }
    handler = handlers[scheme]
    try:
        return handler(target, what=what, reference=reference)
    except BackendError:
        raise
    except Exception as exc:  # the SDKs raise their own hierarchies; normalize them
        raise BackendError(
            f"{what}: cannot resolve {reference!r} from its key store: {type(exc).__name__}: {exc}"
        ) from exc


def _from_vault(target: str, *, what: str, reference: str) -> str:
    """Read a Vault KV v2 key: ``vault:<mount>/data/<path>#<key>``.

    Answered with `urllib` rather than `hvac`. The KV read is one authenticated GET, and
    this module sits in the import path of every connector — a client library, its retry
    policy and its TLS stack are a large thing to link for one request.

    The token comes from ``VAULT_TOKEN`` or, on Kubernetes, from the projected service
    account token exchanged at ``auth/kubernetes/login``. The address comes from
    ``VAULT_ADDR``. Both are the variables the Vault CLI itself reads, so a machine already
    configured for ``vault kv get`` needs nothing further.
    """
    import urllib.request

    address = os.environ.get("VAULT_ADDR", "").rstrip("/")
    if not address:
        raise BackendError(
            f"{what}: {reference!r} needs VAULT_ADDR set to the Vault server address"
        )
    path, _, key = target.partition("#")
    token = _vault_token(address, what=what, reference=reference)
    request = urllib.request.Request(f"{address}/v1/{path.lstrip('/')}")
    request.add_header("X-Vault-Token", token)
    namespace = os.environ.get("VAULT_NAMESPACE", "")
    if namespace:
        request.add_header("X-Vault-Namespace", namespace)
    with urllib.request.urlopen(request, timeout=_timeout()) as response:
        document = json.loads(response.read())
    # KV v2 nests the pairs under data.data; KV v1 puts them directly under data.
    data = document.get("data", {})
    pairs = data.get("data", data)
    if not key:
        if len(pairs) == 1:
            return str(next(iter(pairs.values())))
        raise BackendError(
            f"{what}: {reference!r} names a Vault path holding {len(pairs)} keys; add "
            f"'#<key>' to say which one (available: {sorted(pairs)})"
        )
    if key not in pairs:
        raise BackendError(
            f"{what}: {reference!r} names key {key!r}, which that Vault path does not hold "
            f"(available: {sorted(pairs)})"
        )
    return str(pairs[key])


def _vault_token(address: str, *, what: str, reference: str) -> str:
    """The Vault token: the environment's, else a Kubernetes auth login.

    The Kubernetes path is what makes this work on a worker without a long-lived token:
    the projected service account token is mounted into every pod, and Vault exchanges it
    for a short-lived one scoped to that pod's identity. This is the credential-free route,
    and it is per-worker by construction.
    """
    import urllib.request

    token = os.environ.get("VAULT_TOKEN", "")
    if token:
        return token
    role = os.environ.get("VAULT_K8S_ROLE", "")
    jwt_path = os.environ.get(
        "VAULT_K8S_TOKEN_PATH", "/var/run/secrets/kubernetes.io/serviceaccount/token"
    )
    if not role or not os.path.exists(jwt_path):
        raise BackendError(
            f"{what}: {reference!r} needs either VAULT_TOKEN, or VAULT_K8S_ROLE plus a "
            f"projected service account token at {jwt_path}"
        )
    with open(jwt_path, encoding="utf-8") as handle:
        jwt = handle.read().strip()
    mount = os.environ.get("VAULT_K8S_MOUNT", "kubernetes").strip("/")
    body = json.dumps({"role": role, "jwt": jwt}).encode()
    request = urllib.request.Request(f"{address}/v1/auth/{mount}/login", data=body)
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=_timeout()) as response:
        document = json.loads(response.read())
    client_token = document.get("auth", {}).get("client_token")
    if not client_token:
        raise BackendError(f"{what}: Vault Kubernetes login for {reference!r} returned no token")
    return str(client_token)


def _from_aws_secrets_manager(target: str, *, what: str, reference: str) -> str:
    """Read an AWS Secrets Manager secret by name or ARN, honouring a ``#json-key`` suffix.

    Credentials come from the standard chain, which on a Ray worker means the instance
    profile or the IRSA role that pod holds. Nothing is passed from the driver.
    """
    boto3 = _require("boto3", scheme="aws-sm", extra="aws-secrets", what=what)
    name, _, key = target.partition("#")
    client = boto3.client("secretsmanager", region_name=os.environ.get("AWS_REGION") or None)
    response = client.get_secret_value(SecretId=name)
    secret = response.get("SecretString")
    if secret is None:
        import base64

        secret = base64.b64decode(response["SecretBinary"]).decode("utf-8")
    return _maybe_json_key(secret, key, what=what, reference=reference)


def _from_aws_parameter_store(target: str, *, what: str, reference: str) -> str:
    """Read an SSM parameter, decrypting a SecureString.

    The name is validated before the SDK is reached, because the one mistake everybody
    makes here is omitting the leading slash, and SSM answers that with a
    `ParameterNotFound` that says nothing about why.
    """
    if not target.startswith("/"):
        raise BackendError(
            f"{what}: {reference!r} names an SSM parameter without a leading slash; "
            "SSM parameter names are absolute, e.g. 'aws-ssm:/prod/warehouse/password'"
        )
    boto3 = _require("boto3", scheme="aws-ssm", extra="aws-secrets", what=what)
    client = boto3.client("ssm", region_name=os.environ.get("AWS_REGION") or None)
    response = client.get_parameter(Name=target, WithDecryption=True)
    return str(response["Parameter"]["Value"])


def _from_gcp_secret_manager(target: str, *, what: str, reference: str) -> str:
    """Read a GCP Secret Manager version.

    `target` is the full resource name. A bare ``projects/p/secrets/s`` is completed with
    ``/versions/latest``, because naming a secret and meaning its current version is what
    almost every caller wants and forgetting the suffix is otherwise a confusing 404. The
    shape is checked before the SDK is imported, for the reason given in the Azure handler.
    """
    if not target.startswith("projects/"):
        raise BackendError(
            f"{what}: {reference!r} must name a Secret Manager resource, e.g. "
            "'gcp-sm:projects/my-project/secrets/db-password'"
        )
    module = _require("google.cloud.secretmanager", scheme="gcp-sm", extra="gcp-secrets", what=what)
    name = target if "/versions/" in target else f"{target.rstrip('/')}/versions/latest"
    client = module.SecretManagerServiceClient()
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8")


def _from_azure_key_vault(target: str, *, what: str, reference: str) -> str:
    """Read an Azure Key Vault secret from its full URL.

    Authenticates with `DefaultAzureCredential`, which resolves a managed identity on a VM
    or a workload identity in AKS — again, the worker's own identity rather than the
    driver's.
    """
    # The reference is validated before the SDK is imported. A malformed reference is
    # malformed whether or not azure-keyvault is installed, and being told to pip-install
    # something sends the reader to fix the wrong thing.
    url = target.rstrip("/")
    marker = "/secrets/"
    if marker not in url:
        raise BackendError(
            f"{what}: {reference!r} must be a full Key Vault secret URL, e.g. "
            "azure-kv:https://myvault.vault.azure.net/secrets/db-password"
        )
    vault_url, _, name = url.partition(marker)
    identity = _require("azure.identity", scheme="azure-kv", extra="azure-secrets", what=what)
    secrets = _require(
        "azure.keyvault.secrets", scheme="azure-kv", extra="azure-secrets", what=what
    )
    client = secrets.SecretClient(vault_url=vault_url, credential=identity.DefaultAzureCredential())
    return str(client.get_secret(name.split("/")[0]).value)


def _maybe_json_key(secret: str, key: str, *, what: str, reference: str) -> str:
    """Return `secret`, or the `key` field of it when the secret holds a JSON object.

    Secrets Manager's own console writes a JSON object for anything with more than one
    field, which is how a "password" ends up as ``{"username": ..., "password": ...}``. A
    reference that asked for a password and got that whole document back would be a very
    confusing connection failure, so the ``#key`` suffix selects a field.
    """
    if not key:
        return secret
    try:
        document = json.loads(secret)
    except ValueError as exc:
        raise BackendError(
            f"{what}: {reference!r} asks for key {key!r}, but that secret is not JSON"
        ) from exc
    if key not in document:
        raise BackendError(
            f"{what}: {reference!r} asks for key {key!r}, which that secret does not hold "
            f"(available: {sorted(document)})"
        )
    return str(document[key])


def _require(module: str, *, scheme: str, extra: str, what: str) -> Any:
    """Import `module` or raise a `BackendError` naming the extra that provides it."""
    from importlib import import_module

    try:
        return import_module(module)
    except ImportError as exc:
        raise BackendError(
            f"{what}: the '{scheme}:' scheme needs {module!r}; install it with "
            f"`pip install 'batcher-engine[{extra}]'`"
        ) from exc


def _timeout() -> float:
    """Per-request timeout for the HTTP-answered stores."""
    try:
        return float(os.environ.get("BATCHER_SECRET_TIMEOUT_SECONDS", "10"))
    except ValueError:
        return 10.0
