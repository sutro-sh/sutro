"""Asset values for Function input fields.

A Function whose inputs include an image or a PDF expects those fields to
carry an asset value rather than a string. The helpers here build the exact
shapes a Sutro deployment accepts:

* inline bytes, base64-encoded, with their MIME type,
* an ``https://`` URL the deployment fetches, or
* the name of an asset already stored in the deployment.

Every helper returns a plain ``dict`` subclass, so the value can be dropped
straight into the ``input`` mapping of :meth:`sutro.Sutro.run_function` and
serialized with ``json.dumps``.
"""

import base64
import mimetypes
import os
from typing import Any, Dict, Optional, Union

IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
PDF_MIME_TYPE = "application/pdf"
SUPPORTED_MIME_TYPES = frozenset(IMAGE_MIME_TYPES | {PDF_MIME_TYPE})

# Deployments sniff the bytes and reject anything outside this set, so
# guessing locally only ever produces a clearer error, never a looser one.
_MIME_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".pdf": PDF_MIME_TYPE,
}

_MEDIA_KIND_BY_MIME_TYPE = {
    "image/png": "image",
    "image/jpeg": "image",
    "image/webp": "image",
    PDF_MIME_TYPE: "pdf",
}


def _normalize_mime_type(mime_type: str) -> str:
    normalized = str(mime_type or "").strip().lower()
    # "image/jpg" is common in the wild and is not a registered type.
    return "image/jpeg" if normalized == "image/jpg" else normalized


def _guess_mime_type(path: Union[str, "os.PathLike[str]"]) -> str:
    """Guess an asset's MIME type from its file extension."""
    guessed, _ = mimetypes.guess_type(os.fspath(path))
    mime_type = _normalize_mime_type(guessed or "")
    if mime_type in SUPPORTED_MIME_TYPES:
        return mime_type
    extension = os.path.splitext(os.fspath(path))[1].lower()
    fallback = _MIME_BY_EXTENSION.get(extension)
    if fallback is not None:
        return fallback
    raise ValueError(
        f"Could not determine the MIME type of '{os.fspath(path)}' from its "
        "extension. Pass the bytes and their type to from_bytes() instead."
    )


class Asset(dict):
    """An image or PDF value for a Function input field.

    Use :class:`Image` when the field only ever holds an image; it rejects a
    PDF locally instead of at the deployment.
    """

    # Subclasses narrow what they accept and declare a type on the wire.
    _MEDIA_KIND: Optional[str] = None
    _ALLOWED_MIME_TYPES = SUPPORTED_MIME_TYPES

    @classmethod
    def _check_mime_type(cls, mime_type: str) -> str:
        normalized = _normalize_mime_type(mime_type)
        if normalized not in cls._ALLOWED_MIME_TYPES:
            raise ValueError(
                f"Unsupported MIME type for {cls.__name__}: "
                f"'{mime_type}'. Supported: "
                f"{', '.join(sorted(cls._ALLOWED_MIME_TYPES))}."
            )
        return normalized

    @classmethod
    def _declared_type(cls, mime_type: Optional[str]) -> Optional[str]:
        if cls._MEDIA_KIND is not None:
            return cls._MEDIA_KIND
        if mime_type is None:
            return None
        return _MEDIA_KIND_BY_MIME_TYPE.get(mime_type)

    @classmethod
    def _build(cls, mime_type: Optional[str], **fields: Any) -> "Asset":
        payload: Dict[str, Any] = {}
        declared_type = cls._declared_type(mime_type)
        if declared_type is not None:
            payload["type"] = declared_type
        payload.update({k: v for k, v in fields.items() if v is not None})
        if mime_type is not None:
            payload["mime_type"] = mime_type
        return cls(payload)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        mime_type: str,
        filename: Optional[str] = None,
    ) -> "Asset":
        """Build an asset from raw bytes.

        Args:
            data (bytes): The asset's bytes.
            mime_type (str): The asset's MIME type, for example ``image/png``.
            filename (str, optional): A display name for the asset. Its
                extension must match ``mime_type``. Defaults to None.

        Returns:
            Asset: A base64 asset value.
        """
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("Asset bytes must be bytes-like.")
        data = bytes(data)
        if not data:
            raise ValueError("Asset bytes must not be empty.")
        normalized = cls._check_mime_type(mime_type)
        return cls._build(
            normalized,
            base64=base64.b64encode(data).decode("ascii"),
            filename=os.path.basename(filename) if filename else None,
        )

    @classmethod
    def from_path(cls, path: Union[str, "os.PathLike[str]"]) -> "Asset":
        """Build an asset by reading a local file.

        The MIME type is taken from the file extension.

        Args:
            path (str | os.PathLike): Path to a PNG, JPEG, WebP, or PDF file.

        Returns:
            Asset: A base64 asset value.
        """
        mime_type = cls._check_mime_type(_guess_mime_type(path))
        with open(path, "rb") as handle:
            data = handle.read()
        return cls.from_bytes(data, mime_type, filename=os.path.basename(path))

    @classmethod
    def from_url(cls, url: str) -> "Asset":
        """Reference an asset the deployment downloads over HTTPS.

        Args:
            url (str): An ``https://`` URL. Deployments refuse other schemes.

        Returns:
            Asset: A URL asset value.
        """
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("Asset URLs must be absolute https:// URLs.")
        return cls._build(None, url=url)

    @classmethod
    def from_name(cls, name: str) -> "Asset":
        """Reference an asset already stored in the deployment.

        Args:
            name (str): The asset's name in the deployment's asset store.

        Returns:
            Asset: A stored-asset value.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Asset names must be non-empty strings.")
        return cls._build(None, name=name.strip())


class Image(Asset):
    """An image value for a Function input field."""

    _MEDIA_KIND = "image"
    _ALLOWED_MIME_TYPES = IMAGE_MIME_TYPES
