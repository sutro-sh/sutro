import base64
import hashlib
from unittest.mock import Mock

import pytest

from sutro.assets import Asset, Image, prepare_batch_image_assets
from sutro.sdk import Sutro


def test_repeated_images_are_attached_once_without_mutating_rows():
    image = Image.from_bytes(b"image-fixture", "image/png")
    rows = [{"question": "Describe", "scan": image}, {"scan": image}]
    prepared, attachments = prepare_batch_image_assets(rows)
    digest = hashlib.sha256(b"image-fixture").hexdigest()
    assert prepared == [
        {"question": "Describe", "scan": {"asset_id": digest}},
        {"scan": {"asset_id": digest}},
    ]
    assert attachments == {
        digest: "data:image/png;base64," + base64.b64encode(b"image-fixture").decode()
    }
    assert rows[0]["scan"] is image
    assert "base64" in image


def test_pdf_bytes_and_path_are_deduplicated_without_mutating_rows(tmp_path):
    pdf_bytes = b"%PDF-1.4\npdf-fixture"
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(pdf_bytes)
    inline_pdf = Asset.from_bytes(pdf_bytes, "application/pdf", filename="copy.pdf")
    path_pdf = Asset.from_path(pdf_path)
    rows = [{"scan": inline_pdf}, {"scan": path_pdf}]

    prepared, attachments = prepare_batch_image_assets(rows)

    digest = hashlib.sha256(pdf_bytes).hexdigest()
    assert prepared == [{"scan": {"asset_id": digest}}] * 2
    assert attachments == {
        digest: "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode()
    }
    assert rows[0]["scan"] is inline_pdf
    assert rows[1]["scan"] is path_pdf
    assert inline_pdf["filename"] == "copy.pdf"
    assert path_pdf["filename"] == "scan.pdf"
    assert "base64" in inline_pdf and "base64" in path_pdf


def test_mixed_media_keeps_order_and_does_not_rewrite_plain_pdf_json():
    image = Image.from_bytes(b"image-fixture", "image/png")
    pdf = Asset.from_bytes(b"%PDF-1.4\npdf-fixture", "application/pdf")
    plain_pdf_json = dict(pdf)
    rows = [{"media": [image, pdf, {"record": plain_pdf_json}, pdf]}]

    prepared, attachments = prepare_batch_image_assets(rows)

    image_digest = hashlib.sha256(b"image-fixture").hexdigest()
    pdf_digest = hashlib.sha256(b"%PDF-1.4\npdf-fixture").hexdigest()
    assert prepared == [
        {
            "media": [
                {"asset_id": image_digest},
                {"asset_id": pdf_digest},
                {"record": plain_pdf_json},
                {"asset_id": pdf_digest},
            ]
        }
    ]
    assert list(attachments) == [image_digest, pdf_digest]
    assert attachments[pdf_digest].startswith("data:application/pdf;base64,")


@pytest.mark.parametrize(
    "media_kind,mime_type,asset_bytes",
    [
        ("image", "image/png", b"image-fixture"),
        ("pdf", "application/pdf", b"%PDF-1.4\npdf-fixture"),
    ],
)
def test_base_model_envelope_keeps_media_order_and_url_sources(
    media_kind, mime_type, asset_bytes
):
    asset = Asset.from_bytes(asset_bytes, mime_type)
    rows = [
        {
            "text": "Compare",
            "media": [
                {"type": media_kind, "field": "scan", "index": 0, media_kind: asset},
                {
                    "type": media_kind,
                    "field": "prior",
                    "index": 1,
                    media_kind: Asset.from_url("https://assets.example/prior"),
                },
            ],
        }
    ]
    prepared, attachments = prepare_batch_image_assets(rows)
    digest = hashlib.sha256(asset_bytes).hexdigest()
    assert list(attachments) == [digest]
    assert prepared[0]["media"][0] == {
        "type": media_kind,
        "field": "scan",
        "index": 0,
        media_kind: {"asset_id": digest},
    }
    assert prepared[0]["media"][1] == rows[0]["media"][1]


def test_table_download_sources_are_unchanged():
    assert prepare_batch_image_assets("https://data.example/rows.parquet") == (
        "https://data.example/rows.parquet",
        {},
    )


@pytest.mark.parametrize(
    "media_kind,mime_type",
    [("image", "image/png"), ("pdf", "application/pdf")],
)
@pytest.mark.parametrize(
    "encoded",
    [
        "this is customer text, not an image",
        base64.b64encode(b"customer bytes").decode(),
    ],
)
def test_plain_customer_json_is_not_an_implicit_asset_upload(
    encoded, media_kind, mime_type
):
    value = {
        "type": media_kind,
        "base64": encoded,
        "mime_type": mime_type,
    }
    rows = [{"record": value}, {"record": {"nested": value}}]
    client = Sutro.__new__(Sutro)
    job = Mock(status_code=200)
    job.json.return_value = {"results": "job-text"}
    client.do_request = Mock(return_value=job)

    result = client._run_one_batch_inference(
        rows,
        "text-function",
        column=["record"],
        output_column="output",
        job_priority=0,
        json_schema=None,
        sampling_params=None,
        system_prompt=None,
        cost_estimate=False,
        stay_attached=False,
        random_seed_per_input=False,
        truncate_rows=False,
        name=None,
        description=None,
        id_column=None,
    )

    assert result == "job-text"
    client.do_request.assert_called_once()
    assert client.do_request.call_args.args[1] == "batch-inference"
    assert client.do_request.call_args.kwargs["json"]["inputs"] == rows
    assert "image_assets" not in client.do_request.call_args.kwargs["json"]


@pytest.mark.parametrize(
    "asset_bytes,mime_type",
    [(b"image-fixture", "image/png"), (b"%PDF-1.4\npdf-fixture", "application/pdf")],
)
def test_batch_submission_uploads_unique_asset_then_sends_receipt(asset_bytes, mime_type):
    asset = Asset.from_bytes(asset_bytes, mime_type)
    digest = hashlib.sha256(asset_bytes).hexdigest()
    upload = Mock(status_code=200)
    upload.json.return_value = {"asset_id": digest, "receipt": {"token": "signed"}}
    job = Mock(status_code=200)
    job.json.return_value = {"results": "job-1"}
    client = Sutro.__new__(Sutro)
    client.do_request = Mock(side_effect=[upload, job])
    result = client._run_one_batch_inference(
        [{"scan": asset}, {"scan": asset}],
        "media-function",
        column=["scan"],
        output_column="output",
        job_priority=0,
        json_schema=None,
        sampling_params=None,
        system_prompt=None,
        cost_estimate=False,
        stay_attached=False,
        random_seed_per_input=False,
        truncate_rows=False,
        name=None,
        description=None,
        id_column=None,
    )
    assert result == "job-1"
    assert [call.args[1] for call in client.do_request.call_args_list] == [
        "image-assets",
        "batch-inference",
    ]
    assert client.do_request.call_args_list[0].kwargs["json"] == {
        "data": f"data:{mime_type};base64," + base64.b64encode(asset_bytes).decode()
    }
    payload = client.do_request.call_args_list[1].kwargs["json"]
    assert payload["image_assets"] == {digest: {"token": "signed"}}
    assert payload["inputs"] == [
        {"scan": {"asset_id": digest}},
        {"scan": {"asset_id": digest}},
    ]
