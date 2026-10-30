import base64
import hashlib
from unittest.mock import Mock

import pytest

from sutro.assets import Image, prepare_batch_image_assets
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


def test_base_model_envelope_keeps_image_order_and_url_sources():
    image = Image.from_bytes(b"image-fixture", "image/png")
    rows = [
        {
            "text": "Compare",
            "media": [
                {"type": "image", "field": "scan", "index": 0, "image": image},
                {
                    "type": "image",
                    "field": "prior",
                    "index": 1,
                    "image": Image.from_url("https://images.example/prior.png"),
                },
            ],
        }
    ]
    prepared, attachments = prepare_batch_image_assets(rows)
    assert list(attachments) == [prepared[0]["media"][0]["image"]["asset_id"]]
    assert prepared[0]["media"][1] == rows[0]["media"][1]


def test_table_download_sources_are_unchanged():
    assert prepare_batch_image_assets("https://data.example/rows.parquet") == (
        "https://data.example/rows.parquet",
        {},
    )


@pytest.mark.parametrize(
    "encoded",
    [
        "this is customer text, not an image",
        base64.b64encode(b"customer bytes").decode(),
    ],
)
def test_plain_customer_json_is_not_an_implicit_image_upload(encoded):
    value = {
        "type": "image",
        "base64": encoded,
        "mime_type": "image/png",
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


def test_batch_submission_uploads_unique_image_then_sends_receipt():
    image = Image.from_bytes(b"image-fixture", "image/png")
    digest = hashlib.sha256(b"image-fixture").hexdigest()
    upload = Mock(status_code=200)
    upload.json.return_value = {"asset_id": digest, "receipt": {"token": "signed"}}
    job = Mock(status_code=200)
    job.json.return_value = {"results": "job-1"}
    client = Sutro.__new__(Sutro)
    client.do_request = Mock(side_effect=[upload, job])
    result = client._run_one_batch_inference(
        [{"scan": image}, {"scan": image}],
        "image-function",
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
    payload = client.do_request.call_args_list[1].kwargs["json"]
    assert payload["image_assets"] == {digest: {"token": "signed"}}
    assert payload["inputs"] == [
        {"scan": {"asset_id": digest}},
        {"scan": {"asset_id": digest}},
    ]
