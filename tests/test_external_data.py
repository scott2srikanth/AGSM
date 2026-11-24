import torch

from data.external import ListOps, WikiText103
from data.ruler import read_ruler


def test_wikitext_byte_targets(tmp_path):
    path = tmp_path / "wiki.raw"
    path.write_bytes(b"abcde")
    data = WikiText103(path, "train", 2, 10, "fixture")
    assert len(data) == 2
    tokens, targets = data[1]
    torch.testing.assert_close(tokens, torch.tensor([99, 100]))
    torch.testing.assert_close(targets, torch.tensor([100, 101]))


def test_listops_padding_and_target(tmp_path):
    path = tmp_path / "sample.tsv"
    path.write_text("Source\tTarget\n[MAX 2 1 ]\t2\n")
    data = ListOps(path, "test", 6, 10, "fixture")
    tokens, targets = data[0]
    assert tokens.tolist() == [0, 0, 11, 3, 2, 15]
    assert targets.tolist() == [-100, -100, -100, -100, -100, 2]


def test_ruler_jsonl(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text('{"index": 0, "input": "prompt", "outputs": ["answer"]}\n')
    assert list(read_ruler(path))[0]["outputs"] == ["answer"]
