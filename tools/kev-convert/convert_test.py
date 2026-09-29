import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_convert():
    sys.modules.setdefault("torch", types.SimpleNamespace())
    sys.modules.setdefault("huggingface_hub", types.SimpleNamespace(snapshot_download=lambda *_, **__: None))
    module_path = Path(__file__).with_name("convert.py")
    spec = importlib.util.spec_from_file_location("convert", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


convert = load_convert()


class MetadataTest(unittest.TestCase):
    def test_split_hub_run_defaults_to_main(self):
        self.assertEqual(convert.split_hub_run("jaredpalmer/kev-4b"), ("jaredpalmer/kev-4b", "main"))

    def test_split_hub_run_keeps_explicit_revision(self):
        self.assertEqual(
            convert.split_hub_run("jaredpalmer/kev-4b@night2-du-release"),
            ("jaredpalmer/kev-4b", "night2-du-release"),
        )

    def test_resolved_snapshot_commit_reads_hub_cache_path(self):
        self.assertEqual(
            convert.resolved_snapshot_commit(Path("/tmp/hub/models--jaredpalmer--kev-4b/snapshots/abc123")),
            "abc123",
        )

    def test_resolved_snapshot_commit_ignores_local_paths(self):
        self.assertIsNone(convert.resolved_snapshot_commit(Path("/tmp/kev-4b")))


if __name__ == "__main__":
    unittest.main()
