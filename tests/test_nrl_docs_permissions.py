"""Validate required permissions for the NRL docs publish workflow."""
import os
import re
import unittest


class TestNrlDocsPermissions(unittest.TestCase):
    def test_publish_workflow_has_actions_write(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(repo_root, ".github", "workflows", "nrl-docs-nvidia-publish.yml")
        with open(path, encoding="utf-8") as f:
            workflow = f.read()
        perm_block = re.search(r"^permissions:\n((?:[ ]{2,}\S.*\n)*)", workflow, re.MULTILINE)
        self.assertIsNotNone(perm_block)
        self.assertIn("actions: write", perm_block.group(0))
        self.assertIn("contents: read", perm_block.group(0))

