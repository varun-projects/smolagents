# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import tempfile
import unittest
import uuid

import PIL.Image

from smolagents.agent_types import AgentAudio, AgentImage, AgentText

from .utils.markers import require_soundfile, require_torch


def get_new_path(suffix="") -> str:
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)


@require_soundfile
@require_torch
class AgentAudioTests(unittest.TestCase):
    def test_from_tensor(self):
        import soundfile as sf
        import torch

        tensor = torch.rand(12, dtype=torch.float64) - 0.5
        agent_type = AgentAudio(tensor)
        path = str(agent_type.to_string())

        # Ensure that the tensor and the agent_type's tensor are the same
        self.assertTrue(torch.allclose(tensor, agent_type.to_raw(), atol=1e-4))

        del agent_type

        # Ensure the path remains even after the object deletion
        self.assertTrue(os.path.exists(path))

        # Ensure that the file contains the same value as the original tensor
        new_tensor, _ = sf.read(path)
        self.assertTrue(torch.allclose(tensor, torch.tensor(new_tensor), atol=1e-4))

    def test_from_string(self):
        import soundfile as sf
        import torch

        tensor = torch.rand(12, dtype=torch.float64) - 0.5
        path = get_new_path(suffix=".wav")
        sf.write(path, tensor, 16000)

        agent_type = AgentAudio(path)

        self.assertTrue(torch.allclose(tensor, agent_type.to_raw(), atol=1e-4))
        self.assertEqual(agent_type.to_string(), path)


@require_torch
class TestAgentImage:
    def test_from_tensor(self):
        import torch

        tensor = torch.randint(0, 256, (64, 64, 3))
        agent_type = AgentImage(tensor)
        path = str(agent_type.to_string())

        # Ensure that the tensor and the agent_type's tensor are the same
        assert torch.allclose(tensor, agent_type._tensor, atol=1e-4)

        assert isinstance(agent_type.to_raw(), PIL.Image.Image)

        # Ensure the path remains even after the object deletion
        del agent_type
        assert os.path.exists(path)

    def test_from_string(self, shared_datadir):
        path = shared_datadir / "000000039769.png"
        image = PIL.Image.open(path)
        agent_type = AgentImage(path)

        assert path.samefile(agent_type.to_string())
        assert image == agent_type.to_raw()

        # Ensure the path remains even after the object deletion
        del agent_type
        assert os.path.exists(path)

    def test_from_image(self, shared_datadir):
        path = shared_datadir / "000000039769.png"
        image = PIL.Image.open(path)
        agent_type = AgentImage(image)

        assert not path.samefile(agent_type.to_string())
        assert image == agent_type.to_raw()

        # Ensure the path remains even after the object deletion
        del agent_type
        assert os.path.exists(path)

    def test_from_numpy_without_torch(self):
        """Regression for #2657: AgentImage must accept a numpy array even when torch is absent.

        Before the fix, the try block importing torch caught ModuleNotFoundError and
        skipped the numpy branch entirely, raising TypeError.

        This test patches out torch to simulate an environment without it.
        """
        import sys
        import unittest.mock

        import numpy as np

        # Simulate torch not being installed by hiding it from imports
        with unittest.mock.patch.dict(sys.modules, {"torch": None}):
            array = np.zeros((4, 4, 3), dtype=np.uint8)
            # Must not raise TypeError
            img = AgentImage(array)
            # Must be usable
            raw = img.to_raw()
            assert isinstance(raw, PIL.Image.Image)
            assert raw.size == (4, 4)


class AgentTextTests(unittest.TestCase):
    def test_from_string(self):
        string = "Hey!"
        agent_type = AgentText(string)

        self.assertEqual(string, agent_type.to_string())
        self.assertEqual(string, agent_type.to_raw())
