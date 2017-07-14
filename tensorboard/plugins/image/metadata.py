# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Internal information about the images plugin."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json

from tensorflow.core.framework import summary_pb2


# The plugin name, as it appears in the `SummaryMetadata`.
PLUGIN_NAME = 'images'

# The basename of each actual image summary op. When using
# `summaries.op`, we create a `name_scope` with the user-provided name,
# in which we do some computation and then yield to the summary op of
# this name.
#
# TODO(@wchargin,@dandelionmane): Once `display_name` support is added
# to summaries, drop this hack and replace it.
SUMMARY_OP_NAME = '__internal_summary'

# We don't currently need to store any metadata.
ImagePluginMetadata = collections.namedtuple('ImagePluginMetadata', ())


def create_summary_metadata():
  """Create a `SummaryMetadata` proto for an image summary."""
  plugin_metadata = ImagePluginMetadata()
  content = json.dumps(plugin_metadata._asdict(), sort_keys=True)  # pylint: disable=protected-access
  summary_metadata = summary_pb2.SummaryMetadata()
  summary_metadata.plugin_data.add(plugin_name=PLUGIN_NAME, content=content)
  return summary_metadata


def parse_plugin_metadata(content):
  """Parse summary metadata to a Python object.

  Arguments:
    content: the `content` field of a `SummaryMetadata` proto
      corresponding to the image plugin

  Returns:
    an `ImagePluginMetadata` instance
  """
  return ImagePluginMetadata(**json.loads(content))
