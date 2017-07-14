# -*- coding: utf-8 -*-
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
"""Tests for the image plugin summary generation functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import random

import numpy as np
import six
import tensorflow as tf

from tensorboard.plugins.image import summary


class SummaryTest(tf.test.TestCase):

  def setUp(self):
    super(SummaryTest, self).setUp()
    tf.reset_default_graph()
    self.stubs = tf.test.StubOutForTesting()
    self.stubs.Set(summary, '_encode_png', self.stub_encode_png)
    self.stubs.Set(tf.image, 'encode_png', self.stub_tf_encode_png)

    rnd = random.Random(0)
    self.image_width = 300
    self.image_height = 75
    self.images = self.generate_images(
      rnd, 8, self.image_width, self.image_height, alpha=False)
    self.images_with_alpha = self.generate_images(rnd, 8, 300, 75, alpha=True)

  def generate_images(self, rnd, count, width, height, alpha=False):
    channels = 'RGBA' if alpha else 'RGB'
    return np.array([[[[rnd.randint(0, 255) for _ in channels]
                       for row in xrange(height)]
                      for column in xrange(width)]
                     for image in xrange(count)])

  def tearDown(self):
    self.stubs.CleanUp()
    super(SummaryTest, self).tearDown()

  def stub_encode_png(self, data_array):
    data = str(data_array)
    hashed = hashlib.sha256(data).hexdigest()
    prefix = data[:64]
    suffix = data[-64:]
    digest = '%s...[%s]...%s' % (prefix, hashed, suffix)
    return 'shape:%r;digest:%s' % (data_array.shape, data_array)

  def stub_tf_encode_png(self, data_tensor):
    f = tf.py_func(self.stub_encode_png,
                   [data_tensor],
                   Tout=tf.string,
                   stateful=False)
    # The shape needs to be statically known. In the real code, it's
    # correctly inferred, but we have to set it manually here because we
    # use a `py_func`.
    f.set_shape([])  # string scalar (rank-0)
    return f

  def pb_via_op(self, summary_op, feed_dict=None):
    actual_pbtxt = tf.Session().run(summary_op, feed_dict=feed_dict or {})
    actual_proto = tf.Summary()
    actual_proto.ParseFromString(actual_pbtxt)
    return actual_proto


  def get_summary_pb(self, name, images, max_outputs=3,
                     images_tensor=None, feed_dict=None):
    """Use both `op` and `pb` to get a summary, asserting equality.

    Arguments:
      images: a numpy array
      images_tensor: defaults to `tf.constant(images)`

    Returns:
      a `Summary` protocol buffer
    """
    if images_tensor is None:
      images_tensor = tf.constant(images)
    op = summary.op(name, images_tensor, max_outputs=max_outputs)
    pb = summary.pb(name, images, max_outputs=max_outputs)
    pb_via_op = self.pb_via_op(op, feed_dict=feed_dict)
    self.assertProtoEquals(pb, pb_via_op)
    return pb

  def test_correctly_handles_no_images(self):
    shape = (0, self.image_width, self.image_height, 3)
    images = np.array([]).reshape(shape)
    pb = self.get_summary_pb('mona_lisa', images, max_outputs=3)
    self.assertEqual(1, len(pb.value))
    result = pb.value[0].tensor.string_val
    self.assertEqual(str(self.image_width), result[0])
    self.assertEqual(str(self.image_height), result[1])
    image_results = result[2:]
    self.assertEqual(len(image_results), 0)

  def test_image_count_when_fewer_than_max(self):
    max_outputs = len(self.images) - 3
    assert max_outputs > 0, max_outputs
    pb = self.get_summary_pb('mona_lisa', self.images, max_outputs=max_outputs)
    self.assertEqual(1, len(pb.value))
    result = pb.value[0].tensor.string_val
    image_results = result[2:]  # skip width, height
    self.assertEqual(len(image_results), max_outputs)

  def test_image_count_when_more_than_max(self):
    max_outputs = len(self.images) + 3
    pb = self.get_summary_pb('mona_lisa', self.images, max_outputs=max_outputs)
    self.assertEqual(1, len(pb.value))
    result = pb.value[0].tensor.string_val
    image_results = result[2:]  # skip width, height
    self.assertEqual(len(image_results), len(self.images))

  def _test_dimensions(self, alpha=False, static_dimensions=True):
    if not alpha:
      images = self.images
      channel_count = 3
    else:
      images = self.images_with_alpha
      channel_count = 4

    if static_dimensions:
      images_tensor = tf.constant(images, dtype=tf.uint8)
      feed_dict = {}
    else:
      images_tensor = tf.placeholder(tf.uint8)
      feed_dict = {images_tensor: images}

    pb = self.get_summary_pb('mona_lisa', images,
                             images_tensor=images_tensor, feed_dict=feed_dict)
    self.assertEqual(1, len(pb.value))
    result = pb.value[0].tensor.string_val

    # Check annotated dimensions.
    self.assertEqual(str(self.image_width), result[0])
    self.assertEqual(str(self.image_height), result[1])

    # Check fake PNG data (verifying that the image was passed to the
    # encoder correctly).
    images = result[2:]
    shape = (self.image_width, self.image_height, channel_count)
    shape_tag = 'shape:%r;' % (shape, )
    for image in images:
      self.assertTrue(
        image.startswith(shape_tag),
        'expected fake image data to start with %r, but found: %r'
        % (shape_tag, image[:len(shape_tag) * 2]))

  def test_dimensions(self):
    self._test_dimensions(alpha=False)

  def test_dimensions_with_alpha(self):
    self._test_dimensions(alpha=True)

  def test_dimensions_when_not_statically_known(self):
    self._test_dimensions(alpha=False, static_dimensions=False)

  def test_dimensions_with_alpha_when_not_statically_known(self):
    self._test_dimensions(alpha=True, static_dimensions=False)

  def test_requires_rank_4_in_op(self):
    with six.assertRaisesRegex(self, ValueError, 'must have rank 4'):
      summary.op('mona_lisa', tf.constant([[1, 2, 3], [4, 5, 6]]))

  def test_requires_rank_4_in_pb(self):
    with six.assertRaisesRegex(self, ValueError, 'must have rank 4'):
      summary.pb('mona_lisa', np.array([[1, 2, 3], [4, 5, 6]]))


if __name__ == '__main__':
  tf.test.main()
