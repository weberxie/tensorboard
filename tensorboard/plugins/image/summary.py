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
"""Image summaries and TensorFlow operations to create them.

An image summary stores the width, height, and PNG-encoded data for zero
or more images in a rank-1 string array: `[w, h, png0, png1, ...]`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorboard.plugins.image import metadata


def op(name, images, max_outputs=3, collections=None):
  """Create an image summary op for use in a TensorFlow graph.

  Arguments:
    name: a fixed name for the summary operation (as a `str`).
    images: A `Tensor` representing pixel data with shape `[k, w, h, c]`,
      where `k` is the number of images, `w` and `h` are the width and
      height of the images, and `c` is the number of channels, which
      should be 1, 3, or 4. Any of the dimensions may be statically
      unknown (i.e., `None`).
    max_outputs: `int` or rank-0 integer `Tensor`. At most this many
      images will be emitted at each step. When more than `max_outputs`
      many images are provided, the first `max_outputs` many images will
      be used and the rest discarded.
    collections: Optional list of graph collections keys. The new
      summary op is added to these collections. Defaults to
      `["summaries"]`.
  """
  images.shape.assert_has_rank(4)

  with tf.name_scope(name), \
       tf.control_dependencies([tf.assert_non_negative(max_outputs)]):
    limited_images = images[:max_outputs]
    encoded_images = tf.map_fn(tf.image.encode_png, limited_images,
                               dtype=tf.string,
                               name='encode_each_image')
    image_shape = tf.shape(images)
    dimensions = tf.stack([tf.as_string(image_shape[1], name='width'),
                           tf.as_string(image_shape[2], name='height')],
                          name='dimensions')
    output = tf.concat([dimensions, encoded_images], axis=0)
    summary_metadata = metadata.create_summary_metadata()
    return tf.summary.tensor_summary(name=metadata.SUMMARY_OP_NAME,
                                     tensor=output,
                                     collections=collections,
                                     summary_metadata=summary_metadata)


def pb(name, images, max_outputs=3):
  """Create an image summary for the given data.

  This behaves as if you were to create an `op` with the same arguments
  (wrapped with constant tensors wher appropriate) and then execute that
  summary op in a TensorFlow session.

  Arguments:
    name: a `str`, as described in `op`
    images: as described in `op`, but as a numpy array instead of
      a `Tensor`
    max_outputs: as described in `op`, but as an `int` instead of
      a `Tensor`

  Returns:
    a `Summary` protocol buffer
  """
  if images.ndim != 4:
    raise ValueError('Shape %r must have rank 4' % (images.shape, ))
  limited_images = images[:max_outputs]
  encoded_images = [_encode_png(image) for image in limited_images]
  (width, height) = images.shape[1:3]
  content = [str(width), str(height)] + encoded_images
  tensor = tf.make_tensor_proto(content, dtype=tf.string)
  summary = tf.Summary()
  summary.value.add(tag='%s/%s' % (name, metadata.SUMMARY_OP_NAME),
                    metadata=metadata.create_summary_metadata(),
                    tensor=tensor)
  return summary


def _encode_png(image):
  # TODO(@wchargin): Pick a third-party PNG backend and implement this.
  raise NotImplementedError('Encoding PNGs in Python is not yet supported.')
