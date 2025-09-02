from utilities.tensorflow_config import tf_compile
import tensorflow as tf

#------------------------------------------------
#  Methods for optimizer
#------------------------------------------------

@tf_compile
def pack(x_list):
    """
    convert a list of tensors to a 1D tensor
    """
    return tf.concat([tf.reshape(v, [-1]) for v in x_list], axis=0)


@tf_compile
def unpack_like(x, template_list):
    """
        convert a 1D tensor to a list of tensors with the shape of the template list of tensors
    """
    tensor_list, offset = [], 0
    for v in template_list:
        size = tf.size(v)
        part = tf.reshape(x[offset: offset + size], tf.shape(v))
        tensor_list.append(part)
        offset += size
    return tensor_list

@tf_compile
def add(x_list, y):
    """
    convert a 1D tensor y to a list of tensors and add them to the tensors of x
    """
    for v, d in zip(x_list, unpack_like(y, x_list)):
        v.assign_add(d)