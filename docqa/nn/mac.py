from typing import Optional

import tensorflow as tf

from docqa.nn.layers import AttentionMapper, MergeLayer, SequenceEncoder, get_keras_initialization, SequenceMapper, \
    Mapper, SequenceMultiEncoder, FullyConnected, VariationalDropoutLayer, DropoutLayer
from docqa.nn.ops import VERY_NEGATIVE_NUMBER, exp_mask
from docqa.nn.attention import CtrlBiAttention
from docqa.nn.recurrent_layers import CudnnGru
from docqa.nn.similarity_layers import SimilarityFunction, compute_attention_mask, TriLinear
from tensorflow.contrib.keras.python.keras.initializers import TruncatedNormal

"""
A basic modified MAC Network addition, designed to slot into the best DocQA Model.
"""
class Mac():
    def __init__(self, hidden_dim):
        # control
        self.control_lin = FullyConnected(hidden_dim)
        self.attn = FullyConnected(1)
        # read
        self.mem_drop = DropoutLayer(0.85)
        self.read_drop = DropoutLayer(0.85)
        self.mem_proj = FullyConnected(hidden_dim)
        self.kb_proj = FullyConnected(hidden_dim)
        self.concat = FullyConnected(hidden_dim)
        self.concat2 = FullyConnected(hidden_dim)
        self.bi = CtrlBiAttention(TriLinear(), True)
        self.lin = FullyConnected(hidden_dim)
        self.read_drop = DropoutLayer(0.85)
        self.rattn = FullyConnected(1)
        # write
        self.write = FullyConnected(hidden_dim)
        self.gate = FullyConnected(1)

    def apply(self, is_train, document, question_words, question_vec, prev_cont, position_aware_cont, prev_mem, reuse, document_mask=None, question_mask=None):
        # control unit
        with tf.variable_scope("control", reuse=reuse):
            control = tf.concat([prev_cont, position_aware_cont], axis=1)
            control_question = self.control_lin.apply(is_train, control)
            control_question = tf.expand_dims(control_question, axis=1)
            context_prod = control_question * question_words
            attn_weight = tf.squeeze(self.attn.apply(is_train, context_prod), axis=-1) 
            if question_mask is not None:
                m = tf.sequence_mask(question_mask)
                attn_weight += VERY_NEGATIVE_NUMBER * (1 - tf.cast(m, context_prod.dtype))
            ctrl_attn = tf.nn.softmax(attn_weight, 1)
            attn = tf.expand_dims(ctrl_attn, axis=2)
            next_control = tf.reduce_sum(attn * question_words, axis=1)
        # read unit
        with tf.variable_scope("read", reuse=reuse):
            last_mem = self.mem_drop.apply(is_train, prev_mem)
            know = self.read_drop.apply(is_train, document)
            proj_mem = tf.expand_dims(self.mem_proj.apply(is_train, last_mem), axis=1)
            proj_know = self.kb_proj.apply(is_train, know)
            concat = self.concat2.apply(is_train, tf.nn.elu(self.concat.apply(is_train, tf.concat([proj_mem * proj_know, proj_know], axis=2))   ))
            out = self.lin.apply(is_train, self.bi.apply(is_train, concat, question_words, question_words, ctrl_attn, document_mask,    question_mask))
            attn = self.read_drop.apply(is_train, out)
            attn = tf.squeeze(self.rattn.apply(is_train, attn), axis=-1)
            if document_mask is not None:
                m = tf.sequence_mask(document_mask)
                attn += VERY_NEGATIVE_NUMBER * (1 - tf.cast(m, attn.dtype))
            attn = tf.expand_dims(tf.nn.softmax(attn, 1), axis=2)
            read = tf.reduce_sum(attn * know, axis=1)
        # write unit, with memory gate.
        with tf.variable_scope("write", reuse=reuse):
            concat = self.write.apply(is_train, tf.concat([read, prev_mem, next_control], axis=1))
            gate = tf.sigmoid(self.gate.apply(is_train, next_control) + 1.0)
            next_mem = gate * prev_mem + (1 - gate) * concat
        # return results of cell!
        return next_control, next_mem, out

class MacNetwork():
    """ Basic non-recurrent attention using the given SimilarityFunction """

    def __init__(self, num_mac_cells: int, hidden_dim: int):
        self.cells = num_mac_cells
        self.mac = Mac(hidden_dim)
        self.hidden_dim = hidden_dim
        self.acts = []
        self.qenc = CudnnGru(hidden_dim, w_init=TruncatedNormal(stddev=0.05))
        self.control_proj = FullyConnected(hidden_dim)
        for _ in range(num_mac_cells):
            self.acts.append(FullyConnected(hidden_dim))

    def apply(self, is_train, document, questions, document_mask=None, question_mask=None):
        # create question vec
        # the cudnnGRU layer reverses the sequences and stuff so we just grab last hidden states.
        question_hidden = self.qenc.apply(is_train, questions, question_mask)[:, -1]
        # shared projection
        question_vec = tf.tanh(self.control_proj.apply(is_train, question_hidden))
        # create initial memory and control states
        init_control = question_vec
        init_memory = tf.get_variable('init_memory',
                    shape=(1, self.hidden_dim),
                    trainable=True,
                )
        init_memory = tf.tile(init_memory, [tf.shape(questions)[0], 1])
        # going through the cells!
        control, memory = init_control, init_memory
        for i in range(self.cells):
            # control projection stuff
            position_cont = self.acts[i].apply(is_train, question_vec)
            # call mac cell
            with tf.variable_scope('macmsc', reuse=False if i == 0 else True):
                next_control, next_mem, out = self.mac.apply(
                is_train, document, questions, question_vec, control,
                position_cont, memory, False if i == 0 else True, document_mask, question_mask
                )
            control, memory = next_control, next_mem
        # no yes/no questions, so no need for outputting states.
        return out


