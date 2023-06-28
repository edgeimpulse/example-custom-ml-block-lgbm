import edgeimpulse.jax.jax2tf
import tensorflow as tf
import flatbuffers
import flatbuffers.flexbuffers
import numpy as np
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.lite.python import schema_py_generated as schema_fb
import secrets

def convert(fun, model_path, lgbm, input_signature, output_signature):
    print('Converting JAX function to TFLite model...')
    model = convert_tflite(fun, input_signature)
    print('Patching custom operators...')
    model = patch_custom_operators(model, lgbm.lgbm_attributes, output_signature)
    print('Saving model...')
    flatbuffer_utils.write_model(model, model_path)
    print('Done!')


def convert_tflite(fun, input_signature):
    tf_predict = tf.function(
        edgeimpulse.jax.jax2tf.convert(fun, enable_xla=False),
        input_signature=input_signature,
        autograph=False)
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf_predict.get_concrete_function()], tf_predict)
    converter.allow_custom_ops = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    model = converter.convert()
    return model

def patch_custom_operators(tflite_model, tree_attributes, output_signature):
    model = flatbuffer_utils.convert_bytearray_to_object(tflite_model)
    for op_code in model.operatorCodes:
      if op_code.customCode:
        op_code_str = op_code.customCode.decode('ascii')
        op_code.customCode = "TreeEnsembleClassifier"

    fbb = flatbuffers.flexbuffers.Builder()
    with fbb.Map():
        fbb.UInt('version', 1)
        fbb.String('tree_index_type', 'uint16')
        fbb.String('class_index_type', 'uint8')
        fbb.String('node_value_type', 'float32')
        fbb.String('equality_operator', 'leq')
        fbb.UInt('num_leaf_nodes', tree_attributes['num_leaf_nodes'])
        fbb.UInt('num_internal_nodes', tree_attributes['num_internal_nodes'])
        fbb.UInt('num_trees', len(tree_attributes['tree_root_ids']))
        fbb.String('class_weight_type', 'float32')
        fbb.Blob('nodes_featureids', np.asarray(tree_attributes['nodes_featureids'], dtype=np.uint16))
        fbb.Blob('nodes_values', np.asarray(tree_attributes['nodes_values'], dtype=np.float32))
        fbb.Blob('nodes_truenodeids', np.asarray(tree_attributes['nodes_truenodeids'], dtype=np.uint16))
        fbb.Blob('nodes_falsenodeids', np.asarray(tree_attributes['nodes_falsenodeids'], dtype=np.uint16))
        fbb.Blob('nodes_weights', np.asarray(tree_attributes['nodes_weights'], dtype=np.float32))
        fbb.Blob('nodes_classids', np.asarray(tree_attributes['nodes_classids'], dtype=np.uint8))
        fbb.Blob('tree_root_ids', np.asarray(tree_attributes['tree_root_ids'], dtype=np.uint16))
    options = fbb.Finish()

    for s in model.subgraphs:
      for o in s.operators:
        if str(model.operatorCodes[o.opcodeIndex].customCode) ==  "TreeEnsembleClassifier":
            o.customOptions = np.array(options, dtype=np.uint8)
            for g in o.outputs:
               s.tensors[g].shape = output_signature

    return model
