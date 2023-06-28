import numpy as np
import jax.numpy as jnp
import jax
from edgeimpulse.jax.jax2tf import tree_classifier_p
from jax._src import abstract_arrays
from enum import IntEnum
from onnxconverter_common.data_types import FloatTensorType
from onnxmltools.convert import convert_lightgbm

lgbm_model = None
tree_attributes = None

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def softmax(values):
    values = jnp.exp(values - values.max())
    res = values / values.sum()
    return res

def pred_lgbm(x):
    if tree_attributes['quantize']:
        x = ((x - tree_attributes['x_min']) / tree_attributes['x_diff']) * 255
        x = jnp.clip(x, 0, 255)
        x = jnp.array(x, dtype=jnp.uint8)

    def tree(tree_ix):
        def leaf(leaf_ix):
            return jnp.where(
                x[tree_attributes['nodes_featureids'][leaf_ix]] <= tree_attributes['nodes_values'][leaf_ix],
                tree_attributes['nodes_truenodeids'][leaf_ix],
                tree_attributes['nodes_falsenodeids'][leaf_ix])
        return jax.lax.while_loop(lambda ix: ix < tree_attributes['num_internal_nodes'], leaf, tree_ix)
    rix = jax.vmap(tree)(tree_attributes['tree_root_ids'])
    res = jax.vmap(lambda rx: jnp.zeros((tree_attributes['num_classes'],), dtype=jnp.float32)
                   .at[tree_attributes['nodes_classids'][rx - tree_attributes['num_internal_nodes']]]
                   .set(tree_attributes['nodes_weights'][rx - tree_attributes['num_internal_nodes']])
                   )(rix)
    return softmax(jnp.sum(res, axis=0))

def tree_classifier_prim(x):
  return tree_classifier_p.bind(x)

class LGBM:
    def __init__(self, lgbm, input_shape, num_classes, quantize=False, x=None, y=None):
        self.lgbm = lgbm
        self.lgbm_attributes = extract_lgbm(lgbm, input_shape, num_classes, quantize, x, y)
        self.input_shape = input_shape

        def tree_classifier_impl(x):
            return pred_lgbm(x)
        tree_classifier_p.def_impl(tree_classifier_impl)

        def tree_classifier_abstract_eval(x):
            return abstract_arrays.ShapedArray((num_classes,), dtype=jnp.float32)
        tree_classifier_p.def_abstract_eval(tree_classifier_abstract_eval)

    def predict(self, x):
        return tree_classifier_prim(x)

def get_attrvals_i(converted, name):
    for i in range(len(converted.graph.node._values)):
        for j in range(len(converted.graph.node._values[i].attribute)):
            if converted.graph.node._values[i].attribute[j].name == name:
                return converted.graph.node._values[i].attribute[j].ints
    return None
def get_attrvals_f(converted, name):
    for i in range(len(converted.graph.node._values)):
        for j in range(len(converted.graph.node._values[i].attribute)):
            if converted.graph.node._values[i].attribute[j].name == name:
                return converted.graph.node._values[i].attribute[j].floats
    return None
def get_attrvals_s(converted, name):
    for i in range(len(converted.graph.node._values)):
        for j in range(len(converted.graph.node._values[i].attribute)):
            if converted.graph.node._values[i].attribute[j].name == name:
                return converted.graph.node._values[i].attribute[j].strings
    return None


class NodeMode(IntEnum):
    LEAF = 0
    BRANCH_LEQ = 1
    BRANCH_LT = 2
    BRANCH_GTE = 3
    BRANCH_GT = 4
    BRANCH_EQ = 5
    BRANCH_NEQ = 6
def to_mode(s):
    if 'LEAF' == s:
        return NodeMode.LEAF
    elif 'BRANCH_LEQ' == s:
        return NodeMode.BRANCH_LEQ
    elif 'BRANCH_LT' == s:
        return NodeMode.BRANCH_LT
    elif 'BRANCH_GTE' == s:
        return NodeMode.BRANCH_GTE
    elif 'BRANCH_GT' == s:
        return NodeMode.BRANCH_GT
    elif 'BRANCH_EQ' == s:
        return NodeMode.BRANCH_EQ
    elif 'BRANCH_NEQ' == s:
        return NodeMode.BRANCH_NEQ
    else:
        raise Exception("Unknown node mode: " + str(s))

def extract_lgbm(lgbm, input_shape, num_classes, quantize=False, x=None, y=None):
    initial_type = [('float_input', FloatTensorType(input_shape))]
    converted = convert_lightgbm(lgbm, initial_types=initial_type)

    _nodes_treeids = np.array(get_attrvals_i(converted, 'nodes_treeids'), dtype=jnp.int32)
    _tree_ids = np.array(list(sorted(set(get_attrvals_i(converted, 'nodes_treeids')))))

    _class_ids = np.array(get_attrvals_i(converted, 'class_ids'), dtype=jnp.int32)
    _classes = np.unique(_class_ids)
    _class_weights = np.array(get_attrvals_f(converted, 'class_weights'), dtype=jnp.float32)
    _class_nodeids = np.array(get_attrvals_i(converted, 'class_nodeids'), dtype=jnp.int32)
    _class_treeids = np.array(get_attrvals_i(converted, 'class_treeids'), dtype=jnp.int32)

    _nodes_modes = np.array(list(map(lambda x: int(to_mode(str(x).replace('b', '').replace('\'', ''))), get_attrvals_s(converted, 'nodes_modes'))), dtype=np.int16)
    _nodes_featureids = np.array(get_attrvals_i(converted, 'nodes_featureids'), dtype=jnp.int32)
    _nodes_values = np.array(get_attrvals_f(converted, 'nodes_values'), dtype=jnp.float32)
    _nodes_truenodeids = np.array(get_attrvals_i(converted, 'nodes_truenodeids'), dtype=jnp.int32)
    _nodes_falsenodeids = np.array(get_attrvals_i(converted, 'nodes_falsenodeids'), dtype=jnp.int32)
    _tree_root_ids = np.array([get_attrvals_i(converted, 'nodes_treeids').index(tid) for tid in _tree_ids], dtype=jnp.int32)

    for ix in range(_nodes_truenodeids.shape[0]):
        _nodes_truenodeids[ix] = _nodes_truenodeids[ix] + _tree_root_ids[_nodes_treeids[ix]]
        _nodes_falsenodeids[ix] = _nodes_falsenodeids[ix] + _tree_root_ids[_nodes_treeids[ix]]

    _nodes_weights = np.full(_nodes_treeids.shape[0], -1, dtype=jnp.float32) #.fill(-1)
    _nodes_classids = np.full(_nodes_treeids.shape[0], -1, dtype=jnp.int32) #.fill(-1)
    for ix in range(_class_ids.shape[0]):
        _nodes_weights[_tree_root_ids[_class_treeids[ix]] + _class_nodeids[ix]] = _class_weights[ix]
        _nodes_classids[_tree_root_ids[_class_treeids[ix]] + _class_nodeids[ix]] = _class_ids[ix]

    x_min = None
    x_diff = None
    if quantize:
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
        x_diff = x_max - x_min

    _nodes_modes_new = []
    _nodes_featureids_new = []
    _nodes_values_new = []
    _nodes_truenodeids_new = []
    _nodes_falsenodeids_new = []
    _nodes_classids_new = []
    _nodes_id_mapping = {}
    _nodes_weights_new = []
    _tree_root_ids_new = []

    num_leaf_nodes = 0
    num_internal_nodes = 0
    for ix in range(_nodes_modes.shape[0]):
        if int(_nodes_modes[ix]) == 0:
            continue
        else:
            _nodes_modes_new.append(_nodes_modes[ix])
            _nodes_featureids_new.append(_nodes_featureids[ix])
            _nodes_values_new.append(_nodes_values[ix])
            _nodes_id_mapping[ix] = num_internal_nodes
            num_internal_nodes = num_internal_nodes + 1

    for ix in range(_nodes_modes.shape[0]):
        if int(_nodes_modes[ix]) != 0:
            continue
        _nodes_classids_new.append(_nodes_classids[ix])
        _nodes_weights_new.append(_nodes_weights[ix])
        _nodes_id_mapping[ix] = num_internal_nodes + num_leaf_nodes
        num_leaf_nodes = num_leaf_nodes + 1

    # print('node id mapping: ' + str(_nodes_id_mapping))
    for ix in range(_nodes_modes.shape[0]):
        if int(_nodes_modes[ix]) == 0:
            continue
        _nodes_truenodeids_new.append(_nodes_id_mapping[_nodes_truenodeids[ix]])
        _nodes_falsenodeids_new.append(_nodes_id_mapping[_nodes_falsenodeids[ix]])

    for ix in range(_tree_root_ids.shape[0]):
         _tree_root_ids_new.append(_nodes_id_mapping[_tree_root_ids[ix]])

    # print('node id mapping:' + str(_tree_root_ids_new))
    # print('tree root ids new:' + str(_tree_root_ids_new))
    # print('node modes new:' + str(_nodes_modes_new))
    # print('node featureids new:' + str(_nodes_featureids_new))
    # print('node values new:' + str(_nodes_values_new))
    # print('node truenodeids new:' + str(_nodes_truenodeids_new))
    # print('node falsenodeids new:' + str(_nodes_falsenodeids_new))
    # print('node classids old' + str(_nodes_classids))
    # print('node classids new:' + str(_nodes_classids_new))

    # print('tree root ids new:' + str(len(_tree_root_ids_new)))
    # print('node modes new:' + str(len(_nodes_modes_new)))
    # print('node featureids new:' + str(len(_nodes_featureids_new)))
    # print('node values new:' + str(len(_nodes_values_new)))
    # print('node truenodeids new:' + str(len(_nodes_truenodeids_new)))
    # print('node falsenodeids new:' + str(len(_nodes_falsenodeids_new)))
    # print('node classids new:' + str(len(_nodes_classids_new)))

    # print('node weights:' + str(len(_nodes_weights)))
    # print('node weights new:' + str(len(_nodes_weights_new)))

    # print('class ids:' + str(_class_ids))
    # print('class weights:' + str(_class_weights))

    def format_c_arr(arr):
        return '{' + ', '.join(list(map(lambda x: str(x), list(arr)))) + '}'

    print('')
    print('----------------- Tree attributes -----------------')
    # print('class_ids: ' + format_c_arr(_class_ids))
    # print('classes: ' + format_c_arr(_classes))
    # print('class_weights: ' + format_c_arr(_class_weights))
    # print('class_nodeids: ' + format_c_arr(_class_nodeids))
    # print('class_treeids: ' + format_c_arr(_class_treeids))
    # print('node_tree_ids ' + format_c_arr(_nodes_treeids))
    # print('tree_ids ' + format_c_arr(_tree_ids))
    # print('node_modes: ' + format_c_arr(_nodes_modes))
    # print('nodes_featureids: ' + format_c_arr(_nodes_featureids))
    # print('nodes_values: ' + format_c_arr(_nodes_values))
    # print('nodes_truenodeids: ' + format_c_arr(_nodes_truenodeids))
    # print('nodes_falsenodeids: ' + format_c_arr(_nodes_falsenodeids))
    # print('nodes_weights: ' + format_c_arr(_nodes_weights))
    # print('nodes_classids: ' + format_c_arr(_nodes_classids))
    # print('tree_root_ids: ' + format_c_arr(_tree_root_ids))

    print('num roots: ' + str(len(_tree_root_ids)))
    print('num total nodes: ' + str(len(_nodes_modes)))
    print('num leaf nodes: ' + str(num_leaf_nodes))
    print('num internal nodes: ' + str(num_internal_nodes))
    print('num leaf nodes: ' + str(np.count_nonzero(_nodes_modes == 0)))
    print('num internal nodes: ' + str(np.count_nonzero(_nodes_modes != 0)))
    print('---------------------------------------------------')
    print('')

    global tree_attributes
    tree_attributes = None
    if quantize:
        for ix in range(_nodes_featureids.shape[0]):
            _nodes_values[ix] = ((_nodes_values[ix] - x_min[_nodes_featureids[ix]]) / x_diff[_nodes_featureids[ix]]) * 255
        _nodes_values = np.clip(_nodes_values, 0, 255)

        tree_attributes = {
            'num_leaf_nodes': np.count_nonzero(_nodes_modes == 0),
            'num_internal_nodes': np.count_nonzero(_nodes_modes != 0),
            'equality_operator': 'leq',
            "tree_root_ids": jnp.array(_tree_root_ids, dtype=jnp.int32),
            "nodes_featureids": jnp.array(_nodes_featureids, dtype=jnp.int32),
            "nodes_modes": jnp.array(_nodes_modes, dtype=jnp.int32),
            "nodes_values": jnp.array(_nodes_values, dtype=jnp.uint8),
            "nodes_truenodeids": jnp.array(_nodes_truenodeids, dtype=jnp.int32),
            "nodes_falsenodeids": jnp.array(_nodes_falsenodeids, dtype=jnp.int32),
            "nodes_classids": jnp.array(_nodes_classids, dtype=jnp.int32),
            "nodes_weights": jnp.array(_nodes_weights, dtype=jnp.float32),
            "class_weights": jnp.array(_class_weights, dtype=jnp.float32),
            "num_classes": num_classes,
            "quantize": quantize,
            "x_min": x_min,
            "x_diff": x_diff
        }
    else:
        tree_attributes = {
            'num_leaf_nodes': num_leaf_nodes,
            'num_internal_nodes': num_internal_nodes,
            'equality_operator': 'leq',
            "tree_root_ids": jnp.array(_tree_root_ids_new, dtype=jnp.int32),
            "nodes_featureids": jnp.array(_nodes_featureids_new, dtype=jnp.int32),
            "nodes_modes": jnp.array(_nodes_modes_new, dtype=jnp.int32),
            "nodes_values": jnp.array(_nodes_values_new, dtype=jnp.float32),
            "nodes_truenodeids": jnp.array(_nodes_truenodeids_new, dtype=jnp.int32),
            "nodes_falsenodeids": jnp.array(_nodes_falsenodeids_new, dtype=jnp.int32),
            "nodes_classids": jnp.array(_nodes_classids_new, dtype=jnp.int32),
            "nodes_weights": jnp.array(_nodes_weights_new, dtype=jnp.float32),
            "class_weights": jnp.array(_class_weights, dtype=jnp.float32),
            "num_classes": num_classes,
            "quantize": quantize,
            "x_min": x_min,
            "x_diff": x_diff
        }

    return tree_attributes