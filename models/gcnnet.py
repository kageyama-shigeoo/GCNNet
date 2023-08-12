from functools import partial

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

def instance_normalization(features):
    with tf.variable_scope(None, default_name='IN'):
        mean, variance = tf.nn.moments(
            features, [0], name='IN_stats', keep_dims=True)
        features = tf.nn.batch_normalization(
            features, mean, variance, None, None, 1e-12, name='IN_apply')
    return(features)

normalization_fn_dict = {
    'fused_BN_center': slim.batch_norm,
    'BN': partial(slim.batch_norm, fused=False, center=False),
    'BN_center': partial(slim.batch_norm, fused=False),
    'IN': instance_normalization,
    'NONE': None
}
activation_fn_dict = {
    'ReLU': tf.nn.relu,
    'ReLU6': tf.nn.relu6,
    'LeakyReLU': partial(tf.nn.leaky_relu, alpha=0.01),
    'ELU':tf.nn.elu,
    'NONE': None,
    'Sigmoid': tf.nn.sigmoid,
    'Tanh': tf.nn.tanh,
}

def multi_layer_fc_fn(sv, mask=None, Ks=(64, 32, 64), num_classes=4,
    is_logits=False, num_layer=4, normalization_type="fused_BN_center",
    activation_type='ReLU'):
    """A function to construct multiple layers of a neural network for 
    computing features propagated through each edge.

	Arguments:

	sv: A tensor of dimensions [N, M] or [T, DEGREE, M], where N 
	is the total edge count, M is the feature length. T represents 
	the number of receiving vertices, and DEGREE is the in-degree of each 
	receiving vertex. If a [T, DEGREE, M] tensor is provided, it assumes that 
	the degree of each receiving vertex is uniform.
	mask: An optional tensor of dimensions [N, 1] or [T, DEGREE, 1]. A value of 1 signifies a valid output feature, while 0 indicates an invalid one (set to 0).
	num_layer: Number of layers to be added.
	
	Returns:
	A tensor of dimensions [N, K] or [T, DEGREE, K], where K represents the length of the new edge features.
    """
    assert len(sv.shape) == 2
    assert len(Ks) == num_layer-1
    if is_logits:
        features = sv
        for i in range(num_layer-1):
            features = slim.fully_connected(features, Ks[i],
                activation_fn=activation_fn_dict[activation_type],
                normalizer_fn=normalization_fn_dict[normalization_type],
                )
        features = slim.fully_connected(features, num_classes,
            activation_fn=None,
            normalizer_fn=None
            )
    else:
        features = sv
        for i in range(num_layer-1):
            features = slim.fully_connected(features, Ks[i],
                activation_fn=activation_fn_dict[activation_type],
                normalizer_fn=normalization_fn_dict[normalization_type],
                )
        features = slim.fully_connected(features, num_classes,
            activation_fn=activation_fn_dict[activation_type],
            normalizer_fn=normalization_fn_dict[normalization_type],
            )
    if mask is not None:
        features = features * mask
    return features

def multi_layer_neural_network_fn(features, Ks=(64, 32, 64), is_logits=False,
    normalization_type="fused_BN_center", activation_type='ReLU'):
    """A function to create multiple layers of neural network.
    """
    assert len(features.shape) == 2
    if is_logits:
        for i in range(len(Ks)-1):
            features = slim.fully_connected(features, Ks[i],
                activation_fn=activation_fn_dict[activation_type],
                normalizer_fn=normalization_fn_dict[normalization_type])
        features = slim.fully_connected(features, Ks[-1],
            activation_fn=None,
            normalizer_fn=None)
    else:
        for i in range(len(Ks)):
            features = slim.fully_connected(features, Ks[i],
                activation_fn=activation_fn_dict[activation_type],
                normalizer_fn=normalization_fn_dict[normalization_type])
    return features

def graph_scatter_max_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_max(point_features,
        point_centers, num_centers, name='scatter_max')
    return aggregated

def graph_scatter_sum_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_sum(point_features,
        point_centers, num_centers, name='scatter_sum')
    return aggregated

def graph_scatter_mean_fn(point_features, point_centers, num_centers):
    aggregated = tf.math.unsorted_segment_mean(point_features,
        point_centers, num_centers, name='scatter_mean')
    return aggregated

class ClassAwarePredictor(object):
    """A class to predict 3D bounding boxes and class labels."""

    def __init__(self, cls_fn, loc_fn):
        """
        Args:
            cls_fn: a function to classify labels.
            loc_fn: a function to predict 3D bounding boxes.
        """
        self._cls_fn = cls_fn
        self._loc_fn = loc_fn

    def apply_regular(self, features, num_classes, box_encoding_len,
        normalization_type='fused_BN_center',
        activation_type='ReLU'):
        """
        Args:
            input_v: input feature vectors. [N, M].
            output_v: not used.
            A: not used.
            num_classes: the number of classes to predict.

        returns: logits, box_encodings.
        """
        box_encodings_list = []
        with tf.variable_scope('predictor'):
            with tf.variable_scope('cls'):
                logits = self._cls_fn(
                    features, num_classes=num_classes, is_logits=True,
                    normalization_type=normalization_type,
                    activation_type=activation_type)
            with tf.variable_scope('loc'):
                for class_idx in range(num_classes):
                    with tf.variable_scope('cls_%d' % class_idx):
                        box_encodings = self._loc_fn(
                            features, num_classes=box_encoding_len,
                            is_logits=True,
                            normalization_type=normalization_type,
                            activation_type=activation_type)
                        box_encodings = tf.expand_dims(box_encodings, axis=1)
                        box_encodings_list.append(box_encodings)
            box_encodings = tf.concat(box_encodings_list, axis=1)
        return logits, box_encodings

class ClassAwareSeparatedPredictor(object):
    """A class to predict 3D bounding boxes and class labels."""

    def __init__(self, cls_fn, loc_fn):
        """
        Args:
            cls_fn: a function to classify labels.
            loc_fn: a function to predict 3D bounding boxes.
        """
        self._cls_fn = cls_fn
        self._loc_fn = loc_fn

    def apply_regular(self, features, num_classes, box_encoding_len,
        normalization_type='fused_BN_center',
        activation_type='ReLU'):
        """
        Args:
            input_v: input feature vectors. [N, M].
            output_v: not used.
            A: not used.
            num_classes: the number of classes to predict.

        returns: logits, box_encodings.
        """
        box_encodings_list = []
        with tf.variable_scope('predictor'):
            with tf.variable_scope('cls'):
                logits = self._cls_fn(
                    features, num_classes=num_classes, is_logits=True,
                    normalization_type=normalization_type,
                    activation_type=activation_type)
            features_splits = tf.split(features, num_classes, axis=-1)
            with tf.variable_scope('loc'):
                for class_idx in range(num_classes):
                    with tf.variable_scope('cls_%d' % class_idx):
                        box_encodings = self._loc_fn(
                            features_splits[class_idx],
                            num_classes=box_encoding_len,
                            is_logits=True,
                            normalization_type=normalization_type,
                            activation_type=activation_type)
                        box_encodings = tf.expand_dims(box_encodings, axis=1)
                        box_encodings_list.append(box_encodings)
            box_encodings = tf.concat(box_encodings_list, axis=1)
        return logits, box_encodings

class PointSetPooling(object):
    """A class to implement local graph netural network."""

    def __init__(self,
        point_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        output_fn=multi_layer_neural_network_fn):
        self._point_feature_fn = point_feature_fn
        self._aggregation_fn = aggregation_fn
        self._output_fn = output_fn

    def apply_regular(self,
        point_features,
        point_coordinates,
        keypoint_indices,
        set_indices,
        point_MLP_depth_list=None,
        point_MLP_normalization_type='fused_BN_center',
        point_MLP_activation_type = 'ReLU',
        output_MLP_depth_list=None,
        output_MLP_normalization_type='fused_BN_center',
        output_MLP_activation_type = 'ReLU'):
        """
	Extracts features from point sets.

	Args:

	point_features: A tensor of dimensions [N, M], where N is the number of points and M is the feature length.
	point_coordinates: A tensor of dimensions [N, D], where N is the number of points and D is the dimension of coordinates.
	keypoint_indices: A tensor of dimensions [K, 1] containing indices of K keypoints.
	set_indices: A tensor of dimensions [S, 2] representing pairs (point_index, set_index). (i, j) indicates that 
	point[i] is part of the point set created by grouping around keypoint[j].
	point_MLP_depth_list: A list of MLP units for extracting point features.
	point_MLP_normalization_type: The normalization function used in the point MLP.
	point_MLP_activation_type: The activation function used in the point MLP.
	output_MLP_depth_list: A list of MLP units for embedding set features.
	output_MLP_normalization_type: The normalization function used in the output MLP.
	output_MLP_activation_type: The activation function used in the output MLP.
        Returns:

	A tensor of dimensions [K, output_depth] representing the set feature. 
	The actual output depth depends on the chosen feature extraction options.

        """
        # Gather the points in a set
        point_set_features = tf.gather(point_features, set_indices[:,0])
        point_set_coordinates = tf.gather(point_coordinates, set_indices[:,0])
        # Gather the keypoints for each set
        point_set_keypoint_indices = tf.gather(
            keypoint_indices, set_indices[:, 1])
        point_set_keypoint_coordinates = tf.gather(point_coordinates,
            point_set_keypoint_indices[:,0])
        # points within a set use relative coordinates to its keypoint
        point_set_coordinates = \
            point_set_coordinates - point_set_keypoint_coordinates
        point_set_features = tf.concat(
            [point_set_features, point_set_coordinates], axis=-1)
        with tf.variable_scope('extract_vertex_features'):
            # Step 1: Extract all vertex_features
            extracted_point_features = self._point_feature_fn(
                point_set_features,
                Ks=point_MLP_depth_list, is_logits=False,
                normalization_type=point_MLP_normalization_type,
                activation_type=point_MLP_activation_type)
            set_features = self._aggregation_fn(
                extracted_point_features, set_indices[:, 1],
                tf.shape(keypoint_indices)[0])
        with tf.variable_scope('combined_features'):
            set_features = self._output_fn(set_features,
                Ks=output_MLP_depth_list, is_logits=False,
                normalization_type=output_MLP_normalization_type,
                activation_type=output_MLP_activation_type)
        return set_features

class GraphNetAutoCenter(object):
    """A class to implement point graph netural network layer."""

    def __init__(self,
        edge_feature_fn=multi_layer_neural_network_fn,
        aggregation_fn=graph_scatter_max_fn,
        update_fn=multi_layer_neural_network_fn,
        auto_offset_fn=multi_layer_neural_network_fn):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn

    def apply_regular(self,
        input_vertex_features,
        input_vertex_coordinates,
        NOT_USED,
        edges,
        edge_MLP_depth_list=None,
        edge_MLP_normalization_type='fused_BN_center',
        edge_MLP_activation_type = 'ReLU',
        update_MLP_depth_list=None,
        update_MLP_normalization_type='fused_BN_center',
        update_MLP_activation_type = 'ReLU',
        auto_offset=False,
        auto_offset_MLP_depth_list=None,
        auto_offset_MLP_normalization_type='fused_BN_center',
        auto_offset_MLP_feature_activation_type = 'ReLU',
        ):
        """Applies a single-layer graph network to a given graph.

	Args:

	input_vertex_features: A tensor of dimensions [N, M], where N is the number of vertices and M is the feature length.
	input_vertex_coordinates: A tensor of dimensions [N, D], where N is the number of vertices and D is the dimension of coordinates.
	NOT_USED: Reserved for API compatibility.
	edges: A tensor of dimensions [K, 2] containing pairs of (source, destination) vertex indices for K edges.
	edge_MLP_depth_list: A list of MLP units for extracting edge features.
	edge_MLP_normalization_type: The normalization function used in the edge MLP.
	edge_MLP_activation_type: The activation function used in the edge MLP.
	update_MLP_depth_list: A list of MLP units for extracting update features.
	update_MLP_normalization_type: The normalization function used in the update MLP.
	update_MLP_activation_type: The activation function used in the update MLP.
	auto_offset: A boolean indicating whether to use auto registration.
	auto_offset_MLP_depth_list: A list of MLP units for computing offset.
	auto_offset_MLP_normalization_type: The normalization function used in the auto-offset MLP.
	auto_offset_MLP_feature_activation_type: The activation function used in the auto-offset MLP.
	
	Returns:
	A tensor of dimensions [N, M] representing the updated vertex features.
        """
        # Gather the source vertex of the edges
        s_vertex_features = tf.gather(input_vertex_features, edges[:,0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:,0])
        # [optional] Compute the coordinates offset
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                Ks=auto_offset_MLP_depth_list, is_logits=True,
                normalization_type=auto_offset_MLP_normalization_type,
                activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
        # Gather the destination vertex of the edges
        d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])
        # Prepare initial edge features
        edge_features = tf.concat(
            [s_vertex_features, s_vertex_coordinates - d_vertex_coordinates],
             axis=-1)
        with tf.variable_scope('extract_vertex_features'):
            # Extract edge features
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            # Aggregate edge features
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        # Update vertex features
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(aggregated_edge_features,
                Ks=update_MLP_depth_list, is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        output_vertex_features = update_features + input_vertex_features
        return output_vertex_features
