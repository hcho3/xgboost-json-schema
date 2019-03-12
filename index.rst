################################
XGBoost JSON Schema, Version 1.0
################################

Preface
=======
This document contains an exhaustive description of the XGBoost JSON schema, a
mapping between XGBoost object classes and JSON objects and arrays. The version
1.0 of the schema aims to follow the behavior of the current binary
serialization method.

Representation of ``dmlc::Parameter`` objects
---------------------------------------------
Every object of subclasses of ``dmlc::Parameter`` are to be represented as JSON
objects. We will create a bridge method to seamlessly convert ``dmlc::Parameter``
objects into JSON objects, such that proper value types are used.

For example, consider the parameter class

.. code-block:: cpp

  struct MyParam : public dmlc::Parameter<MyParam> {
    float learning_rate;
    int num_hidden;
    int activation;
    std::string name;
    DMLC_DECLARE_PARAMETER(MyParam) {
      DMLC_DECLARE_FIELD(num_hidden);
      DMLC_DECLARE_FIELD(learning_rate);
      DMLC_DECLARE_FIELD(activation).add_enum("relu", 1).add_enum("sigmoid", 2);
      DMLC_DECLARE_FIELD(name);
    }
  };

and an object created by the following initialization code:

.. code-block:: cpp

  MyParam param;
  param.learning_rate = 0.1f;
  param.num_hidden = 10;
  param.activation = 2;  // sigmoid
  param.name = "MyNet";

This collection is naturally expressed as the following JSON object:

.. parsed-literal::

  {
    "learning_rate": 0.1,
    "num_hidden": 10,
    "activation": "sigmoid",
    "name": "MyNet"
  }

Notations
---------

In the following sections, the schema for each XGBoost class is shown as a JSON
object. Fields whose keys are marked with *italic* are optional and may be
absent in some models. The hyper-linked value indicate that the value shall be
the JSON representation of another XGBoost class. The *italic* value indicate
that the value shall be a primitive type (string, integer, floating-point etc).
Every mention of ``floating-point`` refers to single-precision floating point
(32-bit), unless explicitly stated otherwise.  Every mention of ``integer``
refers to 32-bit integer unless stated otherwise.

Full content of the schema
==========================

Note: Click :ref:`here <example>` for a minimal example of the current schema.

.. contents:: :local:

XGBoostModel
------------
This is the root object for XGBoost model.

.. parsed-literal::

  {
    "version" : [1, 0],
    "leaf_node_schema" : ["leaf_output", "floating-point"],
    "test_node_schema" : [
      ["child_left_id", "integer"],
      ["child_right_id", "integer"],
      ["feature_id", "integer"],
      ["threshold", "floating-point"],
      ["default_left", "boolean"]
    ],
    "node_stat_schema": [
      ["loss_chg", "floating-point"],
      ["sum_hess", "floating-point"],
      ["base_weight", "floating-point"],
      ["instance_cnt", "64-bit integer"]
    ],
    "learner" : Learner_
  }

The fields ``leaf_node_schema``, ``test_node_schema``, and ``node_stat_schema``
exist for informational purposes only, describing the classes LeafNode_,
TestNode_, and NodeStat_ respectively. These schema fields may be omitted
altogether for the interest of space.

The classes LeafNode_, TestNode_, and NodeStat_ are represented as JSON
arrays so that we don't have to repeat the field names over and over again.
Instead, the field names are listed only once (or not at all) in the beginning
of the serialized JSON file.

Learner
-------
.. parsed-literal::

  {
    "learner_model_param" : LearnerModelParam_,
    "predictor_param" : PredictorParam_,
    "name_obj" : *string*,
    "name_gbm" : *string*,
    "gbm" : GBM_,
    "attributes" : StringKeyValuePairCollection_,
    "eval_metrics" : [ *array of string* ],
    *"count_poisson_max_delta_step"* : *floating-point*
  }

The ``count_poisson_max_delta_step`` field is only used for Poisson regression
task, where ``name_obj`` (name of objective function) is set to
``count:poisson``.

The ``attributes`` is a generic key-value collection, retained for compatibility
purposes with the current binary serialization method.

LearnerModelParam
-----------------
This class is a subclass of ``dmlc::Parameter``.

.. parsed-literal::

  {
    "base_score" : *floating-point*,
    "num_feature" : *64-bit integer*,
    *"num_class"* : *integer*
  }

The ``num_class`` is used only for multi-class classification task, in which it
indicates the number of output classes.

GBM
---
Currently, we may choose one of the three subclasses for this placeholder:

* GBTree_: decision tree models (``name_gbm="gbtree"``)
* Dart_: DART (Dropouts meet Multiple Additive Regression Trees) models
  (``name_gbm="dart"``)
* GBLinear_: linear models (``name_gbm="gblinear"``)

We can determine which subclass was used by looking at the ``name_gbm`` field
of Learner_.

GBTree
------
.. parsed-literal::

  {
    "model_param" : GBTreeModelParam_,
    "trees" : [ *array of* RegTree_ ],
    *"tree_info"* : [ *array of integer* ]
  }

``tree_info`` is a reserved field, retained for the sake of compatibility
with the current binary serialization method.

Dart
----
.. parsed-literal::

  {
    "model_param" : GBTreeModelParam_,
    "trees" : [ *array of* RegTree_ ],
    *"tree_info"* : [ *array of int* ],
    *"weight_drop"* : [ *array of floating-point* ]
  }

``tree_info`` is a reserved field, retained for the sake of compatibility
with the current binary serialization method.

RegTree
-------
.. parsed-literal::

  {
    "tree_param" : TreeParam_,
    "nodes" : [ *array of* Node_ ],
    "stats" : [ *array of* NodeStat_ ],
  }

The first ``num_roots`` nodes in the ``nodes`` array specify root node(s).
(The ``num_roots`` field is specified in GBTreeModelParam_.) For most use cases,
the decision tree has one root and ``num_roots`` is 1, so the first entry in the
``nodes`` array specifies the root node.

The ``nodes`` array specify an adjacency list for an acyclic directed binary
tree graph. Each tree node has zero or two outgoing edges and exactly one
incoming edge. Cycles are not allowed.

GBTreeModelParam
----------------
This class is a subclass of ``dmlc::Parameter``.

.. parsed-literal::

  {
    "num_roots" : *integer*,
    "num_feature" : *64-bit integer*,
    "num_output_group" : *integer*
  }

The ``num_output_group`` is the size of prediction per instance. This value is
set to 1 for all tasks except multi-class classification. For multi-class
classification, ``num_output_group`` must be set to the number of classes. This
must be identical to the value for ``num_class`` field of LearnerModelParam_
that was provided at training time.

The ``num_roots`` specified the number of roots in each tree. For most use
cases, this should be set to 1.

TreeParam
---------
This class is a subclass of ``dmlc::Parameter``.

.. parsed-literal::

  {
    *"num_deleted"* : *integer*
  }

The ``num_deleted`` field is optional and indicates that some node IDs are
marked deleted and thus should be re-used for creating new nodes. This exists
since the pruning method leaves gaps in node IDs. When omitted, ``num_deleted``
is assumed to be zero. This field may be deprecated in the future.

Node
----
We may choose one of the two subclasses for this placeholder:

* LeafNode_: leaf node (no child node, real output)
* TestNode_: non-leaf node (two child nodes, test condition)

We distinguish the two types of node by whether the node representation is a
JSON array (test node) or a single floating-point number (leaf node).

LeafNode
--------
Each leaf node is represented as a single floating-point number:

.. parsed-literal::

  *floating-point* (leaf_output)

The ``leaf_output`` field specifies the real-valued output associated with
the leaf node.

TestNode
--------
Each test node is represented as a JSON array of a fixed size, each element
storing the following fields:

.. parsed-literal::

  [
    *integer* (child_left_id),
    *integer* (child_right_id),
    *unsigned integer* (feature_id),
    *floating-point* (threshold),
    *boolean* (default_left)
  ]

The ``feature_id`` and ``threshold`` fields specify the feature ID and threshold
used in the test node, where the test is of form ``data[feature_id] < threshold``.
The ``child_left_id`` and ``child_right_id`` fields specify the nodes to be
taken in a tree traversal when the test ``data[feature_id] < threshold`` is true
and false, respectively. The node IDs are 0-based offsets to the ``nodes``
arrays in RegTree_. The ``default_left`` field indicates the default
direction in a tree traversal when feature value for ``feature_id`` is missing.

NodeStat
--------
Statistics for each node is represented as a JSON array of a fixed size, each
element storing the following fields:

.. parsed-literal::

  [
    *floating-point* (loss_chg),
    *floating-point* (sum_hess),
    *floating-point* (base_weight),
    *64-bit integer* (instance_cnt)
  ]

GBLinear
--------
.. parsed-literal::

  {
    "model_param" : GBLinearModelParam_,
    "weight" : [ *array of floating-point* ]
  }

GBLinearModelParam
------------------
This class is a subclass of ``dmlc::Parameter``.

.. parsed-literal::

  {
    "num_feature" : *64-bit integer*,
    "num_output_group" : *integer*
  }

PredictorParam
--------------
.. parsed-literal::

  {
    "predictor" : *string*,
    *"n_gpus"* : *integer*,
    *"gpu_id"* : *integer*
  }

StringKeyValuePairCollection
----------------------------
This class is a collection of key-value pairs. Both keys and values must be
string types, and keys must consist of alphabet letters, digits (0-9), and
underscore (``_``).

.. _example:

Minimal example
===============

.. code-block:: json

  {
    "version" : [1, 0],
    "leaf_node_schema" : ["leaf_output", "floating-point"],
    "test_node_schema" : [
      ["child_left_id", "integer"],
      ["child_right_id", "integer"],
      ["feature_id", "integer"],
      ["threshold", "floating-point"],
      ["default_left", "boolean"]
    ],
    "node_stat_schema": [
      ["loss_chg", "floating-point"],
      ["sum_hess", "floating-point"],
      ["base_weight", "floating-point"],
      ["instance_cnt", "64-bit integer"]
    ],
    "learner" : {
      "learner_model_param" : {
        "base_score" : 0.5,
        "num_feature" : 126
      },
      "predictor_param" : {
        "predictor" : "cpu_predictor"
      },
      "name_obj" : "binary:logistic",
      "name_gbm" : "gbtree",
      "gbm" : {
        "model_param" : {
          "num_roots" : 1,
          "num_feature" : 126,
          "num_output_group" : 1
        },
        "trees" : [
          {
            "tree_param" : {},
            "nodes" : [
              [1, 2,  28,  0.0,  true],
              [3, 4,  55,  0.5, false],
              [5, 6, 108,  1.0,  true],
               1.8,
              -1.9,
              [7, 8,  66, -0.5,  true],
               1.87,
              -1.99,
               0.94
            ],
            "stats" : [
              [200.0, 1635.2,  0.2, 4000],
              [150.2,  922.8,  1.1, 2200],
              [300.4,  712.5, -1.5, 1800],
              [  0.0,  808.3,  0.0, 2000],
              [  0.0,  114.5,  0.0,  200],
              [100.1,  698.0, -1.8, 1600],
              [  0.0,   14.5,  0.0,  200],
              [  0.0,  686.8,  0.0, 1500],
              [  0.0,   11.2,  0.0,  100]
            ]
          },
          {
            "tree_param" : {},
            "nodes" : [
              [1, 2, 5, 0.5, false],
               1.0,
              -1.0
            ],
            "stats" : [
              [335.0, 135.2,  0.6, 4000],
              [  0.0,  88.3,  0.0, 3000],
              [  0.0,  46.9,  0.0, 1000]
            ]
          }
        ]
      },
      "attributes" : {},
      "eval_metrics" : [ "auc" ]
    }
  }
