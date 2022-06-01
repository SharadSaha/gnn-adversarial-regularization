import neural_structured_learning as nsl
import tensorflow_hub as hub
import os
import argparse
import numpy as np
import tensorflow as tf
from get_data import get_data,read_params


# Initializing the hyperparameters

class HParams(object):
  """Hyperparameters used for training."""
  def __init__(self):
    ### dataset parameters
    self.num_classes = 4
    self.IMG_DIMENSION = [299,299,3]
    ### neural graph learning parameters
    self.distance_type = nsl.configs.DistanceType.L2
    self.graph_regularization_multiplier = 0.2
    self.num_neighbors = 3
    ### model architecture
    self.LOSS = 'sparse_categorical_crossentropy'
    self.OPT = 'adam'
    self.METRICS = ['accuracy']
    ### training parameters
    self.train_epochs = 15
    self.batch_size = 40
    ### eval parameters
    self.eval_steps = None  # All instances in the test set are evaluated.

hp = HParams()


# various types of train features 

def _int64_feature(value):
  """Returns int64 tf.train.Feature."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value.tolist()))


def _bytes_feature(value):
  """Returns bytes tf.train.Feature."""
  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))


def _float_feature(value):
  """Returns float tf.train.Feature."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value.tolist()))


def get_pretrained_model():
    hub_URL = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/5"

    hub_layer = hub.KerasLayer(hub_URL,trainable=False)
    return hub_layer


# create embedding example

def create_embedding_example(image,image_id):
  # Create image embedding
  hub_layer = get_pretrained_model()
  image_embedding = hub_layer([image])
  image_embedding = tf.reshape(image_embedding, shape=[-1])

  # Create features dictionary containing id and embedding
  features = {
      'id': _bytes_feature(str(image_id)),
      'embedding': _float_feature(image_embedding.numpy())
  }
  return tf.train.Example(features=tf.train.Features(feature=features))


# create embeddings

def create_embeddings(images, output_path, starting_id):
  id = int(starting_id)
  with tf.io.TFRecordWriter(output_path) as writer:
    for image in images:
      example = create_embedding_example(image, id)
      id = id + 1
      writer.write(example.SerializeToString())
  return id


def create_similarity_graph():
    graph_builder_config = nsl.configs.GraphBuilderConfig(
    similarity_threshold=0.85, lsh_splits=32, lsh_rounds=15, random_seed=12345)

    nsl.tools.build_graph_from_config(['Graph/embeddings.tfr'],
                                  'Graph/graph.tsv',
                                  graph_builder_config)


# create the final training and testing samples

def _bytes_feature_image(image):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[image]))

def create_example(image, label, record_id):
  """Create tf.Example containing the sample's image, label, and ID."""
  features = {
      'id': _bytes_feature(str(record_id)),
      'image': _bytes_feature_image(image.numpy()),
      'label': _int64_feature(np.asarray([label])),
  }
  return tf.train.Example(features=tf.train.Features(feature=features))

def create_records(images, labels, record_path, starting_record_id):
  record_id = int(starting_record_id)
  with tf.io.TFRecordWriter(record_path) as writer:
    for image, label in zip(images, labels):
      image = tf.io.encode_png(tf.cast(image,tf.uint8))
      example = create_example(image, label, record_id)
      record_id = record_id + 1
      writer.write(example.SerializeToString())
  return record_id



# create a tf record

def read_record(file_path):
    raw_image_dataset = tf.data.TFRecordDataset(file_path)

    image_feature_description = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    
    def _parse_image_function(example_proto):
      return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    

    return parsed_image_dataset




# create augmented data

def create_augmented_data():
    nsl.tools.pack_nbrs(
    'Graph/data/train_data.tfr',
    '',
    'Graph/graph.tsv',
    'Graph/data/augmented_train_data.tfr',
    add_undirected_edges=True,
    max_nbrs=3)




# preparing the dataset

NBR_FEATURE_PREFIX = 'NL_nbr_'
NBR_WEIGHT_SUFFIX = '_weight'

def get_default_image():
  image = tf.ones((299, 299, 3), dtype=tf.uint8)*255
  return tf.io.encode_png(image, compression=-1,name=None)

DEFAULT_IMG = get_default_image()

def make_dataset(file_path,train = False):

  def parse_example(example_proto):
    feature_spec = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    if train:
      for i in range(hp.num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'image')
        nbr_weight_key = '{}{}{}'.format(NBR_FEATURE_PREFIX, i,
                                         NBR_WEIGHT_SUFFIX)
        feature_spec[nbr_feature_key] = tf.io.FixedLenFeature(
            [],tf.string, default_value=DEFAULT_IMG)

        feature_spec[nbr_weight_key] = tf.io.FixedLenFeature(
            [1], tf.float32, default_value=tf.constant([0.0]))

    features = tf.io.parse_single_example(example_proto, feature_spec)

    features['image'] = tf.io.decode_png(features['image'], channels=3)
    if train:
      for i in range(hp.num_neighbors):
        nbr_feature_key = '{}{}_{}'.format(NBR_FEATURE_PREFIX, i, 'image')
        features[nbr_feature_key] = tf.io.decode_png(features[nbr_feature_key],channels=3)

    labels = features.pop('label')
    return features, labels

  dataset = tf.data.TFRecordDataset([file_path])
  if train:
    dataset = dataset.shuffle(10000)
  dataset = dataset.map(parse_example)
  dataset = dataset.batch(hp.batch_size)
  return dataset




  
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    config = read_params(config_path=parsed_args.config)
    processed_train_data_path = os.path.join(config['split_data']['train_path1'],config['split_data']['train_path2'],config['split_data']['train_path3'],'train_data.npy')
    processed_test_data_path = os.path.join(config['split_data']['test_path1'],config['split_data']['test_path2'],config['split_data']['test_path3'],'test_data.npy')
    with open(processed_train_data_path, 'rb') as f:
        X_train,Y_train = np.load(f,allow_pickle=True)
    with open(processed_test_data_path, 'rb') as f:
        X_test,Y_test = np.load(f,allow_pickle=True)
    create_embeddings(X_train, 'Graph/embeddings.tfr', 0)
    create_similarity_graph()
    next_record_id = create_records(X_train, Y_train,'Graph/data/train_data.tfr', 0)
    create_records(X_test,Y_test, 'Graph/data/test_data.tfr',next_record_id)
    create_augmented_data()
    train_dataset = make_dataset('Graph/data/augmented_train_data.tfr', True)
    test_dataset = make_dataset('Graph/data/test_data.tfr')
