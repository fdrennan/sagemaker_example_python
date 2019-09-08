# Exmaple from https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/linear_learner_mnist/linear_learner_mnist.ipynb
import configurations.aws as aws

import numpy as np
import pandas as pd
import boto3, pickle, gzip, urllib.request, os, io
import sagemaker

from sagemaker import get_execution_role
from sagemaker.session import Session
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer, json_deserializer

import sagemaker.amazon.common as smac

aws.init_aws(config_file='config.ini', cache=True)

bucket = Session().default_bucket()
prefix = 'sagemaker/DEMO-linear-mnist'
# Define IAM role

print('Getting execution role')
role = get_execution_role()


# Load the dataset
print('Grabbing dataset http://deeplearning.net/data/mnist/mnist.pkl.gz')
urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", "mnist.pkl.gz")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')


print('Structuring vector data')
vectors = np.array([t.tolist() for t in train_set[0]]).astype('float32')
print('Structuring label data')
labels = np.where(np.array([t.tolist() for t in train_set[1]]) == 0, 1, 0).astype('float32')
print('Data converted to float and restructured')
buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, vectors, labels)
buf.seek(0)


key = 'recordio-pb-data'
print('Uploading data to ' + 's3://{}/{}/train/{}'.format(bucket, prefix, key))

boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)
print('uploaded training data location: {}'.format(s3_train_data))

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('training artifacts will be uploaded to: {}'.format(output_location))


print('Getting linear learner container')
container = get_image_uri(boto3.Session().region_name, 'linear-learner')

print('Creating sagemaker session')
sess = sagemaker.Session()

print('Building estimator')
linear = sagemaker.estimator.Estimator(container,
                                       role,
                                       train_instance_count=1,
                                       train_instance_type='ml.c4.xlarge',
                                       output_path=output_location,
                                       sagemaker_session=sess)

print('Setting Hyperparameters')
linear.set_hyperparameters(feature_dim=784,
                           predictor_type='binary_classifier',
                           mini_batch_size=200)

print('Training model')
linear.fit({'train': s3_train_data})


print('Deploying model')
linear_predictor = linear.deploy(initial_instance_count=1,
                                 instance_type='ml.m4.xlarge')


linear_predictor.content_type = 'text/csv'
linear_predictor.serializer = csv_serializer
linear_predictor.deserializer = json_deserializer

print('Testing deployment')
result = linear_predictor.predict(train_set[0][30:31])
print(result)


print('Testing multiple predictions')
predictions = []
for array in np.array_split(test_set[0], 100):
    result = linear_predictor.predict(array)
    predictions += [r['predicted_label'] for r in result['predictions']]

predictions = np.array(predictions)

print('Creating crosstab')
result = pd.crosstab(np.where(test_set[1] == 0, 1, 0), predictions, rownames=['actuals'], colnames=['predictions'])
print(result)

print('Deleting endpoint')
sagemaker.Session().delete_endpoint(linear_predictor.endpoint)