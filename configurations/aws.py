import configparser
import subprocess
import os


def init_aws(config_file: str, cache=True):

    if cache:
        if os.path.exists('aws_cache'):
            print('Using previous setup for AWS')
            return True

    print('Using configurations at ' + config_file)

    config = configparser.ConfigParser()
    config.read(config_file)

    aws_access_key_id = config['aws']['user']
    aws_secret_access_key = config['aws']['pass']
    default_region = config['aws']['region']

    subprocess.call("aws configure set aws_access_key_id " + aws_access_key_id, shell=True)
    subprocess.call("aws configure set aws_secret_access_key " + aws_secret_access_key, shell=True)
    subprocess.call("aws configure set default.region " + default_region, shell=True)

    if cache:
        if os.path.exists('aws_cache'):
            return True
        else:
            with open('aws_cache', 'a'):
                os.utime('aws_cache', None)

    return True
