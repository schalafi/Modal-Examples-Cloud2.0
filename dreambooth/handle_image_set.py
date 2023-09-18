import os 

import logging
import boto3
from botocore.exceptions import ClientError
"""
FROM dir_name (local) GENERTAE the instance_example_urls.txt
From the dir_name get the images
to upload to S3 and then generate the urls

Write the urls to a file called instance_example_urls.txt
instance_example_urls.txt will be used by the app to retrieve the images
then train dreambooth
"""

def create_presigned_url(bucket_name, object_name, expiration=3600):
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    # Generate a presigned URL for the S3 object
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        print("Error: ",e)
        return None

    # The response contains the presigned URL
    return response

# Create a boto3 S3 client
s3 = boto3.client('s3')
MY_BUCKET = 'my-dreambooth'#Use your bucket name
URLS_FILE = 'instance_example_urls.txt'

# Get the directory name where the images are located (these images are in your computer)
dir_name = 'image_set'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    
# Get the list of images in the directory
images = os.listdir(dir_name)

# Generate presigned URLs for each image
with open(URLS_FILE,'w') as f:
    for image in images:
        # Generate a presigned POST URL for the image
        presigned_url = create_presigned_url(
            bucket_name=MY_BUCKET,
            object_name=image,
            expiration=3600#3600seconds = 1 hour
            )
        print("Url: {} generated for image: {}".format(presigned_url, image))
        
        # Write the presigned URL to a file
        f.write(presigned_url + '\n')

# Upload the images to S3
for image in images:
    # Upload the image to S3 using the presigned URL
    s3.put_object(
        Bucket=MY_BUCKET,
        Key=image,
        Body=open(os.path.join(dir_name, image), 'rb')
    )
