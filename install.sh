# Download the images (used for both experiments & dataset generation)
wget https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip
unzip images_512.zip
mv images_512 imsitu_images

# Download the VASR dataset (used for the experiments)
wget https://my-vasr-bucket.s3.eu-west-1.amazonaws.com/vasr_dataset.zip
unzip vasr_dataset.zip
mkdir experiments/data
mv vasr_dataset experiments/data

# Prepare the experiments output results
mkdir experiments/data/model_results
