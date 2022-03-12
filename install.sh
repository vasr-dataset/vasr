curl https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar > resized.tar
tar -xvf resized.tar
mv of500_images_resized experiments/images

mkdir experiments/data/model_results
