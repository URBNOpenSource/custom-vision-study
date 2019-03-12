#!/bin/bash
# WARNING. THIS DIDNT WORK DUE TO UNKNOWN ISSUES. WORKED WHEN RAN MANUALLY WITH KEY ON COMMAND LINE
# REPLACE {PATH_TO_ZIP} with where the zip file lives
# Also add any additional datasets you created by copy/pasting

curl -X POST -H "Authorization: Bearer $SALESFORCE_KEY" -H "Cache-Control: no-cache" -H "Content-Type: multipart/form-data" -F "path={PATH_TO_ZIP}/cifar10_20p.zip" -F "type=image"  https://api.einstein.ai/v2/vision/datasets/upload

curl -X POST -H "Authorization: Bearer $SALESFORCE_KEY" -H "Cache-Control: no-cache" -H "Content-Type: multipart/form-data" -F "path={PATH_TO_ZIP}/fashion_mnist_10p.zip" -F "type=image"  https://api.einstein.ai/v2/vision/datasets/upload

curl -X POST -H "Authorization: Bearer $SALESFORCE_KEY" -H "Cache-Control: no-cache" -H "Content-Type: multipart/form-data" -F "path={PATH_TO_ZIP}/uo_dress.zip" -F "type=image"  https://api.einstein.ai/v2/vision/datasets/upload
