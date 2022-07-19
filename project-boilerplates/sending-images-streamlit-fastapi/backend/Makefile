##### Prediction API - - - - - - - - - - - - - - - - - - - - - - - - -

run_api:
	uvicorn fast_api.api:app --reload

##### Docker - - - - - - - - - - - - - - - - - - - - - - - - -

docker_build:
	docker build -t template-image-api .

docker_run:
	docker run -p 8000:8000 --name api template-image-api

##### GCP - - - - - - - - - - - - - - - - - - - - - - - - -

GCP_PROJECT_ID=XXX

DOCKER_IMAGE_NAME=XXX

# https://cloud.google.com/storage/docs/locations#location-mr
GCR_MULTI_REGION=XXX

# https://cloud.google.com/compute/docs/regions-zones#available
REGION=XXX

build_gcr_image:
	docker build -t $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME) .

build_gcr_image_m1:
	docker build --platform linux/amd64 -t $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME) .

run_gcr_image:
	docker run -e PORT=8000 -p 8080:8000 $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME)

push_gcr_image:
	docker push $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME)

gcr_deploy:
	gcloud run deploy --image $(GCR_MULTI_REGION)/$(GCP_PROJECT_ID)/$(DOCKER_IMAGE_NAME) --platform managed --region $(REGION)
