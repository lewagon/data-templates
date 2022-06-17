# Simple Face Annotation with Streamlit, FastAPI and OpenCV

This is a boilerplate for any projects that involve sending an image from a web UI to an API, performing some manipulation on the image, and sending it back. The example taken here is a simple face recognition app using `OpenCV`s built-in [Haar Cascade object detection algorithm](https://pyimagesearch.com/2021/04/12/opencv-haar-cascades/).

### What's here:

* [Streamlit](https://docs.streamlit.io/) on the frontend
* [FastAPI](https://fastapi.tiangolo.com/) on the backend
* [PIL/pillow](https://pillow.readthedocs.io/en/stable/) and [opencv-python](https://github.com/opencv/opencv-python) for working with images
* Backend and frontend can be deployed with Docker

### Using this template

> From inside the `sending-images-streamlit-fastapi` folder

1. You can serve the API with `uvicorn backend.api:app --reload` (default port is `8000`)
2. You can serve the frontend with `streamlit run frontend/app.py` (default port is `8501`)

### Using this template with Docker

Both the `frontend` and `backend` have corresponding `Dockerfile`s for the web UI and API.

1. To create a Docker image, inside the corresponding folders run `docker built -t NAME_FOR_THE_IMAGE .`
2. Run a container for either API or UI with `docker run -p MACHINE_PORT:CONTAINER_PORT NAME_FOR_THE_IMAGE`;

  Here, `MACHINE_PORT` is the `localhost` port you want to link to the container, while `CONTAINER_PORT` is the port which will be used by the running app in the container.


3. ❗ You won't be able to reach the API container through `localhost`; You'll need to [link](https://docs.docker.com/network/links/) the containers:

  * **API:** `docker run -p 8000:8000 NAME_FOR_THE_API_IMAGE --name api`
  * **UI:** `docker run -p 8501:8501 --link api:api NAME_FOR_THE_FE_IMAGE`

  This way you can use `api` instead of `localhost` to reach the API container from the frontend

  ❗ Note that Docker docs mention that `--link` might be removed in the future (as of 2022.06). Alternatives can be [user-defined bridges](https://docs.docker.com/network/bridge/#differences-between-user-defined-bridges-and-the-default-bridge) or [Docker Compose](https://docs.docker.com/compose/)

  Have fun!


