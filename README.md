# Custom LightGBM ML block examples for Edge Impulse

This repository is an example on how to [add a custom learning block](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks) to Edge Impulse. This repository contains a LightGBM classifier and a LightGBM regression model.

As a primer, read the [Custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks) page in the Edge Impulse docs.

## Running the pipeline

You run this pipeline via Docker. This encapsulates all dependencies and packages for you.

### Running via Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16.0 or higher.
3. We need an Edge Impulse project with some data. Preferably create a new one and upload your own data, or alternatively:

    **Classifier**

    Clone a classification project, e.g. [Tutorial: continuous motion recognition](https://studio.edgeimpulse.com/public/14299/latest)

    **Regression**

    Clone a regression project, e.g. [Tutorial: temperature regression](https://studio.edgeimpulse.com/public/17972/latest)

4. If you've created a new project, then under **Create impulse** add a processing block, and either a Classification or Regression block (depending on your data).
5. Open a command prompt or terminal window.
6. Initialize the block:

    **Classifier**

    ```
    cd classifier
    $ edge-impulse-blocks init
    # Answer the questions:
    # ? Choose a type of block: "Machine learning block"
    # ? Choose an option: "Create a new block"
    # ? Enter the name of your block: "LightGBM classifier"
    # ? What type of data does this model operate on? "Classification"
    # ? Where can your model train? "Both CPU or GPU (default)"
    ```

    **Regression**

    ```
    cd regression
    $ edge-impulse-blocks init
    # Answer the questions:
    # ? Choose a type of block: "Machine learning block"
    # ? Choose an option: "Create a new block"
    # ? Enter the name of your block: "LightGBM regression"
    # ? What type of data does this model operate on? "Regression"
    # ? Where can your model train? "Both CPU or GPU (default)"
    ```


7. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

8. Build the container:

    **Classifier**

    ```
    $ cd classifier
    $ docker build -t lightgbm-classifier .
    ```

    **Regression**

    ```
    $ cd regression
    $ docker build -t lightgbm-regression .
    ```

9. Run the container to test the script (you don't need to rebuild the container if you make changes):

    **Classifier**

    ```
    $ docker run --rm -v $PWD:/app lightgbm-classifier --data-directory /app/data --out-directory /app/out
    ```

    **Regression**

    ```
    $ docker run --rm -v $PWD:/app lightgbm-regression --data-directory /app/data --out-directory /app/out
    ```

10. This creates a `model.txt` file in the out directory.

#### Adding extra dependencies

If you have extra packages that you want to install within the container, add them to `requirements.txt` and rebuild the container.

#### Adding new arguments

To add new arguments, see [Custom learning blocks > Arguments to your script](https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/adding-custom-learning-blocks#arguments-to-your-script).

## Fetching new data

To get up-to-date data from your project:

1. Install the [Edge Impulse CLI](https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-installation) v1.16 or higher.
2. Open a command prompt or terminal window.
3. Fetch new data via:

    ```
    $ edge-impulse-blocks runner --download-data data/
    ```

## Pushing the block back to Edge Impulse

You can also push this block back to Edge Impulse, that makes it available like any other ML block so you can retrain your model when new data comes in, or deploy the model to device. See [Docs > Adding custom learning blocks](https://docs.edgeimpulse.com/docs/edge-impulse-studio/organizations/adding-custom-transfer-learning-models) for more information.

1. Push the block:

    ```
    $ edge-impulse-blocks push
    ```

2. The block is now available under any of your projects via **Create impulse > Add new learning block**.
