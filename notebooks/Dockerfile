#  official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Install any needed packages specified in Pipfile
RUN pip install pipenv
RUN pipenv install --deploy --ignore-pipfile

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Define environment variable
ENV NAME World

# Run jupyter notebook when the container launches
CMD ["pipenv", "run", "jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]
