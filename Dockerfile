# Use a lightweight Python base image
FROM python:3.10-slim

# Hugging Face strictly requires a non-root user for Docker Spaces
RUN useradd -m -u 1000 user
USER user

# Set up the working directory and PATH
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install the exact dependencies your script uses
# (We install directly here so you don't even need a requirements.txt)
RUN pip install --no-cache-dir fastapi uvicorn openai requests pyyaml pydantic

# Copy all your files into the container with the correct permissions
COPY --chown=user . $HOME/app/

# Expose the required Hugging Face port
EXPOSE 7860

# Start the FastAPI server located in server/app.py
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]